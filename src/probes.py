"""Probe fitting primitives: exact ridge over an alpha path, and an MLP probe.

Ridge is solved through one eigendecomposition of the (centred) train Gram
matrix, which gives every alpha for the price of one -- numerically identical to
`sklearn.linear_model.Ridge(alpha, solver='cholesky')` (asserted in
`tests/test_ridge_matches_sklearn.py`) but ~15x cheaper for the 4096-d `e_t` tap,
where a naive per-alpha refit dominates the whole study's runtime.

Both probes see frozen features only: no gradient reaches the world model.
"""

import numpy as np
import torch
import torch.nn as nn

ALPHAS = np.logspace(-2, 9, 23)
GRAM_CHUNK = 8192  # rows per float32 matmul before accumulating into float64


def r2_score(y_true, y_pred):
    """Coefficient of determination against the *evaluation set's* own mean."""
    y_true = np.asarray(y_true, np.float64)
    y_pred = np.asarray(y_pred, np.float64)
    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    sst = np.sum((y_true - y_true.mean(0)) ** 2, axis=0)
    return 1.0 - sse / np.maximum(sst, 1e-12)


class Standardizer:
    def __init__(self, X):
        self.mean = X.mean(0, keepdims=True)
        self.std = X.std(0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, X):
        return (X - self.mean) / self.std


def ridge_fit_path(Xtr, ytr, alphas=ALPHAS):
    """Return {alpha: (w, b)} for centred ridge regression, one eigendecomposition.

    Xtr (n,d) float32, ytr (n,) or (n,k) float32.
    """
    Xtr = np.asarray(Xtr, np.float32)
    ytr = np.asarray(ytr, np.float32)
    if ytr.ndim == 1:
        ytr = ytr[:, None]
    xmean = Xtr.mean(0, keepdims=True)
    ymean = ytr.mean(0, keepdims=True)
    Xc = Xtr - xmean
    yc = ytr - ymean
    # accumulate the Gram in float64 over float32 blocks: summing 4e4 products of
    # O(1) activations in pure float32 costs ~3 significant digits
    d = Xc.shape[1]
    G = np.zeros((d, d), np.float64)
    b = np.zeros((d, yc.shape[1]), np.float64)
    for i in range(0, len(Xc), GRAM_CHUNK):
        blk = Xc[i:i + GRAM_CHUNK]
        G += (blk.T @ blk).astype(np.float64)
        b += (blk.T @ yc[i:i + GRAM_CHUNK]).astype(np.float64)
    evals, V = np.linalg.eigh(G)
    Vb = V.T @ b
    out = {}
    for a in alphas:
        w = V @ (Vb / (evals + a)[:, None])
        out[float(a)] = (w.astype(np.float32), (ymean - xmean @ w.astype(np.float32)))
    return out


def ridge_predict(X, wb):
    w, b = wb
    return np.asarray(X, np.float32) @ w + b


def fit_ridge_cv(Xtr, Ytr, Xval, Yval, Xte, Yte, alphas=ALPHAS, refit_on_trainval=True):
    """Multi-target ridge: alpha chosen per target on held-out *episodes*.

    Targets sharing a design matrix are fitted together -- the Gram matrix and
    its eigendecomposition (the expensive part at 4096 dims) are shared.

    Returns (r2_test, r2_val, best_alpha), each an array of length n_targets.
    """
    Ytr, Yval, Yte = [np.atleast_2d(np.asarray(y, np.float32).T).T if y.ndim == 1
                      else np.asarray(y, np.float32) for y in (Ytr, Yval, Yte)]
    k = Ytr.shape[1]
    path = ridge_fit_path(Xtr, Ytr, alphas)
    val_scores = np.stack([r2_score(Yval, ridge_predict(Xval, path[float(a)]))
                           for a in alphas])                     # (n_alpha, k)
    best_idx = val_scores.argmax(0)
    best_alpha = np.asarray([float(alphas[i]) for i in best_idx])

    if refit_on_trainval:
        Xall = np.concatenate([Xtr, Xval], 0)
        Yall = np.concatenate([Ytr, Yval], 0)
        path_all = ridge_fit_path(Xall, Yall, np.unique(best_alpha))
    else:
        path_all = path
    pred = np.zeros_like(Yte)
    for j in range(k):
        w, b = path_all[float(best_alpha[j])]
        pred[:, j] = (np.asarray(Xte, np.float32) @ w[:, j]) + b[0, j]
    return r2_score(Yte, pred), val_scores[best_idx, np.arange(k)], best_alpha


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=256, layers=2, out_dim=1):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, Xval, yval, Xte, yte, hidden=256, layers=2, epochs=40,
            batch_size=512, lr=1e-3, weight_decay=1e-4, device="cuda:0", seed=0,
            patience=8, verbose=False):
    """2x256 ReLU probe on frozen features; early-stopped on held-out episodes."""
    torch.manual_seed(seed)
    std = Standardizer(np.asarray(Xtr, np.float32))
    ymu, ysig = float(np.mean(ytr)), float(np.std(ytr) + 1e-8)

    def prep(X, y):
        return (torch.as_tensor(std(np.asarray(X, np.float32)), device=device),
                torch.as_tensor((np.asarray(y, np.float32) - ymu) / ysig, device=device)[:, None])

    Xtr_t, ytr_t = prep(Xtr, ytr)
    Xval_t, _ = prep(Xval, yval)
    Xte_t, _ = prep(Xte, yte)

    model = MLPProbe(Xtr_t.shape[1], hidden, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(Xtr_t)
    best_val, best_state, bad = -np.inf, None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xval_t).squeeze(-1).cpu().numpy() * ysig + ymu
        val = float(r2_score(yval[:, None], pv[:, None])[0])
        if val > best_val + 1e-4:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"    mlp epoch {ep} val R2 {val:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(Xte_t).squeeze(-1).cpu().numpy() * ysig + ymu
    return float(r2_score(yte[:, None], pt[:, None])[0]), best_val
