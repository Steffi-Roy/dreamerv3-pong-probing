"""Fit linear (ridge) and MLP probes on every (model x tap x factor) combination.

Reports test R^2 on held-out *episodes*. The linear->MLP gap is the
"present but entangled" signal (prompt.md 4.6).

Ball factors are evaluated only on frames where the ball is on screen: Pong
writes (ball_x, ball_y) = (0, 0) during the serve pause, and those frames encode
"no ball", not a position.

  python src/run_probes.py --data data/probe_pong_seed0.h5 \
      --latents latents/seed0 --out results/probe_results_seed0.csv
"""

import argparse
import pathlib
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import BALL_FACTORS, PADDLE_FACTORS, POSITION_FACTORS  # noqa: E402
from probes import ALPHAS, fit_ridge_cv, fit_mlp  # noqa: E402
from wm_utils import TAPS  # noqa: E402

VELOCITY_FACTORS = ("ball_vx", "ball_vy")
MAX_BALL_STEP = 20  # |delta| above this is a between-points reset, not motion


def load_targets(data_path, include_velocity=True):
    """Factor matrix, per-factor validity mask, and per-frame split labels."""
    with h5py.File(data_path, "r") as f:
        names = [s.decode() for s in f.attrs["factors"]]
        lab = f["labels"][:].astype(np.float32)
        ep_id = f["episode_id"][:]
        ep_split = np.array([s.decode() for s in f["ep_split"][:]])
        ep_start, ep_len = f["ep_start"][:], f["ep_length"][:]
    col = {n: i for i, n in enumerate(names)}
    split = ep_split[ep_id]

    ball_ok = (lab[:, col["ball_x"]] > 0) & (lab[:, col["ball_y"]] > 0)
    targets, masks = {}, {}
    for fct in POSITION_FACTORS:
        targets[fct] = lab[:, col[fct]]
        masks[fct] = ball_ok.copy() if fct in BALL_FACTORS else np.ones(len(lab), bool)

    if include_velocity:
        prev_ok = np.zeros(len(lab), bool)
        for s, L in zip(ep_start, ep_len):
            prev_ok[s + 1:s + L] = True          # a previous frame exists in-episode
        for fct, src in (("ball_vx", "ball_x"), ("ball_vy", "ball_y")):
            v = np.zeros(len(lab), np.float32)
            v[1:] = lab[1:, col[src]] - lab[:-1, col[src]]
            ok = prev_ok & ball_ok & np.roll(ball_ok, 1) & (np.abs(v) <= MAX_BALL_STEP)
            targets[fct] = v
            masks[fct] = ok
    return targets, masks, split, ep_id


def model_label(path):
    m = re.search(r"latents_step(\d+)\.h5", path.name)
    if m:
        return f"step_{int(m.group(1))}", int(m.group(1))
    m = re.search(r"latents_random_init_seed(\d+)\.h5", path.name)
    if m:
        return f"random_init_s{m.group(1)}", -1
    return path.stem, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--latents", required=True, help="directory of latents_*.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--taps", nargs="*", default=list(TAPS))
    ap.add_argument("--factors", nargs="*", default=None)
    ap.add_argument("--no_velocity", action="store_true")
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--mlp_epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca_dim", type=int, default=0,
                    help="PCA-reduce every tap to this width (matched-dimensionality control)")
    ap.add_argument("--run_name", default="seed0")
    args = ap.parse_args()

    targets, masks, split, _ = load_targets(args.data, include_velocity=not args.no_velocity)
    factors = args.factors or [f for f in list(POSITION_FACTORS) +
                               ([] if args.no_velocity else list(VELOCITY_FACTORS))]
    print("factors:", factors)
    for name, cnt in [("train", (split == "train").sum()), ("val", (split == "val").sum()),
                      ("test", (split == "test").sum())]:
        print(f"  {name} frames: {cnt}")

    latent_files = sorted(pathlib.Path(args.latents).glob("latents_*.h5"))
    print("latent files:", [p.name for p in latent_files])
    rows = []
    for lf in latent_files:
        model, step = model_label(lf)
        with h5py.File(lf, "r") as f:
            available = [t for t in args.taps if t in f]
            for tap in available:
                t0 = time.time()
                X = f[tap][:].astype(np.float32)
                if args.pca_dim and X.shape[1] > args.pca_dim:
                    # matched-dimensionality control: taps differ in width
                    # (e_t 4096 vs h_t 512), and wider is easier to decode linearly
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=args.pca_dim, random_state=0)
                    pca.fit(X[split == "train"][::4])
                    X = pca.transform(X).astype(np.float32)
                dim = X.shape[1]
                # group factors by validity mask so the ridge Gram is shared
                groups = {}
                for fct in factors:
                    groups.setdefault(masks[fct].tobytes(), []).append(fct)
                for _, gfactors in groups.items():
                    m = masks[gfactors[0]]
                    idx = {s: np.nonzero(m & (split == s))[0] for s in ("train", "val", "test")}
                    Y = np.stack([targets[fct] for fct in gfactors], 1)
                    Xtr, Xval, Xte = (X[idx["train"]], X[idx["val"]], X[idx["test"]])
                    Ytr, Yval, Yte = (Y[idx["train"]], Y[idx["val"]], Y[idx["test"]])
                    r2, r2v, alpha = fit_ridge_cv(Xtr, Ytr, Xval, Yval, Xte, Yte, ALPHAS)
                    for j, fct in enumerate(gfactors):
                        rows.append(dict(run=args.run_name, model=model, step=step, tap=tap,
                                         dim=dim, factor=fct, probe="linear",
                                         r2_test=float(r2[j]), r2_val=float(r2v[j]),
                                         alpha=float(alpha[j]), n_train=len(idx["train"]),
                                         n_test=len(idx["test"])))
                    if not args.no_mlp:
                        for j, fct in enumerate(gfactors):
                            r2m, r2mv = fit_mlp(Xtr, Ytr[:, j], Xval, Yval[:, j], Xte, Yte[:, j],
                                                epochs=args.mlp_epochs, device=args.device,
                                                seed=args.seed)
                            rows.append(dict(run=args.run_name, model=model, step=step, tap=tap,
                                             dim=dim, factor=fct, probe="mlp",
                                             r2_test=float(r2m), r2_val=float(r2mv),
                                             alpha=np.nan, n_train=len(idx["train"]),
                                             n_test=len(idx["test"])))
                del X
                done = [r for r in rows if r["model"] == model and r["tap"] == tap]
                summary = " ".join(f"{r['factor']}/{r['probe'][:3]}={r['r2_test']:.3f}"
                                   for r in done)
                print(f"[{model} {tap} d={dim}] {summary}  ({time.time()-t0:.0f}s)", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows)")

    piv = df[df.probe == "linear"].pivot_table(index=["model", "tap"], columns="factor",
                                               values="r2_test")
    print("\nlinear-probe test R^2:\n", piv.round(3).to_string())


if __name__ == "__main__":
    main()
