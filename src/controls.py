"""The two controls the probe numbers are meaningless without (prompt.md 4.5).

FLOOR -- random-init world model. A 4096-d random projection of the frame is
already linearly decodable, so "R^2 is high at e_t" means nothing until you
compare against an *untrained* encoder at the same tap and the same
dimensionality. `--random_init K` dumps latents for K differently-seeded
untrained world models, which run_probes.py then treats like any checkpoint.

CEILING -- a small CNN trained directly on the 64x64 frame. If the CNN recovers
the ball but the latents do not, the information is in the pixels and the gap is
the encoder's doing, not the environment's.

  python src/controls.py --data data/probe_pong_seed0.h5 --logdir runs/pong_seed0 \
      --latents latents/seed0 --random_init 2 --cnn_probe --out results/pixel_cnn_seed0.csv
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import make_ari_env  # noqa: E402
from extract_latents import run_checkpoint  # noqa: E402
from probes import r2_score  # noqa: E402
from run_probes import load_targets  # noqa: E402
from wm_utils import build_world_model, load_config  # noqa: E402


class PixelCNN(nn.Module):
    """Small conv net on the raw 64x64 observation: the input-information reference."""

    def __init__(self, depth=32, out_dim=1):
        super().__init__()
        c = depth
        self.body = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1), nn.ReLU(),        # 32
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.ReLU(),    # 16
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.ReLU(),  # 8
            nn.Conv2d(4 * c, 8 * c, 4, 2, 1), nn.ReLU(),  # 4
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(8 * c * 16, 256), nn.ReLU(), nn.Linear(256, out_dim))

    def forward(self, x):
        return self.head(self.body(x))


def train_pixel_cnn(images, y, idx_tr, idx_val, idx_te, device="cuda:0", epochs=8,
                    batch_size=256, lr=3e-4, seed=0, patience=3, verbose=True):
    torch.manual_seed(seed)
    ymu, ysig = float(y[idx_tr].mean()), float(y[idx_tr].std() + 1e-8)
    model = PixelCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batch(idx):
        x = torch.as_tensor(images[np.sort(idx)], device=device).float().permute(0, 3, 1, 2) / 255.0
        t = torch.as_tensor((y[np.sort(idx)] - ymu) / ysig, device=device)[:, None]
        return x, t

    @torch.no_grad()
    def evaluate(idx):
        model.eval()
        preds = []
        for i in range(0, len(idx), 1024):
            x, _ = batch(idx[i:i + 1024])
            preds.append(model(x).squeeze(-1).cpu().numpy())
        return float(r2_score(y[np.sort(idx)][:, None],
                              (np.concatenate(preds) * ysig + ymu)[:, None])[0])

    best_val, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(seed + ep).permutation(idx_tr)
        for i in range(0, len(perm), batch_size):
            x, t = batch(perm[i:i + batch_size])
            loss = nn.functional.mse_loss(model(x), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = evaluate(idx_val)
        if verbose:
            print(f"    cnn epoch {ep} val R2 {val:.4f}", flush=True)
        if val > best_val + 1e-4:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return evaluate(idx_te), best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--random_init", type=int, default=0, help="number of random-init seeds")
    ap.add_argument("--cnn_probe", action="store_true")
    ap.add_argument("--out", default="results/pixel_cnn.csv")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_episodes", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--run_name", default="seed0")
    args = ap.parse_args()

    outdir = pathlib.Path(args.latents)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- floor: untrained world models ------------------------------------
    if args.random_init:
        config = load_config(args.logdir, device=args.device)
        env = make_ari_env(seed=0)
        for k in range(args.random_init):
            out_path = outdir / f"latents_random_init_seed{k}.h5"
            if out_path.exists():
                print(f"skip {out_path} (exists)")
                continue
            torch.manual_seed(1000 + k)
            np.random.seed(1000 + k)
            wm = build_world_model(config, env.observation_space, env.action_space,
                                   ckpt_path=None, device=args.device)
            print(f"[random-init {k}] extracting -> {out_path}", flush=True)
            run_checkpoint(wm, args.data, out_path, batch_episodes=args.batch_episodes,
                           save_recon=False, device=args.device, seed=k)
            del wm
            torch.cuda.empty_cache()

    # ---- ceiling: CNN on pixels -------------------------------------------
    if args.cnn_probe:
        targets, masks, split, _ = load_targets(args.data)
        with h5py.File(args.data, "r") as f:
            print("loading frames into RAM ...", flush=True)
            images = f["image"][:]
        rows = []
        for fct, y in targets.items():
            m = masks[fct]
            idx = {s: np.nonzero(m & (split == s))[0] for s in ("train", "val", "test")}
            t0 = time.time()
            r2, r2v = train_pixel_cnn(images, y.astype(np.float32), idx["train"], idx["val"],
                                      idx["test"], device=args.device, epochs=args.epochs)
            print(f"[pixel-cnn {fct}] test R2 {r2:.4f} (val {r2v:.4f}) {time.time()-t0:.0f}s",
                  flush=True)
            rows.append(dict(run=args.run_name, model="pixel_cnn", step=-2, tap="pixels",
                             dim=64 * 64 * 3, factor=fct, probe="cnn", r2_test=r2, r2_val=r2v,
                             alpha=np.nan, n_train=len(idx["train"]), n_test=len(idx["test"])))
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
