"""Reconstruction quality, globally and inside the ball / paddle boxes.

The mechanism claim (Balestriero & LeCun 2024) is that a pixel-reconstruction
loss spends capacity in proportion to pixel variance, so a 2x4-pixel ball is
cheap to ignore. The signature is a *high global* PSNR/SSIM next to a *poor
ball-region* PSNR -- which only means something if the paddle region, an object
of the same "kind" but ~8x the area, is reconstructed well.

Boxes are placed from the AtariARI labels, rescaled with the offsets measured by
check_ari.py. Frames where the ball is off screen are skipped for the ball box.

  python src/reconstruction_metrics.py --data data/probe_pong_seed0.h5 \
      --latents latents/seed0/latents_step0100000.h5 --out results/recon_seed0.csv
"""

import argparse
import pathlib
import sys

import h5py
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import ball_visible, paddle_center_to_obs, rescale_to_obs  # noqa: E402

# Object half-sizes on the 64x64 frame. Native Pong: ball 2x4 px, paddle 4x16 px,
# so after the 64/160 x 64/210 rescale the ball is under one pixel and the paddle
# about 1.6 x 4.9 -- the size asymmetry the hypothesis is about.
BALL_HALF = 3        # covers the ball plus its max-pool ghost from the previous frame
BALL_HALF_TIGHT = 1  # 3x3: the 7x7 box is ~98% background, which dilutes the error
PADDLE_HALF_X, PADDLE_HALF_Y = 2, 4


def box_mask(shape, cx, cy, hx, hy):
    m = np.zeros(shape, bool)
    x0, x1 = int(np.clip(cx - hx, 0, shape[1])), int(np.clip(cx + hx + 1, 0, shape[1]))
    y0, y1 = int(np.clip(cy - hy, 0, shape[0])), int(np.clip(cy + hy + 1, 0, shape[0]))
    m[y0:y1, x0:x1] = True
    return m


def mse_psnr(a, b, mask=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size == 0:
        return np.nan, np.nan
    mse = float(np.mean((a - b) ** 2))
    return mse, 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--latents", required=True, help="latents_*.h5 containing /recon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_frame_out", default=None, help="npz with per-frame values + example triptychs")
    ap.add_argument("--model", default=None, help="label for the results row")
    ap.add_argument("--run_name", default="seed0")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of overwriting (per-checkpoint sweep)")
    args = ap.parse_args()

    with h5py.File(args.latents, "r") as fl:
        assert "recon" in fl, f"{args.latents} has no /recon (re-run extract_latents.py --save_recon)"
        recon = fl["recon"][:]
        index = fl["recon_index"][:]     # position of each recon in the source file
    n = len(recon)
    with h5py.File(args.data, "r") as fd:
        names = [s.decode() for s in fd.attrs["factors"]]
        image = fd["image"][:][index]
        labels = fd["labels"][:][index].astype(np.float32)
        ep_id = fd["episode_id"][:][index]
        ep_split = np.array([s.decode() for s in fd["ep_split"][:]])
    col = {nm: i for i, nm in enumerate(names)}
    split = ep_split[ep_id]
    print(f"{n} reconstructed frames; splits: "
          f"{ {s: int((split == s).sum()) for s in ('train','val','test')} }")

    bx, by = rescale_to_obs(labels[:, col["ball_x"]], labels[:, col["ball_y"]])
    vis = ball_visible(labels[:, col["ball_x"]], labels[:, col["ball_y"]])
    px, py = paddle_center_to_obs("player", labels[:, col["player_y"]])
    ex, ey = paddle_center_to_obs("enemy", labels[:, col["enemy_y"]])

    per = {k: np.full(n, np.nan) for k in
           ("psnr_global", "ssim_global", "mse_global", "mse_ball", "psnr_ball",
            "mse_ball_tight", "psnr_ball_tight", "mse_paddle", "psnr_paddle",
            "mse_background", "psnr_background")}
    err_sum = np.zeros((64, 64), np.float64)  # mean per-pixel error map for Figure 1

    for i in range(n):
        gt, rc = image[i], recon[i]
        per["psnr_global"][i] = peak_signal_noise_ratio(gt, rc, data_range=255)
        per["ssim_global"][i] = structural_similarity(gt, rc, data_range=255, channel_axis=2)
        per["mse_global"][i] = float(np.mean((gt.astype(np.float64) - rc) ** 2))

        pm = box_mask((64, 64), px[i], py[i], PADDLE_HALF_X, PADDLE_HALF_Y)
        em = box_mask((64, 64), ex[i], ey[i], PADDLE_HALF_X, PADDLE_HALF_Y)
        paddle = pm | em
        per["mse_paddle"][i], per["psnr_paddle"][i] = mse_psnr(gt, rc, paddle)

        if vis[i]:
            bm = box_mask((64, 64), bx[i], by[i], BALL_HALF, BALL_HALF)
            per["mse_ball"][i], per["psnr_ball"][i] = mse_psnr(gt, rc, bm)
            bt = box_mask((64, 64), bx[i], by[i], BALL_HALF_TIGHT, BALL_HALF_TIGHT)
            per["mse_ball_tight"][i], per["psnr_ball_tight"][i] = mse_psnr(gt, rc, bt)
            bg = ~(bm | paddle)
        else:
            bg = ~paddle
        per["mse_background"][i], per["psnr_background"][i] = mse_psnr(gt, rc, bg)
        err_sum += np.mean((gt.astype(np.float64) - rc) ** 2, axis=-1)

    rows = []
    for scope in ("all", "test"):
        sel = np.ones(n, bool) if scope == "all" else (split == "test")
        row = dict(run=args.run_name, model=args.model or pathlib.Path(args.latents).stem,
                   scope=scope, n_frames=int(sel.sum()),
                   n_ball_visible=int((sel & vis).sum()))
        for k, v in per.items():
            row[k] = float(np.nanmean(v[sel]))
        row["psnr_gap_global_minus_ball"] = row["psnr_global"] - row["psnr_ball"]
        row["mse_ratio_ball_over_paddle"] = row["mse_ball"] / max(row["mse_paddle"], 1e-12)
        row["mse_ratio_balltight_over_paddle"] = (row["mse_ball_tight"]
                                                  / max(row["mse_paddle"], 1e-12))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"wrote {out}")

    if args.per_frame_out:
        pick = np.nonzero(vis & (split == "test"))[0][:8]
        np.savez_compressed(args.per_frame_out, error_map=err_sum / n,
                            examples_gt=image[pick], examples_recon=recon[pick],
                            examples_ball_xy=np.stack([bx[pick], by[pick]], 1),
                            examples_player_xy=np.stack([px[pick], py[pick]], 1),
                            examples_enemy_xy=np.stack([ex[pick], ey[pick]], 1),
                            **{k: v for k, v in per.items()})
        print(f"wrote {args.per_frame_out}")


if __name__ == "__main__":
    main()
