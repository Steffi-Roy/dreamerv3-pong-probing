"""Build Table 1 and Figures 1-2 for the report (prompt.md 5).

  Table 1  test R^2 per factor x tap x probe, with the random-init floor and the
           pixel-CNN ceiling  -> results/table1.{md,tex}
  Figure 1 mechanism: input / reconstruction / per-pixel error triptych plus the
           global-vs-ball-region PSNR bars -> figures/fig1_mechanism.png
  Figure 2 progression: R^2 vs training checkpoint for ball and paddle at each
           tap -> figures/fig2_progression.png
  Figure 3 (extra) linear->MLP gap per tap -> figures/fig3_linear_mlp_gap.png

  python src/make_figures.py --probes results/probe_results_seed0.csv \
      --pixel_cnn results/pixel_cnn_seed0.csv --recon_npz results/recon_frames_seed0.npz \
      --recon_csv results/recon_seed0.csv --outdir figures --tabledir results
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAP_ORDER = ["e_t", "z_t", "h_t", "s_t"]
TAP_LABEL = {"e_t": r"$e_t$ (encoder)", "z_t": r"$z_t$ (stoch)",
             "h_t": r"$h_t$ (deter)", "s_t": r"$s_t$ (feat)", "z_prob": r"$z_t$ probs"}
FACTORS = ["player_y", "enemy_y", "ball_x", "ball_y"]
FACTOR_LABEL = {"player_y": "player$_y$", "enemy_y": "enemy$_y$",
                "ball_x": "ball$_x$", "ball_y": "ball$_y$",
                "ball_vx": "ball$_{vx}$", "ball_vy": "ball$_{vy}$"}
PADDLE_C, BALL_C = "#2e6f9e", "#c0392b"


def load(probes, pixel_cnn=None):
    df = pd.read_csv(probes)
    if pixel_cnn and pathlib.Path(pixel_cnn).exists():
        df = pd.concat([df, pd.read_csv(pixel_cnn)], ignore_index=True)
    df["is_random"] = df.model.str.startswith("random_init")
    df["is_pixel"] = df.model == "pixel_cnn"
    return df


def final_step(df):
    trained = df[~df.is_random & ~df.is_pixel]
    return int(trained.step.max()) if len(trained) else -1


def table1(df, tabledir, factors=FACTORS):
    step = final_step(df)
    rows = []

    def collect(sub, label):
        for probe in ("linear", "mlp", "cnn"):
            s = sub[sub.probe == probe]
            if not len(s):
                continue
            for tap in [t for t in TAP_ORDER + ["pixels"] if t in set(s.tap)]:
                st = s[s.tap == tap]
                r = {"model": label, "tap": tap, "probe": probe,
                     "dim": int(st.dim.iloc[0])}
                for f in factors:
                    v = st[st.factor == f].r2_test
                    r[f] = float(v.mean()) if len(v) else np.nan
                rows.append(r)

    collect(df[(df.step == step) & ~df.is_random & ~df.is_pixel], f"trained (step {step})")
    collect(df[df.is_random], "random-init floor")
    collect(df[df.is_pixel], "pixel CNN (ceiling)")
    tab = pd.DataFrame(rows)

    tabledir = pathlib.Path(tabledir)
    tabledir.mkdir(parents=True, exist_ok=True)
    (tabledir / "table1.md").write_text(tab.round(3).to_markdown(index=False))

    lines = [r"\begin{tabular}{llr" + "r" * len(factors) + "}", r"\toprule",
             "model & tap & dim & " + " & ".join(FACTOR_LABEL.get(f, f) for f in factors)
             + r" \\", r"\midrule"]
    last = None
    for _, r in tab.iterrows():
        if last is not None and r["model"] != last:
            lines.append(r"\midrule")
        last = r["model"]
        cells = " & ".join("--" if np.isnan(r[f]) else f"{r[f]:.3f}" for f in factors)
        lines.append(f"{r['model']} ({r['probe']}) & {r['tap'].replace('_t','$_t$')} "
                     f"& {r['dim']} & {cells} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tabledir / "table1.tex").write_text("\n".join(lines))
    print(tab.round(3).to_string(index=False))
    print(f"wrote {tabledir}/table1.md and table1.tex")
    return tab


def fig1(recon_npz, recon_csv, outdir):
    d = np.load(recon_npz)
    gt, rc = d["examples_gt"], d["examples_recon"]
    err_map = d["error_map"]
    ball_xy = d["examples_ball_xy"]
    player_xy, enemy_xy = d["examples_player_xy"], d["examples_enemy_xy"]
    rec = pd.read_csv(recon_csv)
    row = rec[rec.scope == "test"].iloc[0]

    fig = plt.figure(figsize=(7.4, 3.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.45], wspace=0.5, hspace=0.15)

    def boxes(ax, i, ball_ec, paddle_ec):
        cx, cy = ball_xy[i]
        ax.add_patch(plt.Rectangle((cx - 3.5, cy - 3.5), 7, 7, fill=False, ec=ball_ec, lw=1.0))
        for arr in (player_xy, enemy_xy):
            qx, qy = arr[i]
            ax.add_patch(plt.Rectangle((qx - 2.5, qy - 4.5), 5, 9, fill=False,
                                       ec=paddle_ec, lw=0.9, ls="--"))
    for r in range(2):
        i = r
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(gt[i]); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("input", fontsize=9)
        ax.set_ylabel(f"frame {i+1}", fontsize=8)
        boxes(ax, i, BALL_C, PADDLE_C)

        ax = fig.add_subplot(gs[r, 1]); ax.imshow(rc[i]); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("reconstruction", fontsize=9)
        boxes(ax, i, BALL_C, PADDLE_C)

        ax = fig.add_subplot(gs[r, 2])
        e = np.mean((gt[i].astype(float) - rc[i].astype(float)) ** 2, -1)
        im = ax.imshow(e, cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("squared error", fontsize=9)
        boxes(ax, i, "w", "#7fb3d5")

    ax = fig.add_subplot(gs[:, 3])
    names = ["global", "bg", "paddle", "ball"]
    vals = [row.psnr_global, row.psnr_background, row.psnr_paddle,
            row.get("psnr_ball_tight", row.psnr_ball)]
    ax.bar(names, vals, color=["#7f8c8d", "#bdc3c7", PADDLE_C, BALL_C])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylabel("PSNR (dB)", fontsize=9)
    ax.set_title(f"SSIM$_{{global}}$={row.ssim_global:.3f}", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xticklabels(names, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig1_mechanism.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / "fig1_mechanism.pdf", bbox_inches="tight")
    print(f"wrote {outdir}/fig1_mechanism.png")

    # mean error map over all reconstructed frames
    fig2_, ax = plt.subplots(figsize=(2.6, 2.6))
    im = ax.imshow(err_map, cmap="magma")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("mean per-pixel error", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig2_.savefig(outdir / "fig1b_error_map.png", dpi=220, bbox_inches="tight")
    print(f"wrote {outdir}/fig1b_error_map.png")


def fig2(df, outdir, probe="linear"):
    d = df[(df.probe == probe) & ~df.is_pixel]
    steps = sorted(d[~d.is_random].step.unique())
    taps = [t for t in TAP_ORDER if t in set(d.tap)]
    fig, axes = plt.subplots(1, len(taps), figsize=(7.0, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, tap in zip(axes, taps):
        for factors, color, lab in ((["player_y", "enemy_y"], PADDLE_C, "paddles"),
                                    (["ball_x", "ball_y"], BALL_C, "ball")):
            ys = [d[(d.tap == tap) & (d.step == s) & d.factor.isin(factors)].r2_test.mean()
                  for s in steps]
            ax.plot(np.array(steps) / 1000, ys, "-o", ms=3, color=color, label=lab)
            fl = d[d.is_random & (d.tap == tap) & d.factor.isin(factors)].r2_test.mean()
            if not np.isnan(fl):
                ax.axhline(fl, color=color, ls=":", lw=1.0)
        ax.set_title(TAP_LABEL.get(tap, tap), fontsize=9)
        ax.set_xlabel("policy steps (k)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"test $R^2$ ({probe})", fontsize=9)
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.text(0.5, -0.10, "dotted = random-init floor at the same tap and dimensionality",
             ha="center", fontsize=7)
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig2_progression.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / "fig2_progression.pdf", bbox_inches="tight")
    print(f"wrote {outdir}/fig2_progression.png")


def fig3(df, outdir):
    step = final_step(df)
    d = df[(df.step == step) & ~df.is_random & ~df.is_pixel]
    taps = [t for t in TAP_ORDER if t in set(d.tap)]
    factors = [f for f in FACTORS if f in set(d.factor)]
    x = np.arange(len(taps))
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    w = 0.8 / len(factors)
    for i, f in enumerate(factors):
        gaps = []
        for tap in taps:
            s = d[(d.tap == tap) & (d.factor == f)]
            lin = s[s.probe == "linear"].r2_test.mean()
            mlp = s[s.probe == "mlp"].r2_test.mean()
            gaps.append(mlp - lin)
        color = BALL_C if f.startswith("ball") else PADDLE_C
        ax.bar(x + i * w - 0.4 + w / 2, gaps, w, label=FACTOR_LABEL.get(f, f),
               color=color, alpha=0.55 if i % 2 else 1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([TAP_LABEL.get(t, t) for t in taps], fontsize=8)
    ax.set_ylabel(r"MLP $R^2$ - linear $R^2$", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    outdir = pathlib.Path(outdir)
    fig.savefig(outdir / "fig3_linear_mlp_gap.png", dpi=220, bbox_inches="tight")
    print(f"wrote {outdir}/fig3_linear_mlp_gap.png")


def fig4_training(metrics_path, outdir, action_repeat=4):
    """Eval/train return vs policy steps -- the 'is the agent non-trivial' check."""
    import json
    rows = [json.loads(l) for l in open(metrics_path)]
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for key, color, lab in (("train_return", "#95a5a6", "train"),
                            ("eval_return", "#2e6f9e", "eval")):
        pts = [(r["step"] / action_repeat / 1000, r[key]) for r in rows if key in r]
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "-o" if key == "eval_return" else "-", ms=3, lw=1.2,
                color=color, alpha=0.9 if key == "eval_return" else 0.6, label=lab)
    ax.axhline(-21, color="k", ls=":", lw=0.8)
    ax.text(0.02, 0.06, "random policy floor (-21)", transform=ax.transAxes, fontsize=6)
    ax.set_xlabel("policy steps (k)", fontsize=8)
    ax.set_ylabel("episode return", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(outdir / "fig4_training_return.png", dpi=220, bbox_inches="tight")
    print(f"wrote {outdir}/fig4_training_return.png")


def fig5_recon_progress(recon_all_csv, outdir):
    """Region-wise reconstruction quality vs training step (the other half of Fig 2)."""
    df = pd.read_csv(recon_all_csv)
    df = df[df.scope == "test"].copy()
    df["step"] = df.model.str.extract(r"step_(\d+)").astype(float)
    df = df.dropna(subset=["step"]).sort_values("step")
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for col, color, lab in (("psnr_background", "#bdc3c7", "background"),
                            ("psnr_global", "#7f8c8d", "global"),
                            ("psnr_paddle", PADDLE_C, "paddle box"),
                            ("psnr_ball_tight", BALL_C, "ball box (3x3)")):
        if col in df:
            ax.plot(df.step / 1000, df[col], "-o", ms=3, color=color, label=lab)
    ax.set_xlabel("policy steps (k)", fontsize=8)
    ax.set_ylabel("PSNR (dB)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig5_recon_progress.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / "fig5_recon_progress.pdf", bbox_inches="tight")
    print(f"wrote {outdir}/fig5_recon_progress.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--pixel_cnn", default=None)
    ap.add_argument("--recon_npz", default=None)
    ap.add_argument("--recon_csv", default=None)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--tabledir", default="results")
    ap.add_argument("--metrics", default=None, help="runs/<run>/metrics.jsonl for the return curve")
    ap.add_argument("--recon_all", default=None, help="per-checkpoint reconstruction CSV")
    args = ap.parse_args()

    df = load(args.probes, args.pixel_cnn)
    table1(df, args.tabledir)
    fig2(df, args.outdir)
    fig3(df, args.outdir)
    if args.recon_npz and args.recon_csv:
        fig1(args.recon_npz, args.recon_csv, args.outdir)
    if args.metrics and pathlib.Path(args.metrics).exists():
        fig4_training(args.metrics, args.outdir)
    if args.recon_all and pathlib.Path(args.recon_all).exists():
        fig5_recon_progress(args.recon_all, args.outdir)


if __name__ == "__main__":
    main()
