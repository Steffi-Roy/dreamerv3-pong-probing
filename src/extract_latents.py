"""Dump the four latent taps (and optionally decoder reconstructions) per checkpoint.

Taps (prompt.md 4.4):
  e_t = encoder(obs)            feedforward, current frame only
  z_t = post['stoch']           posterior stochastic latent (32 cat x 32 classes)
  h_t = post['deter']           recurrent state (temporal history)
  s_t = get_feat(post)          [z_t, h_t] -- what the decoder and actor see
plus z_prob = softmax(post['logit']), the sampling-noise-free version of z_t,
kept as a robustness check on the stochastic sample.

The world model is frozen (eval, requires_grad_(False)); sequences are replayed
episode-by-episode so `h_t` carries the same history it had online.

  python src/extract_latents.py --data data/probe_pong_seed0.h5 \
      --logdir runs/pong_seed0 --out latents/seed0 --save_recon
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ari_env import make_ari_env  # noqa: E402
from wm_utils import build_world_model, list_checkpoints, load_config  # noqa: E402


RECON_SKIP = 20  # frames to skip at episode start (serve pause: no ball on screen)


def episode_batches(f, batch_episodes):
    starts, lengths = f["ep_start"][:], f["ep_length"][:]
    order = np.arange(len(starts))
    for i in range(0, len(order), batch_episodes):
        yield order[i:i + batch_episodes], starts[order[i:i + batch_episodes]], \
            lengths[order[i:i + batch_episodes]]


@torch.no_grad()
def run_checkpoint(wm, data_path, out_path, batch_episodes=8, save_recon=False,
                   recon_max=6000, device="cuda:0", seed=0):
    """Dump taps for one frozen model; `recon_max` is a budget spread over episodes."""
    torch.manual_seed(seed)  # post['stoch'] is a sample; fix it for reproducibility
    fin = h5py.File(data_path, "r")
    n_total = fin["image"].shape[0]
    fout = h5py.File(out_path, "w")
    taps, recon_ds, n_recon = {}, None, 0
    n_episodes = len(fin["ep_start"])
    recon_per_episode = max(1, recon_max // max(1, n_episodes))
    t0 = time.time()

    for bi, (eps, starts, lengths) in enumerate(episode_batches(fin, batch_episodes)):
        T = int(lengths.max())
        B = len(eps)
        image = np.zeros((B, T, 64, 64, 3), np.uint8)
        action = np.zeros((B, T, fin["action"].shape[1]), np.float32)
        is_first = np.zeros((B, T), np.float32)
        for b, (s, L) in enumerate(zip(starts, lengths)):
            image[b, :L] = fin["image"][s:s + L]
            action[b, :L] = fin["action"][s:s + L]
            is_first[b, :L] = fin["is_first"][s:s + L]

        obs = {
            "image": torch.as_tensor(image, device=device),
            "action": torch.as_tensor(action, device=device),
            "is_first": torch.as_tensor(is_first, device=device),
            "is_terminal": torch.zeros((B, T), device=device),
        }
        obs = wm.preprocess(obs)
        embed = wm.encoder(obs)
        post, _ = wm.dynamics.observe(embed, obs["action"], obs["is_first"])
        stoch = post["stoch"]
        flat = lambda x: x.reshape(list(x.shape[:2]) + [-1])
        out = {
            "e_t": embed,
            "z_t": flat(stoch),
            "h_t": post["deter"],
            "s_t": wm.dynamics.get_feat(post),
            "z_prob": flat(torch.softmax(post["logit"], dim=-1)),
        }
        feat = out["s_t"]

        # unpad and append, keeping the flat frame order of the source file
        for name, tensor in out.items():
            arr = tensor.detach().cpu().numpy().astype(np.float16)
            rows = np.concatenate([arr[b, :L] for b, L in enumerate(lengths)], 0)
            if name not in taps:
                taps[name] = fout.create_dataset(
                    name, shape=(0, rows.shape[1]), maxshape=(None, rows.shape[1]),
                    dtype=np.float16, chunks=(min(256, max(1, rows.shape[0])), rows.shape[1]),
                    compression="lzf")
            d = taps[name]
            d.resize(d.shape[0] + len(rows), axis=0)
            d[-len(rows):] = rows

        if save_recon:
            # A slice from every episode (not the first N frames overall), so the
            # reconstruction metrics can be computed on held-out episodes too.
            # `recon_index` holds the frame's position in the source file.
            rec = wm.heads["decoder"](feat)["image"].mode()
            rec = (rec.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            keep, keep_idx = [], []
            for b, (s, L) in enumerate(zip(starts, lengths)):
                lo = min(RECON_SKIP, max(0, int(L) - 1))       # skip the serve pause
                hi = min(int(L), lo + recon_per_episode)
                keep.append(rec[b, lo:hi])
                keep_idx.append(np.arange(s + lo, s + hi))
            rows = np.concatenate(keep, 0)
            idx = np.concatenate(keep_idx, 0)
            if recon_ds is None:
                recon_ds = fout.create_dataset(
                    "recon", shape=(0, 64, 64, 3), maxshape=(None, 64, 64, 3),
                    dtype=np.uint8, chunks=(64, 64, 64, 3), compression="lzf")
                index_ds = fout.create_dataset("recon_index", shape=(0,), maxshape=(None,),
                                               dtype=np.int64)
            else:
                index_ds = fout["recon_index"]
            recon_ds.resize(n_recon + len(rows), axis=0)
            recon_ds[n_recon:] = rows
            index_ds.resize(n_recon + len(idx), axis=0)
            index_ds[n_recon:] = idx
            n_recon += len(rows)

        if (bi + 1) % 5 == 0:
            print(f"  batch {bi+1}: {taps['e_t'].shape[0]}/{n_total} frames "
                  f"({time.time()-t0:.0f}s)", flush=True)

    fout.attrs["n_frames"] = taps["e_t"].shape[0]
    fout.attrs["n_recon"] = n_recon
    for k, d in taps.items():
        print(f"  {k}: {d.shape}")
    assert taps["e_t"].shape[0] == n_total, (taps["e_t"].shape[0], n_total)
    fout.close()
    fin.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit checkpoint paths; default = all in <logdir>/checkpoints")
    ap.add_argument("--batch_episodes", type=int, default=8)
    ap.add_argument("--save_recon", action="store_true")
    ap.add_argument("--recon_max", type=int, default=6000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.logdir, device=args.device)
    env = make_ari_env(seed=0)
    obs_space, act_space = env.observation_space, env.action_space

    if args.checkpoints:
        ckpts = [(int(pathlib.Path(p).stem.split("_")[1]), pathlib.Path(p)) for p in args.checkpoints]
    else:
        ckpts = list_checkpoints(args.logdir)
    print("checkpoints:", [(s, p.name) for s, p in ckpts])

    for step, path in ckpts:
        out_path = outdir / f"latents_step{step:07d}.h5"
        if out_path.exists():
            print(f"skip {out_path} (exists)")
            continue
        print(f"[{step}] {path}")
        wm = build_world_model(config, obs_space, act_space, ckpt_path=path,
                               device=args.device)
        run_checkpoint(wm, args.data, out_path, batch_episodes=args.batch_episodes,
                       save_recon=args.save_recon, recon_max=args.recon_max,
                       device=args.device, seed=args.seed)
        del wm
        torch.cuda.empty_cache()
        print(f"[{step}] wrote {out_path}")


if __name__ == "__main__":
    main()
