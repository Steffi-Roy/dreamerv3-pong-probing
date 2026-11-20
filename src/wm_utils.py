"""Shared helpers: rebuild a frozen DreamerV3 world model from a checkpoint.

Checkpoints are written by `src/train_dreamer.py`. Because training runs with
`compile: True`, `torch.compile` inserts `_orig_mod.` into every parameter name;
we strip it so the same weights load into an eager (uncompiled) model.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1] / "repos" / "dreamerv3-torch"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import models  # noqa: E402
import networks  # noqa: E402  (re-exported for callers)

TAPS = ("e_t", "z_t", "h_t", "s_t")


def load_config(logdir, device=None):
    """Load the config dumped by train_dreamer.py into an argparse.Namespace."""
    cfg = json.loads((pathlib.Path(logdir) / "config.json").read_text())
    config = argparse.Namespace(**cfg)
    if device is not None:
        config.device = device
    return config


def strip_compile_prefix(state_dict):
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def split_prefix(state_dict, prefix):
    n = len(prefix)
    return {k[n:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def build_world_model(config, obs_space, act_space, ckpt_path=None, device="cuda:0",
                      with_actor=False):
    """Instantiate WorldModel (+ optionally the actor) and load frozen weights.

    `ckpt_path=None` gives the random-init model used as the control floor
    (prompt.md 4.5); the RNG is seeded by the caller for reproducibility.
    """
    config.device = device
    wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
    behavior = models.ImagBehavior(config, wm).to(device) if with_actor else None

    if ckpt_path is not None:
        sd = strip_compile_prefix(torch.load(ckpt_path, map_location=device)["agent_state_dict"])
        missing = wm.load_state_dict(split_prefix(sd, "_wm."), strict=True)
        assert not getattr(missing, "missing_keys", []), missing
        if behavior is not None:
            behavior.load_state_dict(split_prefix(sd, "_task_behavior."), strict=True)

    wm.eval().requires_grad_(False)
    if behavior is not None:
        behavior.eval().requires_grad_(False)
    return (wm, behavior) if with_actor else wm


def list_checkpoints(logdir):
    """Checkpoint paths sorted by policy step, as [(step, path), ...]."""
    ckptdir = pathlib.Path(logdir) / "checkpoints"
    out = []
    for p in sorted(ckptdir.glob("checkpoint_*.pt")):
        step = int(p.stem.split("_")[1])
        out.append((step, p))
    out.sort(key=lambda t: t[0])
    # de-duplicate steps (the "_final" snapshot can coincide with a periodic one)
    seen, uniq = set(), []
    for step, p in out:
        if step in seen:
            continue
        seen.add(step)
        uniq.append((step, p))
    return uniq


@torch.no_grad()
def taps_from_sequence(wm, image, action, is_first, is_terminal=None, decode=False):
    """Run the frozen world model over a batch of sequences and return the 4 taps.

    Args
      image:      (B, T, 64, 64, 3) uint8
      action:     (B, T, A) float32 one-hot (action taken *at* that step)
      is_first:   (B, T) bool
    Returns dict with e_t (B,T,De), z_t (B,T,Dz), h_t (B,T,Dh), s_t (B,T,Dz+Dh)
    and, if `decode`, recon (B,T,64,64,3) float in [0,1].
    """
    device = next(wm.parameters()).device
    if is_terminal is None:
        is_terminal = np.zeros(image.shape[:2], np.float32)
    data = {
        "image": np.asarray(image),
        "action": np.asarray(action, np.float32),
        "is_first": np.asarray(is_first, np.float32),
        "is_terminal": np.asarray(is_terminal, np.float32),
    }
    data = wm.preprocess({k: torch.as_tensor(v, device=device) for k, v in data.items()})
    embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)
    stoch = post["stoch"]
    stoch = stoch.reshape(list(stoch.shape[:-2]) + [-1]) if stoch.dim() == 4 else stoch
    out = {"e_t": embed, "z_t": stoch, "h_t": post["deter"], "s_t": feat}
    if decode:
        out["recon"] = wm.heads["decoder"](feat)["image"].mode()
    return out
