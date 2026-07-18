"""Render representative Push-T rollout videos for the AAAI media supplement:
one episode per automatic failure class (near-miss / stalled / oscillating /
other) plus one success, dev checkpoint. Episodes are rolled out fresh (seeds
1000+), classified with the paper's taxonomy heuristics computed on the fly,
and written as MP4 at 10 fps.

Run: python experiments/render_media.py --out paper/aaai/media
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.harness import RolloutHarness  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402

L = 8


def rollout_one(h, seed, base_seed):
    """Single-env rollout with frame capture and per-replan signal logging."""
    raw = h.reset([seed])
    frames, rewards, poss, bcoh = [], [], [], []
    committed = None
    device = h.device
    for t in range(h.max_steps):
        stacked = h.observe(raw)
        if t % L == 0:
            gen = torch.Generator(device=device).manual_seed(base_seed * 100003 + t)
            chunks, _ = h.sample_chunks(stacked, 16, generator=gen)
            if committed is not None:
                m = h.horizon - h.exec_start - L
                fresh_ov = chunks[:, :, h.exec_start : h.exec_start + m, :]
                cc = committed[:, h.exec_start + L :, :].unsqueeze(1)
                d = torch.linalg.vector_norm(fresh_ov - cc, dim=-1).mean(-1)
                bcoh.append(float(d.min(1).values[0]))
            committed = chunks[:, 0]
            exec_buf = chunks[0, 0, h.exec_start : h.exec_start + L, :].cpu().numpy()
            idx = 0
        a = h.to_real(torch.from_numpy(exec_buf[idx : idx + 1]).to(device))
        idx += 1
        res = h.step(a)
        raw = res.obs
        frames.append(h.env.call("render")[0])
        rewards.append(float(np.asarray(res.reward).ravel()[0]))
        if "agent_pos" in raw:
            poss.append(np.asarray(raw["agent_pos"], np.float32)[0].copy())
        if bool(np.asarray(res.terminated).ravel()[0]) or bool(
            np.asarray(res.truncated).ravel()[0]
        ):
            success = bool(h.successes(res.info, 1)[0])
            return frames, rewards, poss, bcoh, success
    return frames, rewards, poss, bcoh, False


def classify(rewards, poss, bcoh):
    maxrew = max(rewards) if rewards else 0.0
    if maxrew >= 0.8:
        return "near_miss"
    pos = np.stack(poss) if poss else np.zeros((2, 2))
    late = float(np.abs(np.diff(pos[-64:], axis=0)).sum())
    if late < 150.0:  # low late motion (heuristic threshold for single episodes)
        return "stalled"
    if bcoh and float(np.mean(bcoh)) > 0.12:
        return "oscillating"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ROOT / "paper/aaai/media"))
    ap.add_argument("--max_episodes", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    h = RolloutHarness(PushtEnv(), str(ROOT / "checkpoints/diffusion_pusht"), n_envs=1)
    wanted = {"success", "near_miss", "stalled", "oscillating", "other"}
    got = {}
    for i in range(args.max_episodes):
        seed = 1000 + i
        frames, rewards, poss, bcoh, success = rollout_one(h, seed, base_seed=i + 1)
        cls = "success" if success else classify(rewards, poss, bcoh)
        if cls in wanted and cls not in got:
            path = out / f"pusht_{cls}.mp4"
            imageio.mimsave(path, frames, fps=10, macro_block_size=1)
            got[cls] = seed
            print(f"{cls}: seed {seed}, {len(frames)} frames -> {path.name}", flush=True)
        if wanted == set(got):
            break
    print("done:", got)


if __name__ == "__main__":
    main()
