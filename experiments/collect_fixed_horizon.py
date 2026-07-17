"""Fixed-horizon collection: every Push-T episode runs the FULL 300 steps
(success no longer terminates), removing duration confounding BY CONSTRUCTION.
Success label = task solved at any step (recorded, not terminating).

Under this protocol both classes are observed for equal duration, so
episode-max AUROC is interpretable without a permutation null. If the
strong-regime coherence signals stay near their prefix-protocol values
(~0.55-0.65) rather than their duration-confounded values (~0.67-0.69), the
duration-confounding claim is confirmed on-line rather than by permutation.

Run: python experiments/collect_fixed_horizon.py --ckpt checkpoints/diffusion_pusht \
       --n_episodes 300 --out results/logs_fixedh_dev
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.harness import RolloutHarness  # noqa: E402
from guard.collect import collect_batch  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402


class NoSuccessTermination(gym.Wrapper):
    """Mask success-termination; record success in info; truncate only at horizon."""

    def reset(self, **kw):
        self._succ = False
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if info.get("is_success", False) or term:
            self._succ = True
        info["is_success"] = self._succ
        return obs, rew, False, trunc, info  # never terminate on success


class FixedHorizonHarness(RolloutHarness):
    def __init__(self, env_cfg, ckpt, n_envs, device="cuda"):
        super().__init__(env_cfg, ckpt, n_envs, device)
        # rebuild the vector env with wrapped sub-envs
        import gym_pusht  # noqa: F401

        def mk():
            e = gym.make(
                "gym_pusht/PushT-v0",
                obs_type=env_cfg.obs_type,
                render_mode=env_cfg.render_mode,
                visualization_width=env_cfg.visualization_width,
                visualization_height=env_cfg.visualization_height,
                max_episode_steps=env_cfg.episode_length,
            )
            return NoSuccessTermination(e)

        self.env.close()
        self.env = gym.vector.SyncVectorEnv([mk for _ in range(n_envs)])
        self.max_steps = env_cfg.episode_length

    @staticmethod
    def successes(info, n_envs):
        # success is carried in per-step info (vectorized): fall back to final_info
        if "is_success" in info:
            return np.asarray(info["is_success"], dtype=bool)
        if "final_info" in info and isinstance(info["final_info"], dict):
            return np.asarray(info["final_info"]["is_success"], dtype=bool)
        return np.zeros(n_envs, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints/diffusion_pusht"))
    ap.add_argument("--n_episodes", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument(
        "--absorbing",
        action="store_true",
        help="after success, execute hold-position actions (physical freeze)",
    )
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    h = FixedHorizonHarness(PushtEnv(), args.ckpt, n_envs=args.n_envs)
    if args.absorbing:
        # wrap step: freeze succeeded envs at their current position
        orig_step = h.step
        state = {"succ": np.zeros(args.n_envs, bool), "pos": None}

        def absorbing_step(a_np):
            if state["pos"] is not None and state["succ"].any():
                a_np = a_np.copy()
                a_np[state["succ"]] = state["pos"][state["succ"]]
            res = orig_step(a_np)
            s = h.successes(res.info, args.n_envs)
            state["succ"] = state["succ"] | s
            state["pos"] = np.asarray(res.obs["agent_pos"], np.float32)[:, : a_np.shape[1]]
            return res

        h.step = absorbing_step
        orig_reset = h.reset

        def absorbing_reset(seeds):
            state["succ"][:] = False
            state["pos"] = None
            return orig_reset(seeds)

        h.reset = absorbing_reset
    nb = (args.n_episodes + args.n_envs - 1) // args.n_envs
    t0 = time.time()
    succ, records = [], []
    for b in range(nb):
        seeds = list(range(b * args.n_envs, (b + 1) * args.n_envs))
        s, recs = collect_batch(h, seeds, k=args.k, base_seed=b + 1)
        succ.extend(s.tolist())
        records.extend(recs)
        print(
            f"batch {b + 1}/{nb}: SR={np.mean(s) * 100:.1f}% cum={np.mean(succ) * 100:.1f}% "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    with open(out / "rich_logs.pkl", "wb") as f:
        pickle.dump(
            {
                "success": succ,
                "records": records,
                "ckpt": args.ckpt,
                "k": args.k,
                "n_envs": args.n_envs,
                "fixed_horizon": True,
            },
            f,
        )
    print(f"saved {out}/rich_logs.pkl episodes={len(succ)} SR={np.mean(succ) * 100:.1f}%")


if __name__ == "__main__":
    main()
