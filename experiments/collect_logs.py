"""Collect rich detector logs on a checkpoint (baseline rollouts only).

Run: python experiments/collect_logs.py --ckpt checkpoints/diffusion_pusht \
       --n_episodes 300 --n_envs 25 --k 16 --out results/logs_official
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guard.collect import collect_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "diffusion_pusht"))
    ap.add_argument("--benchmark", type=str, default="pusht", choices=["pusht", "square", "can"])
    ap.add_argument("--n_episodes", type=int, default=300)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    if args.benchmark == "pusht":
        from envs.harness import RolloutHarness
        from lerobot.envs.configs import PushtEnv

        h = RolloutHarness(PushtEnv(), args.ckpt, n_envs=args.n_envs)
    else:
        from envs.robomimic_harness import RobomimicHarness

        task = {"square": "NutAssemblySquare", "can": "PickPlaceCan"}[args.benchmark]
        h = RobomimicHarness(args.ckpt, n_envs=args.n_envs, task=task)
    nb = (args.n_episodes + args.n_envs - 1) // args.n_envs

    t0 = time.time()
    succ, records = [], []
    for b in range(nb):
        seeds = list(range(b * args.n_envs, (b + 1) * args.n_envs))
        s, recs = collect_batch(h, seeds, k=args.k, base_seed=b + 1)
        succ.extend(s.tolist())
        records.extend(recs)
        print(
            f"batch {b + 1}/{nb}: SR={np.mean(s) * 100:.1f}%  cum={np.mean(succ) * 100:.1f}%  "
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
            },
            f,
        )
    print(
        f"saved {out / 'rich_logs.pkl'}  episodes={len(succ)} SR={np.mean(succ) * 100:.1f}% "
        f"records={len(records)}"
    )


if __name__ == "__main__":
    main()
