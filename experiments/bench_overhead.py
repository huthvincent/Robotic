"""Wall-clock overhead of K-sample monitoring (reviewer-8): time one replan's
batched diffusion call at K=1 (plain execution) vs K in {2,16} (monitoring
draws) for E=25 parallel envs on the dev checkpoint. Median over 30 calls
after 5 warmups, CUDA-synchronized.

Run: python experiments/bench_overhead.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.harness import RolloutHarness  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402

E = 25
h = RolloutHarness(PushtEnv(), str(ROOT / "checkpoints/diffusion_pusht"), n_envs=E)
raw = h.reset(seeds=list(range(E)))
stacked = h.observe(raw)


def timed_sample(k, reps=30, warmup=5):
    ts = []
    for i in range(reps + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h.sample_chunks(stacked, k=k)
        torch.cuda.synchronize()
        if i >= warmup:
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


res = {"n_envs": E}
for k in (1, 2, 16):
    res[f"K{k}_ms"] = round(timed_sample(k) * 1000, 1)
    print(f"K={k}: {res[f'K{k}_ms']:.1f} ms per replan ({E} envs batched)", flush=True)
res["overhead_K2_vs_K1"] = round(res["K2_ms"] / res["K1_ms"], 3)
res["overhead_K16_vs_K1"] = round(res["K16_ms"] / res["K1_ms"], 3)
(ROOT / "results/final/overhead.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
