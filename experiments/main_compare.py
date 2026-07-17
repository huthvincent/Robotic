"""Main test-time-method comparison on Push-T (frozen DP), with compute cost.

Methods: receding-horizon baseline (R=8), closed-loop baseline (R=1),
temporal ensembling, BID, and GUARD (gated committed re-sample). Reports success
rate, mean diffusion-calls/episode (compute), and paired McNemar vs the receding
baseline. Answers: does dense test-time selection help this base, and does GUARD
reach comparable success at far lower compute?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.harness import RolloutHarness  # noqa: E402
from baselines.closed_loop import closed_loop_rollout  # noqa: E402
from guard.controller import rollout_batch  # noqa: E402
from experiments.mvp_pusht import CKPT  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402

BCOH = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"]


def mcnemar(base, other):
    base, other = np.asarray(base, bool), np.asarray(other, bool)
    b = int((base & ~other).sum())
    c = int((~base & other).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    kk = min(b, c)
    return b, c, min(1.0, 2.0 * sum(math.comb(n, i) for i in range(kk + 1)) * 0.5**n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=75)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--gate", type=str, default="bcoh_distmin")
    ap.add_argument("--pct", type=float, default=90.0)
    ap.add_argument("--bid_k", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "main_compare"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # benchmark=True for speed; within one process cuDNN algo choice is stable across
    # calls, so the receding-baseline vs GUARD pairing stays consistent within the run.
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    h = RolloutHarness(PushtEnv(), CKPT, n_envs=args.n_envs)
    nb = (args.n_episodes + args.n_envs - 1) // args.n_envs
    seeds_of = lambda b: list(range(b * args.n_envs, (b + 1) * args.n_envs))

    # receding-horizon baseline via controller (also gives gate distribution)
    def rec(mode, recovery="committed", thr=None, k=16):
        succ, logs = [], []
        for b in range(nb):
            s, lg = rollout_batch(
                h,
                seeds_of(b),
                mode=mode,
                recovery=recovery,
                gate_signal=args.gate,
                threshold=thr,
                k=k,
                base_seed=b + 1,
            )
            succ.extend(s.tolist())
            logs.extend(lg)
        return np.array(succ, bool), logs

    def cl(mode, **kw):
        succ, calls = [], []
        for b in range(nb):
            s, c = closed_loop_rollout(h, seeds_of(b), mode=mode, base_seed=b + 1, **kw)
            succ.extend(s.tolist())
            calls.append(c)
        return np.array(succ, bool), float(np.mean(calls))

    results = {}
    print("[receding baseline]")
    base_succ, base_logs = rec("baseline")
    recs = [r for r in base_logs if any(r[s] > 0 for s in BCOH)]
    thr = float(np.percentile([r[args.gate] for r in recs], args.pct))
    n_replan = len(recs) / len(base_succ)
    results["receding_baseline"] = dict(sr=float(base_succ.mean()), calls=round(n_replan, 1))
    print(f"  SR={base_succ.mean() * 100:.1f}%  ~{n_replan:.0f} replans/ep")

    print("[GUARD gated committed]")
    g_succ, g_logs = rec("guard", recovery="committed", thr=thr)
    interv = float(np.mean([r["intervened"] for r in g_logs]))
    b, c, p = mcnemar(base_succ, g_succ)
    guard_calls = n_replan * (1 + interv * 16)  # 1 fresh trigger sample + K on intervention
    results["GUARD"] = dict(
        sr=float(g_succ.mean()),
        calls=round(guard_calls, 1),
        intervention=interv,
        helped=c,
        hurt=b,
        p=p,
    )
    print(
        f"  SR={g_succ.mean() * 100:.1f}%  Δ={(g_succ.mean() - base_succ.mean()) * 100:+.1f}pp  "
        f"interv={interv * 100:.1f}%  helped={c} hurt={b} p={p:.3f}"
    )

    for name, kw in [
        ("baseline_cl", {}),
        ("temporal_ensemble", {}),
        ("bid", dict(k=args.bid_k, k_weak=4, weak_steps=4)),
    ]:
        print(f"[{name}]")
        s, calls = cl(name, **kw)
        bb, cc, pp = mcnemar(base_succ, s)
        results[name] = dict(sr=float(s.mean()), calls=round(calls, 1), helped=cc, hurt=bb, p=pp)
        print(
            f"  SR={s.mean() * 100:.1f}%  calls/ep={calls:.0f}  Δ={(s.mean() - base_succ.mean()) * 100:+.1f}pp  p={pp:.3f}"
        )

    (out / "main.json").write_text(json.dumps(results, indent=2))
    print("\n==== MAIN COMPARE ====\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
