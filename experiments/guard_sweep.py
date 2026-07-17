"""Strengthen the closing-the-loop result: sweep gate threshold x recovery
variant on Push-T, with paired McNemar significance vs the frozen baseline.

Baseline is run once; every GUARD config is run on the SAME seeds so outcomes
are paired (GUARD diverges from baseline only after its first intervention).

Run: python experiments/guard_sweep.py --n_episodes 200 --n_envs 25 --gate bcoh_distmin
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.mvp_pusht import run_batch, CKPT  # noqa: E402
from envs.harness import RolloutHarness  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402

BCOH = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"]


def mcnemar(base, guard):
    """base, guard: bool arrays (per-episode success). Return (b, c, p_two_sided).
    b = base success & guard fail (hurt), c = base fail & guard success (helped)."""
    base, guard = np.asarray(base, bool), np.asarray(guard, bool)
    b = int(np.sum(base & ~guard))
    c = int(np.sum(~base & guard))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return b, c, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--gate", type=str, default="bcoh_distmin")
    ap.add_argument("--percentiles", type=float, nargs="+", default=[82, 86, 90, 93, 96])
    ap.add_argument("--recoveries", type=str, nargs="+", default=["committed", "consensus"])
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "guard_sweep"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    h = RolloutHarness(PushtEnv(), CKPT, n_envs=args.n_envs)
    n_batches = (args.n_episodes + args.n_envs - 1) // args.n_envs

    def run(mode, threshold=None, recovery="committed"):
        succ, logs = [], []
        for b in range(n_batches):
            seeds = list(range(b * args.n_envs, (b + 1) * args.n_envs))
            s, lg = run_batch(
                h,
                seeds,
                mode,
                threshold=threshold,
                k=args.k,
                base_seed=b + 1,
                gate_signal=args.gate,
                recovery=recovery,
            )
            succ.extend(s.tolist())
            logs.extend(lg)
        return np.array(succ, bool), logs

    t0 = time.time()
    print("[baseline]")
    base_succ, base_logs = run("baseline")
    base_sr = float(base_succ.mean())
    recs = [r for r in base_logs if any(r[s] > 0 for s in BCOH)]
    gate_vals = np.array([r[args.gate] for r in recs])
    print(f"  baseline SR={base_sr * 100:.1f}%  ({time.time() - t0:.0f}s)")

    rows = []
    for recovery in args.recoveries:
        for pct in args.percentiles:
            thr = float(np.percentile(gate_vals, pct))
            g_succ, g_logs = run("guard", threshold=thr, recovery=recovery)
            g_sr = float(g_succ.mean())
            interv = float(np.mean([r["intervened"] for r in g_logs]))
            b, c, p = mcnemar(base_succ, g_succ)
            row = dict(
                recovery=recovery,
                pct=pct,
                threshold=thr,
                success=g_sr,
                delta=g_sr - base_sr,
                intervention=interv,
                hurt=b,
                helped=c,
                p=p,
            )
            rows.append(row)
            print(
                f"  {recovery:9s} p{pct:>4.0f} thr={thr:.3f}  SR={g_sr * 100:5.1f}%  "
                f"Δ={(g_sr - base_sr) * 100:+5.1f}pp  interv={interv * 100:4.1f}%  "
                f"helped={c} hurt={b} p={p:.3f}"
            )

    result = dict(baseline_sr=base_sr, n_episodes=len(base_succ), gate=args.gate, rows=rows)
    (out / "sweep.json").write_text(json.dumps(result, indent=2))
    np.savez(out / "base_succ.npz", base_succ=base_succ)

    # Pareto figure: delta vs intervention, per recovery
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 4))
        for recovery in args.recoveries:
            rr = [r for r in rows if r["recovery"] == recovery]
            rr.sort(key=lambda r: r["intervention"])
            plt.plot(
                [r["intervention"] * 100 for r in rr],
                [r["delta"] * 100 for r in rr],
                "-o",
                ms=4,
                label=recovery,
            )
        plt.axhline(0, color="k", lw=0.8)
        plt.xlabel("intervention rate (%)")
        plt.ylabel("Δ success vs baseline (pp)")
        plt.title(f"GUARD closing-the-loop (gate={args.gate})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "pareto.png", dpi=130)
        print(f"saved {out / 'pareto.png'}")
    except Exception as e:
        print("plot skipped:", e)

    print("\n==== SWEEP ====\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
