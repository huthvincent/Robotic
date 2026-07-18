"""Decisive recovery comparison: baseline vs {committed, consensus, extend, hold}
at a matched trigger budget, paired McNemar significance. Determinism enforced.

Run: python experiments/recovery_compare.py --n_episodes 200 --pct 90
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
from guard.controller import rollout_batch  # noqa: E402
from experiments.mvp_pusht import CKPT  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402

BCOH = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"]


def build_ood_bank(logs_dir):
    """Embedding bank for an OOD-gated repair run, built exactly like the offline
    detector's (final_report.load_run): the same 50/50 episode split with seed 0,
    successes only, every other calibration success. Keeps the gate signal in the
    repair loop identical to the one evaluated in the detection tables."""
    import pickle
    from collections import defaultdict

    d = pickle.load(open(Path(logs_dir) / "rich_logs.pkl", "rb"))
    success, records = d["success"], d["records"]
    by_ep = defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r)
    eps = sorted(by_ep)
    perm = list(eps)
    np.random.default_rng(0).shuffle(perm)
    cal_eps = [e for e in perm[: len(perm) // 2] if success[e]]
    bank_eps = cal_eps[::2]
    return np.concatenate(
        [np.stack([r["emb"].astype(np.float32) for r in by_ep[e]]) for e in bank_eps]
    )


def mcnemar(base, guard):
    base, guard = np.asarray(base, bool), np.asarray(guard, bool)
    b = int((base & ~guard).sum())
    c = int((~base & guard).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    kk = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(kk + 1)) * (0.5**n)
    return b, c, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--gate", type=str, default="bcoh_distmin")
    ap.add_argument("--pct", type=float, nargs="+", default=[90.0])
    ap.add_argument(
        "--recoveries", type=str, nargs="+", default=["committed", "consensus", "extend", "hold"]
    )
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--benchmark", type=str, default="pusht", choices=["pusht", "square"])
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "recovery_compare"))
    ap.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="shift env seeds and sampling streams; use to extend a prior "
        "run with fresh, non-overlapping paired episodes",
    )
    ap.add_argument(
        "--threshold_override",
        type=float,
        default=None,
        help="skip in-run calibration and reuse a frozen gate threshold "
        "(e.g. from a prior run's compare.json)",
    )
    ap.add_argument(
        "--ood_bank_from",
        type=str,
        default=None,
        help="rich-logs dir supplying the embedding bank; required for "
        "--gate ood_knn (the informative signal on Square)",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    if args.benchmark == "pusht":
        h = RolloutHarness(PushtEnv(), args.ckpt, n_envs=args.n_envs)
    else:
        from envs.robomimic_harness import RobomimicHarness

        h = RobomimicHarness(args.ckpt, n_envs=args.n_envs)
    n_batches = (args.n_episodes + args.n_envs - 1) // args.n_envs

    boff = args.seed_offset // args.n_envs  # keep sampling streams disjoint too
    bank = build_ood_bank(args.ood_bank_from) if args.ood_bank_from else None
    if args.gate == "ood_knn" and bank is None:
        ap.error("--gate ood_knn requires --ood_bank_from <rich-logs dir>")

    def run(mode, recovery="committed", threshold=None):
        succ, logs = [], []
        for b in range(n_batches):
            seeds = list(
                range(args.seed_offset + b * args.n_envs, args.seed_offset + (b + 1) * args.n_envs)
            )
            s, lg = rollout_batch(
                h,
                seeds,
                mode=mode,
                recovery=recovery,
                gate_signal=args.gate,
                threshold=threshold,
                k=args.k,
                base_seed=boff + b + 1,
                ood_bank=bank,
            )
            succ.extend(s.tolist())
            logs.extend(lg)
        return np.array(succ, bool), logs

    print("[baseline]")
    base_succ, base_logs = run("baseline")
    base_sr = float(base_succ.mean())
    # replans where a committed plan exists (the only ones a repair gate can fire on)
    recs = [r for r in base_logs if any(r[s] > 0 for s in BCOH)]
    gate_vals = np.array([r[args.gate] for r in recs])
    print(f"  baseline SR={base_sr * 100:.1f}%  (n={len(base_succ)})")

    rows = []
    for pct in args.pct:
        thr = (
            args.threshold_override
            if args.threshold_override is not None
            else float(np.percentile(gate_vals, pct))
        )
        for recovery in args.recoveries:
            g_succ, g_logs = run("guard", recovery=recovery, threshold=thr)
            g_sr = float(g_succ.mean())
            interv = float(np.mean([r["intervened"] for r in g_logs]))
            b, c, p = mcnemar(base_succ, g_succ)
            # re-fire statistics: does the gate fire again after an intervention?
            fires = {}
            for r in g_logs:
                if r["intervened"]:
                    fires[r["episode"]] = fires.get(r["episode"], 0) + 1
            n_fired = len(fires)
            n_refired = sum(1 for v in fires.values() if v >= 2)
            refire = dict(
                episodes_fired=n_fired,
                episodes_refired=n_refired,
                refire_rate=round(n_refired / n_fired, 3) if n_fired else None,
                mean_fires_per_fired_episode=round(float(np.mean(list(fires.values()))), 2)
                if n_fired
                else None,
            )
            rows.append(
                dict(
                    pct=pct,
                    recovery=recovery,
                    threshold=thr,
                    success=g_sr,
                    delta=g_sr - base_sr,
                    intervention=interv,
                    hurt=b,
                    helped=c,
                    p=p,
                    refire=refire,
                )
            )
            print(
                f"  p{pct:>4.0f} {recovery:9s} SR={g_sr * 100:5.1f}%  Δ={(g_sr - base_sr) * 100:+5.1f}pp  "
                f"interv={interv * 100:4.1f}%  helped={c} hurt={b} p={p:.3f}"
            )

    (out / "compare.json").write_text(
        json.dumps(
            dict(
                baseline_sr=base_sr,
                n_episodes=len(base_succ),
                gate=args.gate,
                seed_offset=args.seed_offset,
                threshold_override=args.threshold_override,
                rows=rows,
            ),
            indent=2,
        )
    )
    print("\nsaved", out / "compare.json")


if __name__ == "__main__":
    main()
