"""Recovery in the regime it should help: evaluate under-trained DP checkpoints,
pick one with recovery headroom (~35-60% success), and run paired baseline vs
GUARD there. Under-trained policies are more multimodal, so chunk-boundary
mode-jumps dominate failures and the trigger-gated recovery has room to help.

Run after training: python experiments/mid_base_study.py
"""

from __future__ import annotations

import argparse
import glob
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
    ap.add_argument(
        "--ckpt_glob",
        type=str,
        default=str(ROOT / "checkpoints/dp_pusht_mid/checkpoints/*/pretrained_model"),
    )
    ap.add_argument("--n_eval", type=int, default=50)
    ap.add_argument("--n_full", type=int, default=200)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--gate", type=str, default="bcoh_distmin")
    ap.add_argument("--pcts", type=float, nargs="+", default=[85, 90, 93])
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "mid_base_study"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    ckpts = sorted(glob.glob(args.ckpt_glob))
    print("checkpoints:", [c.split("/checkpoints/")[-1].split("/")[0] for c in ckpts])

    def run(h, mode, recovery="committed", thr=None, n_ep=args.n_full):
        nb = (n_ep + args.n_envs - 1) // args.n_envs
        succ, logs = [], []
        for b in range(nb):
            seeds = list(range(b * args.n_envs, (b + 1) * args.n_envs))
            s, lg = rollout_batch(
                h,
                seeds,
                mode=mode,
                recovery=recovery,
                gate_signal=args.gate,
                threshold=thr,
                k=args.k,
                base_seed=b + 1,
            )
            succ.extend(s.tolist())
            logs.extend(lg)
        return np.array(succ, bool), logs

    # 1) quick success eval of each checkpoint, pick one with headroom
    evals = {}
    for c in ckpts:
        h = RolloutHarness(PushtEnv(), c, n_envs=args.n_envs)
        s, _ = run(h, "baseline", n_ep=args.n_eval)
        step = c.split("/checkpoints/")[-1].split("/")[0]
        evals[step] = float(s.mean())
        print(f"  {step}: SR={s.mean() * 100:.1f}%")
        del h
    # prefer closest to 0.5 within [0.3,0.62]
    cand = {k: v for k, v in evals.items() if 0.30 <= v <= 0.62}
    if not cand:
        cand = evals
    best_step = min(cand, key=lambda k: abs(cand[k] - 0.5))
    best_ckpt = [c for c in ckpts if f"/{best_step}/" in c][0]
    print(f"selected {best_step} (SR={evals[best_step] * 100:.1f}%)")

    # 2) full paired baseline vs GUARD on selected checkpoint
    h = RolloutHarness(PushtEnv(), best_ckpt, n_envs=args.n_envs)
    base_succ, base_logs = run(h, "baseline")
    recs = [r for r in base_logs if any(r[s] > 0 for s in BCOH)]
    gate_vals = np.array([r[args.gate] for r in recs])
    base_sr = float(base_succ.mean())
    print(f"[{best_step}] baseline SR={base_sr * 100:.1f}% (n={len(base_succ)})")

    rows = []
    for pct in args.pcts:
        thr = float(np.percentile(gate_vals, pct))
        for recovery in ("committed", "consensus"):
            g_succ, g_logs = run(h, "guard", recovery=recovery, thr=thr)
            interv = float(np.mean([r["intervened"] for r in g_logs]))
            b, c, p = mcnemar(base_succ, g_succ)
            rows.append(
                dict(
                    pct=pct,
                    recovery=recovery,
                    sr=float(g_succ.mean()),
                    delta=float(g_succ.mean()) - base_sr,
                    intervention=interv,
                    helped=c,
                    hurt=b,
                    p=p,
                )
            )
            print(
                f"  p{pct:>3.0f} {recovery:9s} SR={g_succ.mean() * 100:5.1f}%  "
                f"Δ={(g_succ.mean() - base_sr) * 100:+5.1f}pp  interv={interv * 100:4.1f}%  "
                f"helped={c} hurt={b} p={p:.4f}"
            )

    result = dict(
        evals=evals, selected=best_step, baseline_sr=base_sr, n_episodes=len(base_succ), rows=rows
    )
    (out / "study.json").write_text(json.dumps(result, indent=2))
    print("\nsaved", out / "study.json")


if __name__ == "__main__":
    main()
