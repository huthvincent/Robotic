"""Yield (selective autonomy) decision-curve analysis with time-aware baselines.

Anti-triviality defense: on early-terminating benchmarks a TIMEOUT rule ("defer if
still running at step T0") is a near-perfect post-hoc classifier of failure — but it
defers at the END of the episode. The operational value of a signal-based alarm is
EARLINESS: hand off mid-episode at similar retained success, saving wasted robot time.

For each deferral policy at matched coverage we report:
  - retained success rate (among kept episodes)
  - false-deferral rate on successes
  - median handoff time among deferred failures (fraction of episode horizon)
  - wasted-steps saved vs the timeout rule (per deferred failure)
Policies: signal running-max crossing (conformal tau sweep), timeout rule, random,
prefix-W signal (decide by replan W only).

Run: python experiments/yield_analysis.py results/logs_official [--signal bcoh_distmin]
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def pdist_sets(A, B):
    return np.linalg.norm(A[:, None] - B[None], axis=-1).mean(-1)


def load(path):
    data = pickle.load(open(Path(path) / "rich_logs.pkl", "rb"))
    success, records = data["success"], data["records"]
    fail = {e: (not success[e]) for e in range(len(success))}
    by_ep = defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r)
    for e in by_ep:
        by_ep[e].sort(key=lambda r: r["t"])
    return success, fail, by_ep


def derive_signal(by_ep, e, sig):
    rs = by_ep[e]
    out = []
    for j in range(1, len(rs)):
        head = rs[j]["head_ov"].astype(np.float32)
        tail_prev = rs[j - 1]["tail_ov"].astype(np.float32)
        cc = tail_prev[0]
        d_k = np.linalg.norm(head - cc[None], axis=-1).mean(-1)
        if sig == "bcoh_distmin":
            out.append(float(d_k.min()))
        elif sig == "stac_energy":
            d_ab = pdist_sets(tail_prev, head)
            d_aa = pdist_sets(tail_prev, tail_prev)
            d_bb = pdist_sets(head, head)
            out.append(float(2 * d_ab.mean() - d_aa.mean() - d_bb.mean()))
        else:
            raise KeyError(sig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=str)
    ap.add_argument("--signal", type=str, default="bcoh_distmin")
    args = ap.parse_args()
    success, fail, by_ep = load(args.run)
    eps_all = sorted(by_ep)
    perm = list(eps_all)
    np.random.default_rng(0).shuffle(perm)
    half = len(perm) // 2
    cal_eps = [e for e in perm[:half] if not fail[e]]
    test_eps = sorted(perm[half:])

    sigs = {e: derive_signal(by_ep, e, args.signal) for e in eps_all}
    test_ok = [e for e in test_eps if len(sigs[e])]
    y = np.array([fail[e] for e in test_ok], bool)
    n_rep = {e: len(sigs[e]) for e in test_ok}  # replan count (proxy for duration)
    max_rep = max(n_rep.values())

    def eval_policy(defer_time):
        """defer_time: dict e -> replan index of deferral (None = kept)."""
        kept = np.array([defer_time[e] is None for e in test_ok])
        deferred = ~kept
        retained_sr = float((~y[kept]).mean()) if kept.sum() else float("nan")
        fd = float(deferred[~y].mean()) if (~y).sum() else float("nan")
        handoff = [
            defer_time[e] / max(1, n_rep[e] - 1)
            for e in test_ok
            if defer_time[e] is not None and fail[e]
        ]
        saved = [
            (n_rep[e] - 1 - defer_time[e]) for e in test_ok if defer_time[e] is not None and fail[e]
        ]
        return dict(
            coverage=float(kept.mean()),
            retained_sr=retained_sr,
            false_defer=fd,
            defer_on_fail=float(deferred[y].mean()) if y.sum() else float("nan"),
            median_handoff_frac=float(np.median(handoff)) if handoff else None,
            mean_replans_saved_per_deferred_fail=float(np.mean(saved)) if saved else None,
        )

    results = {"signal": args.signal, "policies": {}}

    # --- 1) signal running-max crossing at conformal taus ---
    cal_max = np.array([max(sigs[e]) for e in cal_eps if len(sigs[e])])
    for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
        q = min(1.0, np.ceil((len(cal_max) + 1) * (1 - a)) / len(cal_max))
        tau = float(np.quantile(cal_max, q))
        dt = {}
        for e in test_ok:
            cross = next((i for i, v in enumerate(sigs[e]) if v >= tau), None)
            dt[e] = cross
        results["policies"][f"signal_a{a}"] = eval_policy(dt)

    # --- 2) timeout rule at matched coverages ---
    for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cov_target = results["policies"][f"signal_a{a}"]["coverage"]
        # threshold T0 such that fraction of episodes with n_rep <= T0 ~= cov_target
        durations = np.array(sorted(n_rep[e] for e in test_ok))
        T0 = int(durations[min(len(durations) - 1, int(np.floor(cov_target * len(durations))))])
        dt = {e: (T0 if n_rep[e] - 1 > T0 else None) for e in test_ok}
        r = eval_policy(dt)
        r["T0_replans"] = T0
        results["policies"][f"timeout_cov{cov_target:.2f}"] = r

    # --- 3) random deferral at matched coverages ---
    rng = np.random.default_rng(1)
    for a in [0.3]:
        cov_target = results["policies"][f"signal_a{a}"]["coverage"]
        rs_list, hf = [], []
        for rep in range(50):
            defer_set = set(
                rng.choice(
                    test_ok, int(round((1 - cov_target) * len(test_ok))), replace=False
                ).tolist()
            )
            dt = {
                e: (rng.integers(0, max(1, n_rep[e] - 1)) if e in defer_set else None)
                for e in test_ok
            }
            rs_list.append(eval_policy(dt)["retained_sr"])
        results["policies"][f"random_cov{cov_target:.2f}"] = dict(
            coverage=cov_target, retained_sr=float(np.mean(rs_list))
        )

    # --- 4) prefix-8 signal deferral ---
    cal_pref = np.array([max(sigs[e][:8]) for e in cal_eps if len(sigs[e]) >= 8])
    for a in [0.2, 0.3]:
        q = min(1.0, np.ceil((len(cal_pref) + 1) * (1 - a)) / len(cal_pref))
        tau = float(np.quantile(cal_pref, q))
        dt = {}
        for e in test_ok:
            s8 = sigs[e][:8]
            cross = next((i for i, v in enumerate(s8) if v >= tau), None)
            dt[e] = cross
        results["policies"][f"prefix8_a{a}"] = eval_policy(dt)

    out = Path(args.run) / f"yield_analysis_{args.signal}.json"
    out.write_text(json.dumps(results, indent=2))
    print(
        f"test n={len(test_ok)} fail={int(y.sum())} SR={(~y).mean() * 100:.1f}%  signal={args.signal}\n"
    )
    print(
        f"{'policy':22s} {'cov':>5s} {'retSR':>6s} {'FPdef':>6s} {'TPdef':>6s} {'handoff':>8s} {'saved':>6s}"
    )
    for name, r in results["policies"].items():
        hf = r.get("median_handoff_frac")
        sv = r.get("mean_replans_saved_per_deferred_fail")
        print(
            f"{name:22s} {r['coverage']:5.2f} {r['retained_sr'] * 100:5.1f}% "
            f"{(r.get('false_defer') or 0) * 100:5.1f}% {(r.get('defer_on_fail') or 0) * 100:5.1f}% "
            f"{hf if hf is not None else float('nan'):8.2f} "
            f"{sv if sv is not None else float('nan'):6.1f}"
        )
    print("\nsaved", out)


if __name__ == "__main__":
    main()
