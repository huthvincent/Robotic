"""Calibration analysis of the GUARD trigger (no GPU; uses saved baseline logs).

Demonstrates the conformal guarantee: a threshold set on a calibration set of
SUCCESSFUL rollouts controls the false-intervention rate on held-out successful
rollouts (calibration works), while still flagging steps from failed episodes
(useful detection). Compares the temporal trigger against the dispersion signal.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/mvp_pusht_v1")
BCOH = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"]
SIGNALS = ["bcoh_distmin", "bcoh_distmean", "disp_all", "disp_first"]


def main():
    data = pickle.load(open(RUN / "baseline_logs.pkl", "rb"))
    logs, success = data["logs"], data["success"]
    fail_by_ep = {ep: (not ok) for ep, ok in enumerate(success)}
    recs = [r for r in logs if any(r[s] > 0 for s in BCOH)]

    eps = sorted({r["episode"] for r in recs})
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    # calibration set = SUCCESSFUL episodes in the first half; test = second half (all)
    half = len(eps) // 2
    cal_eps = [e for e in eps[:half] if not fail_by_ep[e]]
    test_eps = set(eps[half:])
    test_succ_eps = {e for e in test_eps if not fail_by_ep[e]}
    test_fail_eps = {e for e in test_eps if fail_by_ep[e]}

    def steps(sig, ep_set):
        return np.array([r[sig] for r in recs if r["episode"] in ep_set])

    out = {
        "n_cal_eps": len(cal_eps),
        "n_test_eps": len(test_eps),
        "n_test_fail_eps": len(test_fail_eps),
        "signals": {},
    }
    print(
        f"cal(success) eps={len(cal_eps)}  test eps={len(test_eps)} (fail={len(test_fail_eps)})\n"
    )
    print(
        f"{'signal':14s} {'target':>7s} {'realized_FP_on_test_success':>28s} {'flag_rate_test_fail':>20s}"
    )
    for sig in SIGNALS:
        cal_scores = steps(sig, set(cal_eps))
        out["signals"][sig] = {}
        for alpha in (0.05, 0.10, 0.20):
            tau = float(np.quantile(cal_scores, 1 - alpha))
            fp = float((steps(sig, test_succ_eps) >= tau).mean())  # false interventions on nominal
            flag_fail = float((steps(sig, test_fail_eps) >= tau).mean())  # flagged in failed eps
            out["signals"][sig][f"alpha_{alpha}"] = dict(
                tau=tau, realized_fp=fp, flag_fail=flag_fail
            )
            print(f"{sig:14s} {alpha:7.2f} {fp:28.3f} {flag_fail:20.3f}")
        print()

    (RUN / "calibration.json").write_text(json.dumps(out, indent=2))
    print("saved", RUN / "calibration.json")
    print(
        "\nInterpretation: realized_FP on held-out SUCCESSFUL rollouts should track the target "
        "alpha (conformal calibration holds); flag_rate on FAILED episodes should be much higher "
        "(useful detection). A well-calibrated trigger separates the two."
    )


if __name__ == "__main__":
    main()
