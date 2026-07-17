"""Selective autonomy: use the calibrated trigger to DEFER (abstain on) the
episodes it flags as likely failures, and measure success on the retained
(autonomously executed) portion. A good trigger raises retained success as
coverage drops — the honest, positive use of a strong failure detector.

Uses saved baseline logs (no GPU). Compares the temporal trigger vs dispersion
vs a random-defer control.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/mvp_pusht_v1")
BCOH = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"]


def retained_success(order_desc, success, coverages):
    """order_desc: episode indices sorted by defer-priority (defer first).
    Return retained success rate at each coverage."""
    n = len(order_desc)
    out = []
    for cov in coverages:
        keep = order_desc[int(round(n * (1 - cov))) :]  # drop the top (1-cov) fraction
        out.append(float(np.mean([success[e] for e in keep])) if len(keep) else float("nan"))
    return out


def main():
    data = pickle.load(open(RUN / "baseline_logs.pkl", "rb"))
    logs, success = data["logs"], data["success"]
    recs = [r for r in logs if any(r[s] > 0 for s in BCOH)]
    eps = sorted({r["episode"] for r in recs})
    succ = {e: bool(success[e]) for e in eps}
    base_sr = float(np.mean([succ[e] for e in eps]))

    def ep_score(sig):
        return {e: max(r[sig] for r in recs if r["episode"] == e) for e in eps}

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    out = {"baseline_success": base_sr, "coverages": coverages, "curves": {}}
    print(f"baseline success (coverage 1.0) = {base_sr * 100:.1f}%\n")
    print(f"{'defer-by':16s} " + " ".join(f"cov{c:.1f}" for c in coverages))
    rng = np.random.default_rng(0)
    for sig in ["bcoh_distmin", "bcoh_distmean", "disp_all"]:
        sc = ep_score(sig)
        order = sorted(eps, key=lambda e: -sc[e])  # highest trigger deferred first
        curve = retained_success(order, succ, coverages)
        out["curves"][sig] = curve
        print(f"{sig:16s} " + " ".join(f"{v * 100:5.1f}" for v in curve))
    # random-defer control (avg of 200 shuffles)
    rand = np.zeros(len(coverages))
    for _ in range(200):
        order = list(eps)
        rng.shuffle(order)
        rand += np.array(retained_success(order, succ, coverages))
    rand /= 200
    out["curves"]["random"] = rand.tolist()
    print(f"{'random':16s} " + " ".join(f"{v * 100:5.1f}" for v in rand))

    (RUN / "selective_autonomy.json").write_text(json.dumps(out, indent=2))
    best = out["curves"]["bcoh_distmin"]
    print(
        f"\nDeferring 20% by boundary-coherence: retained success {base_sr * 100:.1f}% -> "
        f"{best[2] * 100:.1f}% (+{(best[2] - base_sr) * 100:.1f}pp); random control -> {rand[2] * 100:.1f}%."
    )
    print("saved", RUN / "selective_autonomy.json")


if __name__ == "__main__":
    main()
