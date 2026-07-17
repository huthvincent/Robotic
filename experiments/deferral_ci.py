"""Bootstrap 95% CIs for the deferral operating points quoted in the paper
(reviewer-8 M5). For each run and conformal alpha: episode-level bootstrap over
the TEST split, re-deriving the threshold from the (fixed) calibration split
each time is NOT done -- the threshold is a deployment constant, so we bootstrap
the test-set metrics (coverage, retained success, false-deferral) at the frozen
threshold, which is the uncertainty a deployed operating point carries.

Output: results/final/deferral_ci.json
Run: python experiments/deferral_ci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.final_report import load_run  # noqa: E402

RUNS = [
    ("pusht-strong:dev", "results/logs_official"),
    ("pusht-strong:s1000", "results/logs_seed1000"),
    ("pusht-strong:s2000", "results/logs_seed2000"),
    ("pusht-weak:nocrop", "results/logs_pusht_weak_nocrop"),
    ("pusht-weak:s3000", "results/logs_pusht_weak_s3000"),
    ("pusht-weak:s4000", "results/logs_pusht_weak_s4000"),
    ("square-weak:s1000", "results/logs_square_s1000"),
    ("square-weak:s2000", "results/logs_square_s2000"),
    ("can-strong:s1000", "results/logs_can_s1000"),
]
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5)
B = 2000

out = {}
for name, path in RUNS:
    run = load_run(path)
    derived, fail = run["derived"], run["fail"]
    sig = "ood_knn" if name.startswith(("pusht-weak", "square")) else "bcoh_distmin"
    cal_pool = run["quant_eps"] if sig == "ood_knn" else run["cal_eps"]
    test_ok = [e for e in run["test_eps"] if len(derived[e][sig])]
    calmax = np.array([max(derived[e][sig]) for e in cal_pool if len(derived[e][sig])])
    mx = np.array([max(derived[e][sig]) for e in test_ok])
    y = np.array([fail[e] for e in test_ok], bool)  # True = failure
    rng = np.random.default_rng(0)
    res_a = {}
    for a in ALPHAS:
        q = min(1.0, np.ceil((len(calmax) + 1) * (1 - a)) / len(calmax))
        tau = float(np.quantile(calmax, q))
        kept = mx < tau

        def point(idx):
            k = kept[idx]
            yy = y[idx]
            cov = float(k.mean())
            ret = float((~yy[k]).mean()) if k.sum() else np.nan
            fd = float((mx[idx][~yy] >= tau).mean()) if (~yy).sum() else np.nan
            return cov, ret, fd

        cov0, ret0, fd0 = point(np.arange(len(test_ok)))
        boots = np.array(
            [point(rng.integers(0, len(test_ok), len(test_ok))) for _ in range(B)], dtype=float
        )
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        res_a[str(a)] = dict(
            tau=tau,
            n_test=len(test_ok),
            n_kept=int(kept.sum()),
            coverage=round(cov0, 3),
            coverage_ci=[round(lo[0], 3), round(hi[0], 3)],
            retained=round(ret0, 3),
            retained_ci=[round(lo[1], 3), round(hi[1], 3)],
            false_defer=round(fd0, 3),
            false_defer_ci=[round(lo[2], 3), round(hi[2], 3)],
        )
        print(
            f"{name} a={a}: cov {cov0:.2f} [{lo[0]:.2f},{hi[0]:.2f}] "
            f"ret {ret0:.3f} [{lo[1]:.3f},{hi[1]:.3f}] (kept n={int(kept.sum())})",
            flush=True,
        )
    out[name] = dict(signal=sig, alphas=res_a)

(ROOT / "results/final/deferral_ci.json").write_text(json.dumps(out, indent=2))
print("saved results/final/deferral_ci.json")
