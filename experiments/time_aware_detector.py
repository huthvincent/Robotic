"""Reviewer-9 (2): does alarm history add value over a strong time-aware
detector, at comparable coverage and handoff?

Online-sequential deferral: at each replan t the detector observes elapsed time
(t) and the running-max alarm score; it defers at the FIRST replan a rule fires.
Three training-free detectors, all calibrated on success-only rollouts:

  T  (elapsed only)  : defer at first t with t >= T0          [= the timeout rule]
  A  (alarm only)    : defer at first t with runmax >= tau    [= our conformal alarm]
  A+T (combined, OR) : defer at first t with (runmax >= tau OR t >= T0)

For each detector we sweep its parameters, collect (coverage, retained-success,
median-handoff) points, and report, at matched coverage anchors, the operating
point with the highest retained success and its handoff. The question is whether
A+T reaches T's retained success at an EARLIER handoff -- i.e. whether the alarm
lets a time-aware detector act sooner without giving up selectivity.

Output: results/final/time_aware.json
Run: python experiments/time_aware_detector.py
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
    ("can-strong:s1000", "results/logs_can_s1000"),
    ("pusht-weak:nocrop", "results/logs_pusht_weak_nocrop"),
    ("square-weak:s1000", "results/logs_square_s1000"),
    ("square-weak:s2000", "results/logs_square_s2000"),
    ("pusht-mid:60k", "results/logs_pusht_mid60k"),
]
ANCHORS = (0.6, 0.7, 0.8)


def eval_detector(series, fail, defer_time):
    """defer_time(e) -> replan index of first deferral, or None (kept).
    Returns coverage, retained success, median relative handoff (deferred fails)."""
    kept, ret_ok, handoffs = 0, 0, []
    for e, s in series.items():
        L = len(s)
        dt = defer_time(e, s)
        if dt is None:
            kept += 1
            if not fail[e]:
                ret_ok += 1
        elif fail[e]:
            handoffs.append(dt / max(1, L - 1))
    n = len(series)
    cov = kept / n
    retained = ret_ok / kept if kept else float("nan")
    handoff = float(np.median(handoffs)) if handoffs else None
    return cov, retained, handoff


def timeout_point(points, cov_anchor, tol=0.06):
    """The timeout operating point closest to the coverage anchor (timeout is a
    one-parameter family: at a given coverage its retained and handoff are fixed)."""
    cand = [p for p in points if abs(p[0] - cov_anchor) <= tol and not np.isnan(p[1])]
    if not cand:
        return None
    c, r, h = min(cand, key=lambda p: abs(p[0] - cov_anchor))
    return dict(
        coverage=round(c, 3), retained=round(r, 3), handoff=(round(h, 3) if h is not None else None)
    )


def earliest_at(points, cov_anchor, min_retained, tol=0.06):
    """Among points near the coverage anchor with retained >= min_retained, the
    EARLIEST handoff -- the question is whether a detector can hand off sooner
    than timeout WITHOUT giving up (much) retained success."""
    cand = [
        p
        for p in points
        if abs(p[0] - cov_anchor) <= tol
        and not np.isnan(p[1])
        and p[1] >= min_retained
        and p[2] is not None
    ]
    if not cand:
        return None
    c, r, h = min(cand, key=lambda p: p[2])
    return dict(coverage=round(c, 3), retained=round(r, 3), handoff=round(h, 3))


out = {}
for name, path in RUNS:
    if not (Path(path) / "rich_logs.pkl").exists():
        continue
    run = load_run(path)
    fail, derived = run["fail"], run["derived"]
    sig = "ood_knn" if name.startswith(("pusht-weak", "pusht-mid", "square")) else "bcoh_distmin"
    cal_pool = run["quant_eps"] if sig == "ood_knn" else run["cal_eps"]
    test_ok = [e for e in run["test_eps"] if len(derived[e][sig])]
    series = {e: np.asarray(derived[e][sig], float) for e in test_ok}

    # alarm thresholds: conformal (1-alpha) quantile of success calibration running-max
    calmax = np.array(
        [np.asarray(derived[e][sig], float).max() for e in cal_pool if len(derived[e][sig])]
    )
    taus = {}
    for a in np.round(np.arange(0.05, 0.55, 0.05), 2):
        q = min(1.0, np.ceil((len(calmax) + 1) * (1 - a)) / len(calmax))
        taus[float(a)] = float(np.quantile(calmax, q))
    # timeout thresholds: candidate T0 over the observed replan counts
    lens = sorted({len(s) for s in series.values()})
    T0s = sorted(set(int(x) for x in np.quantile(lens, np.linspace(0.0, 1.0, 25))))

    def first_alarm(s, tau):
        idx = np.nonzero(s >= tau)[0]
        return int(idx[0]) if len(idx) else None

    def first_timeout(s, T0):
        return T0 if (len(s) - 1) > T0 else None

    pts = {"T": [], "A": [], "AT": []}
    # timeout-only
    for T0 in T0s:
        pts["T"].append(eval_detector(series, fail, lambda e, s, T0=T0: first_timeout(s, T0)))
    # alarm-only
    for a, tau in taus.items():
        pts["A"].append(eval_detector(series, fail, lambda e, s, tau=tau: first_alarm(s, tau)))
    # combined OR
    for a, tau in taus.items():
        for T0 in T0s:

            def dt(e, s, tau=tau, T0=T0):
                fa, ft = first_alarm(s, tau), first_timeout(s, T0)
                cand = [x for x in (fa, ft) if x is not None]
                return min(cand) if cand else None

            pts["AT"].append(eval_detector(series, fail, dt))

    # Timeout's usable coverage tops out near the success rate (successes are
    # short, failures long), so anchors are chosen PER RUN inside timeout's
    # non-degenerate range: 0.5/0.75/0.95 of its max non-trivial coverage.
    tcovs = [c for (c, r, h) in pts["T"] if c < 0.999]
    cmax = max(tcovs) if tcovs else 0.5
    anchor_covs = [round(f * cmax, 2) for f in (0.5, 0.75, 0.95)]
    anchors = {}
    for cov in anchor_covs:
        t = timeout_point(pts["T"], cov, tol=0.08)
        rec = {"cov_target": cov, "T": t}
        if t is not None:
            floor = t["retained"] - 0.03
            rec["A"] = earliest_at(pts["A"], t["coverage"], floor, tol=0.08)
            rec["AT"] = earliest_at(pts["AT"], t["coverage"], floor, tol=0.08)
        anchors[str(cov)] = rec
    out[name] = dict(
        signal=sig,
        sr=round(float(np.mean(run["success"])), 3),
        n_test=len(test_ok),
        cmax_timeout=round(cmax, 3),
        anchors=anchors,
        frontier={
            d: [
                [
                    round(c, 3),
                    round(r, 3) if not np.isnan(r) else None,
                    (round(h, 3) if h is not None else None),
                ]
                for (c, r, h) in pts[d]
            ]
            for d in pts
        },
    )
    a7 = anchors.get("0.7", {})
    t, at = a7.get("T"), a7.get("AT")
    if t and at and t["handoff"] is not None:
        gain = round(t["handoff"] - at["handoff"], 3)
        print(
            f"{name:20s} cov~{t['coverage']}  T: ret {t['retained']} @ho {t['handoff']} | "
            f"A+T (ret>={t['retained'] - 0.03:.2f}): ret {at['retained']} @ho {at['handoff']} "
            f" -> earlier by {gain}",
            flush=True,
        )
    elif t:
        print(
            f"{name:20s} cov~{t['coverage']}  T: ret {t['retained']} @ho {t['handoff']} | "
            f"A+T: no point within 3pp retained",
            flush=True,
        )

(ROOT / "results/final/time_aware.json").write_text(json.dumps(out, indent=2))
print("saved results/final/time_aware.json")
