"""Reviewer-10 (1): the paper's expected-utility model had no time term, so a
late timeout (which keeps essentially only already-terminated successes)
strictly dominated the calibrated alarm under our own metric -- an internal
contradiction, since the case for the alarm is precisely WHEN it fires.

Time-aware expected utility. Execution is not free: wall-clock, energy, wear,
and a blocked workcell all scale with how long the robot runs. Let

    c_t = cost of executing for one full reference episode, in units of one
          task success (R_ref = mean episode length under full autonomy).

Per-episode utility of a deferral policy:

    EU(c_t) = retained * cov                     autonomous successes
            + (p_fb - c_fb) * (1 - cov)          fallback value minus its cost
            - c_t * E[executed replans] / R_ref  execution cost

Full autonomy pays the full execution cost (E[executed]/R_ref = 1 by
construction); a deferral policy stops deferred episodes at the firing replan,
so EARLIER firing now buys utility. We evaluate full autonomy, every timeout
threshold, and every conformal-alarm operating point inside this model and
report the break-even c_t at which the best alarm overtakes the best timeout.

Output: results/final/utility_time.json
Run: python experiments/utility_time.py
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
]
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5)
CTS = (0.0, 0.05, 0.1, 0.2, 0.5)
PFBS = (0.3, 0.6, 0.9)


def policy_stats(series, fail, defer_at):
    """defer_at(e, s) -> first-firing replan index or None (kept).
    Returns coverage, retained success (among kept), mean executed replans."""
    kept, ok, executed = 0, 0, []
    for e, s in series.items():
        d = defer_at(e, s)
        if d is None:
            kept += 1
            ok += 0 if fail[e] else 1
            executed.append(len(s))
        else:
            executed.append(d + 1)
    n = len(series)
    cov = kept / n
    retained = (ok / kept) if kept else float("nan")
    return cov, retained, float(np.mean(executed))


def eu(cov, retained, exec_mean, R_ref, p_fb, c_t, c_fb=0.0):
    ret = 0.0 if np.isnan(retained) else retained
    return ret * cov + (p_fb - c_fb) * (1 - cov) - c_t * exec_mean / R_ref


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
    calmax = np.array(
        [np.asarray(derived[e][sig], float).max() for e in cal_pool if len(derived[e][sig])]
    )

    cov0, ret0, exec0 = policy_stats(series, fail, lambda e, s: None)
    R_ref = exec0

    policies = {"autonomy": dict(cov=cov0, retained=ret0, exec=exec0)}
    lens = sorted({len(s) for s in series.values()})
    for T0 in sorted(set(int(x) for x in np.quantile(lens, np.linspace(0, 1, 25)))):
        c, r, ex = policy_stats(
            series, fail, lambda e, s, T0=T0: (T0 if (len(s) - 1) > T0 else None)
        )
        if c < 0.999:
            policies[f"timeout_T{T0}"] = dict(cov=c, retained=r, exec=ex)
    taus = {}
    for a in ALPHAS:
        q = min(1.0, np.ceil((len(calmax) + 1) * (1 - a)) / len(calmax))
        taus[a] = float(np.quantile(calmax, q))

    def d_alarm(e, s, tau):
        idx = np.nonzero(s >= tau)[0]
        return int(idx[0]) if len(idx) else None

    for a, tau in taus.items():
        c, r, ex = policy_stats(series, fail, lambda e, s, tau=tau: d_alarm(e, s, tau))
        policies[f"alarm_a{a}"] = dict(cov=c, retained=r, exec=ex)
    # combined OR rule: defer at min(first alarm crossing, timeout T0). Its
    # parameter space contains pure timeout (tau=inf) and pure alarm (T0=inf),
    # so best-combined >= max(best timeout, best alarm) by construction; the
    # question is how much execution cost the alarm term saves on top.
    T0s = [int(k.split("T")[1]) for k in policies if k.startswith("timeout")]
    # include tau=inf (alarm never fires) so the combined family contains pure
    # timeout as a limit point and best-combined >= best-timeout by construction
    comb_taus = dict(taus)
    comb_taus[0.0] = float("inf")
    for a, tau in comb_taus.items():
        for T0 in T0s:

            def d_comb(e, s, tau=tau, T0=T0):
                fa = d_alarm(e, s, tau)
                ft = T0 if (len(s) - 1) > T0 else None
                cand = [x for x in (fa, ft) if x is not None]
                return min(cand) if cand else None

            c, r, ex = policy_stats(series, fail, d_comb)
            policies[f"comb_a{a}_T{T0}"] = dict(cov=c, retained=r, exec=ex)

    grid = {}
    for p_fb in PFBS:
        for c_t in CTS:
            vals = {
                k: eu(v["cov"], v["retained"], v["exec"], R_ref, p_fb, c_t)
                for k, v in policies.items()
            }
            to_keys = [k for k in vals if k.startswith("timeout")]
            al_keys = [k for k in vals if k.startswith("alarm")]
            cb_keys = [k for k in vals if k.startswith("comb")]
            bt = max(to_keys, key=lambda k: vals[k])
            ba = max(al_keys, key=lambda k: vals[k])
            bc = max(cb_keys, key=lambda k: vals[k])
            grid[f"pfb{p_fb}_ct{c_t}"] = dict(
                autonomy=round(vals["autonomy"], 3),
                best_timeout=round(vals[bt], 3),
                best_timeout_at=bt,
                best_alarm=round(vals[ba], 3),
                best_alarm_at=ba,
                best_combined=round(vals[bc], 3),
                best_combined_at=bc,
                alarm_minus_timeout=round(vals[ba] - vals[bt], 3),
                combined_minus_timeout=round(vals[bc] - vals[bt], 3),
            )

    # break-even c_t per p_fb: smallest c_t where the best alarm >= best timeout
    breakeven = {}
    for p_fb in PFBS:
        be = None
        for c_t in np.arange(0.0, 1.0001, 0.005):
            vals = {
                k: eu(v["cov"], v["retained"], v["exec"], R_ref, p_fb, float(c_t))
                for k, v in policies.items()
            }
            ba = max(v for k, v in vals.items() if k.startswith("alarm"))
            bt = max(v for k, v in vals.items() if k.startswith("timeout"))
            if ba >= bt:
                be = round(float(c_t), 3)
                break
        breakeven[str(p_fb)] = be

    out[name] = dict(
        signal=sig,
        R_ref=round(R_ref, 1),
        policies={
            k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in policies.items()
        },
        grid=grid,
        breakeven_ct=breakeven,
    )
    print(f"{name:20s} R_ref={R_ref:5.1f}  break-even c_t: {breakeven}", flush=True)

(ROOT / "results/final/utility_time.json").write_text(json.dumps(out, indent=2))
print("saved results/final/utility_time.json")
