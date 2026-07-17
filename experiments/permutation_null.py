"""Reviewer-mandated controls (all offline):

1. PERMUTATION NULL for episode-max (and mean) aggregation: shuffle per-replan
   scores across episodes while preserving each episode's length, recompute the
   aggregated AUROC. Quantifies how much of max-agg AUROC is duration leakage
   (failures run longer -> more draws -> higher max), beyond the 0.5 baseline.
2. SURVIVOR DECOMPOSITION at W: full-episode-max AUROC restricted to episodes
   with >= W replans. Separates the prefix drop into (a) survivor-population
   shift vs (b) information truncation.
3. RECOVERY EFFECT CIs: per recovery condition, n / helped / hurt, delta with
   95% CI (paired), and the minimum detectable effect, from saved JSONs.

Run: python experiments/permutation_null.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.final_report import load_run, auroc  # noqa: E402

RUNS = [
    ("pusht-strong:dev", "results/logs_official"),
    ("pusht-strong:s1000", "results/logs_seed1000"),
    ("pusht-strong:s2000", "results/logs_seed2000"),
    ("pusht-weak:nocrop", "results/logs_pusht_weak_nocrop"),
    ("pusht-weak:s3000", "results/logs_pusht_weak_s3000"),
    ("pusht-weak:s4000", "results/logs_pusht_weak_s4000"),
    ("square-weak:s1000", "results/logs_square_s1000"),
    ("square-weak:s2000", "results/logs_square_s2000"),
    # appended after the original eight (per-run RNG -> prior outputs unchanged)
    ("can-strong:s1000", "results/logs_can_s1000"),
]
SIGNALS = ["bcoh_distmin", "stac_mmd", "stac_energy", "disp_all", "ood_knn"]
N_PERM = 500
W = 8

out = {"perm_null": {}, "survivor_decomp": {}, "recovery_ci": {}}

for name, path in RUNS:
    if not (Path(path) / "rich_logs.pkl").exists():
        continue
    run = load_run(path)
    fail, derived = run["fail"], run["derived"]
    test_ok = [e for e in run["test_eps"] if len(derived[e]["bcoh_distmin"])]
    y = np.array([fail[e] for e in test_ok], bool)
    out["perm_null"][name] = {}
    out["survivor_decomp"][name] = {}
    rng = np.random.default_rng(0)
    for s in SIGNALS:
        series = [np.asarray(derived[e][s], float) for e in test_ok]
        lens = [len(v) for v in series]
        raw_max = np.array([v.max() for v in series])
        raw_auroc = auroc(raw_max, y)
        pool = np.concatenate(series)
        nulls = []
        for _ in range(N_PERM):
            perm = rng.permutation(pool)
            mx, i = [], 0
            for L in lens:
                mx.append(perm[i : i + L].max())
                i += L
            nulls.append(auroc(np.array(mx), y))
        nulls = np.array(nulls)
        # survivor decomposition: full-episode max on episodes with >= W replans
        surv = [i for i, v in enumerate(series) if len(v) >= W]
        y_s = y[surv]
        full_on_surv = auroc(raw_max[surv], y_s)
        pref_on_surv = auroc(np.array([series[i][:W].max() for i in surv]), y_s)
        out["perm_null"][name][s] = dict(
            raw=round(raw_auroc, 3),
            null_mean=round(float(nulls.mean()), 3),
            null_lo=round(float(np.percentile(nulls, 2.5)), 3),
            null_hi=round(float(np.percentile(nulls, 97.5)), 3),
            excess=round(raw_auroc - float(nulls.mean()), 3),
        )
        out["survivor_decomp"][name][s] = dict(
            full_max_all=round(raw_auroc, 3),
            full_max_on_survivors=round(full_on_surv, 3),
            prefix_on_survivors=round(pref_on_surv, 3),
            n_surv=len(surv),
            n_fail_surv=int(y_s.sum()),
        )
    print(f"{name}: done", flush=True)


# ---- recovery effect CIs ----
def rec_ci(n, helped, hurt):
    delta = (helped - hurt) / n
    se = np.sqrt(helped + hurt) / n  # paired (McNemar) normal approx
    mde = 1.96 * se
    return dict(
        n=n,
        helped=helped,
        hurt=hurt,
        delta_pp=round(100 * delta, 1),
        ci_lo_pp=round(100 * (delta - 1.96 * se), 1),
        ci_hi_pp=round(100 * (delta + 1.96 * se), 1),
        mde_pp=round(100 * mde, 1),
    )


sources = {
    "strong_gated_p90_committed(main_compare_v2)": (
        "results/main_compare_v2/main.json",
        lambda d: (75, d["GUARD"]["helped"], d["GUARD"]["hurt"]),
    ),
    "weak60k_sweep(mid_base_study)": (
        "results/mid_base_study/study.json",
        lambda d: (d["n_episodes"], d["rows"][2]["helped"], d["rows"][2]["hurt"]),
    ),
    "strong1500_committed_p90_ext": (
        "results/recovery_power2000_ext/compare.json",
        lambda d: (d["n_episodes"], d["rows"][0]["helped"], d["rows"][0]["hurt"]),
    ),
}
for key, (p, extract) in sources.items():
    try:
        d = json.loads((ROOT / p).read_text())
        n, h, hu = extract(d)
        out["recovery_ci"][key] = rec_ci(n, h, hu)
    except Exception as e:
        out["recovery_ci"][key] = f"skip: {e}"
# guard_sweep rows (150 eps)
try:
    d = json.loads((ROOT / "results/guard_sweep_v1/sweep.json").read_text())
    for r in d["rows"]:
        out["recovery_ci"][f"strong150_{r['recovery']}_p{int(r['pct'])}"] = rec_ci(
            d["n_episodes"], r["helped"], r["hurt"]
        )
except Exception as e:
    pass
# pooled 2000-episode confirmatory condition (power500 + frozen-threshold extension)
try:
    d5 = json.loads((ROOT / "results/recovery_power500/compare.json").read_text())
    de = json.loads((ROOT / "results/recovery_power2000_ext/compare.json").read_text())
    out["recovery_ci"]["strong2000_committed_p90_pooled"] = rec_ci(
        d5["n_episodes"] + de["n_episodes"],
        d5["rows"][0]["helped"] + de["rows"][0]["helped"],
        d5["rows"][0]["hurt"] + de["rows"][0]["hurt"],
    )
except Exception as e:
    out["recovery_ci"]["strong2000_committed_p90_pooled"] = f"skip: {e}"
# square spot check if done
try:
    d = json.loads((ROOT / "results/recovery_square/sweep.json").read_text())
    for r in d["rows"]:
        out["recovery_ci"][f"square_{r['recovery']}_p{int(r['pct'])}"] = rec_ci(
            d["n_episodes"], r["helped"], r["hurt"]
        )
except Exception:
    pass

(ROOT / "results/final/controls.json").write_text(json.dumps(out, indent=2))
print("\n== permutation null (max-agg) ==")
for name in out["perm_null"]:
    row = out["perm_null"][name]
    print(name, {s: f"{v['raw']}(null {v['null_mean']},+{v['excess']})" for s, v in row.items()})
print("\n== survivor decomposition (W=8) ==")
for name in ("pusht-strong:dev", "square-weak:s1000"):
    if name in out["survivor_decomp"]:
        print(
            name,
            out["survivor_decomp"][name]["bcoh_distmin"],
            out["survivor_decomp"][name]["ood_knn"],
        )
print("\n== recovery CIs ==")
for k, v in out["recovery_ci"].items():
    print(f"  {k}: {v}")
print("\nsaved results/final/controls.json")
