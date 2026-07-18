"""Reviewer-10 (5): how much of the strong-vs-weak monitorability gap does
failure COMPOSITION alone explain?

AUROC decomposes exactly over failure types (success set fixed):
    AUROC = sum_t w_t * AUROC_t,   w_t = fraction of failures of type t,
because AUROC = P(score_fail > score_succ) = sum_t P(type t) P(> | type t).

We therefore hold each regime's per-type AUROCs fixed and swap in the OTHER
regime's failure mix. If the reweighted strong-regime AUROC moves most of the
way to the weak-regime value, composition alone explains the gap; the residual
is per-type detectability change (the same failure type scoring differently
under a different policy).

Uses the automatic Push-T taxonomy in results/final/extras.json (per-type n and
per-type type-vs-success AUROC). Groups: strong = 3 strong seeds pooled, weak =
3 crop-ablated seeds pooled (mid-60k reported separately).

Output: results/final/reweight.json
Run: python experiments/composition_reweight.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = json.loads((ROOT / "results/final/extras.json").read_text())
TAX = EXT["taxonomy"]

GROUPS = {
    "strong": ["pusht-strong:dev", "pusht-strong:s1000", "pusht-strong:s2000"],
    "weak": ["pusht-weak:nocrop", "pusht-weak:s3000", "pusht-weak:s4000"],
    "mid60k": ["pusht-mid:60k"],
}
SIGS = ["bcoh_distmin", "stac_mmd", "ood_knn"]
TYPES = ["near_miss", "stalled", "oscillating", "other"]


def group_stats(runs):
    """Pooled failure mix and n-weighted per-type AUROC across a run group."""
    n = {t: 0 for t in TYPES}
    num = {s: {t: 0.0 for t in TYPES} for s in SIGS}
    for r in runs:
        for t, d in TAX[r].items():
            n[t] += d["n"]
            for s in SIGS:
                num[s][t] += d[s] * d["n"]
    total = sum(n.values())
    mix = {t: n[t] / total for t in TYPES}
    auc = {s: {t: (num[s][t] / n[t]) if n[t] else None for t in TYPES} for s in SIGS}
    return mix, auc


def combine(mix, auc_s):
    return sum(mix[t] * auc_s[t] for t in TYPES if auc_s[t] is not None and mix[t] > 0)


out = {}
stats = {g: group_stats(runs) for g, runs in GROUPS.items()}
for s in SIGS:
    mix_s, auc_s = stats["strong"]
    mix_w, auc_w = stats["weak"]
    a_ss = combine(mix_s, auc_s[s])  # strong detectability, strong mix (actual)
    a_ww = combine(mix_w, auc_w[s])  # weak detectability, weak mix (actual)
    a_sw = combine(mix_w, auc_s[s])  # strong detectability, WEAK mix (counterfactual)
    a_ws = combine(mix_s, auc_w[s])  # weak detectability, STRONG mix (counterfactual)
    gap = a_ww - a_ss
    comp_share = (a_sw - a_ss) / gap if abs(gap) > 1e-9 else None
    out[s] = dict(
        strong_actual=round(a_ss, 3),
        weak_actual=round(a_ww, 3),
        strong_detect_weak_mix=round(a_sw, 3),
        weak_detect_strong_mix=round(a_ws, 3),
        gap=round(gap, 3),
        composition_share=round(comp_share, 3) if comp_share is not None else None,
    )
    print(
        f"{s:14s} strong {a_ss:.3f} -> weak-mix counterfactual {a_sw:.3f} "
        f"(actual weak {a_ww:.3f}); composition explains "
        f"{100 * comp_share:.0f}% of the gap" if comp_share is not None else f"{s}: no gap",
        flush=True,
    )
out["mix"] = {g: {t: round(v, 3) for t, v in stats[g][0].items()} for g in GROUPS}
(ROOT / "results/final/reweight.json").write_text(json.dumps(out, indent=2))
print("saved results/final/reweight.json")
