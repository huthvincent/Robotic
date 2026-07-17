"""Figure 3: (a) prefix-W curves per regime; (b) timeout frontier vs alarm points;
(c) calibration robustness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rep = json.loads((ROOT / "results/final/report.json").read_text())
ext = json.loads((ROOT / "results/final/extras.json").read_text())
RUNS = rep["runs"]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
WS = [4, 6, 8, 10, 12]

# (a) prefix-W curves
series = [
    (
        "pusht-strong",
        ["pusht-strong:dev", "pusht-strong:s1000", "pusht-strong:s2000"],
        "bcoh_distmin",
        "#2166ac",
        "coherence, strong Push-T (3 seeds)",
    ),
    (
        "pusht-strong",
        ["pusht-strong:dev", "pusht-strong:s1000", "pusht-strong:s2000"],
        "ood_knn",
        "#9ecae1",
        "OOD, strong Push-T",
    ),
    (
        "square-weak",
        ["square-weak:s1000", "square-weak:s2000"],
        "ood_knn",
        "#8c510a",
        "OOD, weak Square (2 seeds)",
    ),
    (
        "pusht-weak",
        ["pusht-weak:nocrop", "pusht-weak:s3000", "pusht-weak:s4000"],
        "ood_knn",
        "#d8a165",
        "OOD, weak Push-T (3 seeds)",
    ),
]
for _, keys, sig, c, lab in series:
    ys = np.array([[RUNS[k]["prefix"][f"{sig}_W{w}"] for w in WS] for k in keys])
    m = ys.mean(0)
    ax[0].plot(WS, m, "-o", color=c, label=lab, ms=5)
    if len(keys) > 1:
        ax[0].fill_between(WS, ys.min(0), ys.max(0), color=c, alpha=0.18)
ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
ax[0].set_xlabel("decision replan $W$")
ax[0].set_ylabel("prefix-conditional AUROC")
ax[0].set_xticks(WS)
ax[0].set_ylim(0.35, 1.0)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[0].set_title("(a) Leak-free early prediction across $W$")

# (b) timeout frontier vs alarm operating points (dev seed, coherence)
fr = ext["timeout_frontier"]["pusht-strong:dev"]
ax[1].plot(
    [f["coverage"] for f in fr],
    [f["retained"] * 100 for f in fr],
    "-s",
    color="#b2182b",
    ms=4,
    label="timeout rule (all $T_0$)",
)
ya = json.loads((ROOT / "results/logs_official/yield_analysis_bcoh_distmin.json").read_text())
P = ya["policies"]
for a in [0.1, 0.2, 0.3, 0.4, 0.5]:
    p = P[f"signal_a{a}"]
    ax[1].scatter(p["coverage"], p["retained_sr"] * 100, color="#2166ac", s=55, zorder=3)
    ax[1].annotate(
        f"α={a}\n@{p['median_handoff_frac']:.0%} of ep." if a in (0.1, 0.4) else f"α={a}",
        (p["coverage"], p["retained_sr"] * 100),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        color="#2166ac",
    )
ax[1].axhline(64.0, color="gray", lw=0.8, ls="--")
ax[1].set_xlabel("coverage (fraction kept)")
ax[1].set_ylabel("retained success (%)")
ax[1].grid(alpha=0.3)
ax[1].legend(fontsize=8, loc="lower left")
ax[1].set_title(
    "(b) Timeout frontier vs calibrated alarm\n(alarm defers at 44--75% of episode; timeout at 94--97%)"
)

# (c) calibration robustness
rob = ext["cal_robust"]
for key, c, lab in [
    ("pusht-strong:dev", "#2166ac", "strong Push-T (coherence)"),
    ("square-weak:s1000", "#8c510a", "weak Square (OOD)"),
]:
    ns, mus, sds = [], [], []
    for n, v in sorted(rob[key].items(), key=lambda kv: int(kv[0])):
        ns.append(int(n))
        mus.append(v["fp_mean"])
        sds.append(v["fp_sd"])
    ax[2].errorbar(ns, mus, yerr=sds, fmt="-o", color=c, capsize=4, label=lab)
ax[2].axhline(0.3, color="k", ls=":", lw=1.2)
ax[2].text(38, 0.31, "target α=0.3", fontsize=8)
ax[2].set_xlabel("calibration set size (successful episodes)")
ax[2].set_ylabel("realized false-deferral")
ax[2].set_xscale("log")
ax[2].grid(alpha=0.3)
ax[2].legend(fontsize=8)
ax[2].set_title("(c) Calibration robustness\n(50 random splits per size)")

plt.tight_layout()
plt.savefig(ROOT / "results/final/fig_extras.png", dpi=150)
plt.savefig(ROOT / "paper/aaai/fig_extras.pdf")
print("saved fig_extras")
