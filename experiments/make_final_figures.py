"""Final paper figure: 4 panels from results/final/report.json + yield analyses.
(a) strong-regime pooled AUROC by signal   (b) regime dependence (signal x regime)
(c) evaluation-metric inflation (leakage)  (d) yield earliness-selectivity frontier
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rep = json.loads((ROOT / "results/final/report.json").read_text())
POOL = rep["pooled"]
RUNS = rep["runs"]

fig, axs1 = plt.subplots(1, 2, figsize=(11.5, 4.4))
fig2, axs2 = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax = [axs1[0], axs1[1], axs2[0], axs2[1]]

# (a) pooled strong AUROC
sigs = [
    "bcoh_distmin",
    "stac_mmd",
    "stac_energy",
    "stac_chamfer",
    "bcoh_distmean",
    "disp_all",
    "ood_knn",
]
labels = [
    "bound.\ncoh (min)",
    "STAC\nMMD",
    "STAC\nenergy",
    "STAC\nchamfer",
    "bound.\ncoh (mean)",
    "disper-\nsion",
    "embed.\nOOD",
]
colors = ["#2166ac"] * 5 + ["#b2182b", "#8c510a"]
SS = rep["seed_stats"]["pusht-strong"]["signals"]
vals = [SS[s]["mean"] for s in sigs]
errs = [SS[s]["std"] or 0 for s in sigs]
ax[0].bar(range(len(sigs)), vals, yerr=errs, color=colors, capsize=3, alpha=0.85)
for i, s in enumerate(sigs):
    pts0 = list(SS[s]["per_seed"].values())
    ax[0].scatter([i] * len(pts0), pts0, s=14, color="k", zorder=3)
ax[0].set_xticks(range(len(sigs)))
ax[0].set_xticklabels(labels, fontsize=9)
ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
ax[0].set_ylim(0.4, 1.0)
ax[0].set_ylabel("episode AUROC (max-agg)")
ax[0].set_title("(a) Strong regime: family saturation\n(Push-T, mean$\\pm$s.d. over 3 seeds)")

# (b) regime dependence
regimes = [
    ("pusht-strong", "Push-T strong\n(52-64%, 3 seeds)"),
    ("pusht-weak", "Push-T weak\n(25%)"),
    ("square-weak", "Square weak\n(16-25%, 2 seeds)"),
]
show = [
    ("bcoh_distmin", "boundary coherence", "#2166ac"),
    ("stac_mmd", "STAC MMD", "#67a9cf"),
    ("ood_knn", "embedding OOD", "#8c510a"),
    ("disp_all", "dispersion", "#b2182b"),
]
w = 0.2
for i, (s, lab, c) in enumerate(show):
    xs = np.arange(len(regimes)) + (i - 1.5) * w
    ys = [POOL[r][s]["auroc_max"] for r, _ in regimes]
    ax[1].bar(xs, ys, width=w, label=lab, color=c)
ax[1].set_xticks(range(len(regimes)))
ax[1].set_xticklabels([n for _, n in regimes], fontsize=10)
ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
ax[1].set_ylim(0.2, 1.0)
ax[1].legend(fontsize=9)
ax[1].set_title("(b) Signal ranking depends on regime")

# (c) metric inflation for coherence signal on strong regime (dev + seeds mean)
# seed-level ±1 s.d. whiskers on every estimated bar (reviewer-8: prefix bar had none)
runs_strong = [k for k in RUNS if k.startswith("pusht-strong")]
mx_v = [RUNS[k]["signals"]["bcoh_distmin"]["auroc_max"] for k in runs_strong]
cs_v = [RUNS[k]["signals"]["bcoh_distmin"]["auroc_cusum"] for k in runs_strong]
pw_v = [RUNS[k]["prefix"]["bcoh_distmin_W8"] for k in runs_strong]
mx, cs, pw = np.mean(mx_v), np.mean(cs_v), np.mean(pw_v)
nl = 1.0
bars = [
    ("episode length\n(null)", nl, "#4d4d4d", 0.0),
    ("CUSUM\n(leaky)", cs, "#b2182b", float(np.std(cs_v))),
    ("episode max", mx, "#2166ac", float(np.std(mx_v))),
    ("prefix-W8\n(leak-free)", pw, "#67a9cf", float(np.std(pw_v))),
]
ax[2].bar(
    [b[0] for b in bars],
    [b[1] for b in bars],
    color=[b[2] for b in bars],
    yerr=[b[3] for b in bars],
    capsize=3,
)
ax[2].axhline(0.5, color="k", ls="--", lw=0.8)
ax[2].set_ylim(0.4, 1.05)
try:
    ctrl = json.loads((ROOT / "results/final/controls.json").read_text())
    nulls = [
        ctrl["perm_null"][k]["bcoh_distmin"]["null_mean"]
        for k in ctrl["perm_null"]
        if k.startswith("pusht-strong")
    ]
    pn = float(np.mean(nulls))
    ax[2].axhline(pn, color="#e08214", ls=":", lw=1.6)
    ax[2].text(2.6, pn + 0.012, f"permutation null {pn:.2f}", color="#e08214", fontsize=9)
except Exception:
    pass
for i, b in enumerate(bars):
    ax[2].text(i, b[1] + b[3] + 0.01, f"{b[1]:.2f}", ha="center", fontsize=10)
ax[2].set_title("(a) Duration leakage inflates metrics\n(boundary coherence, strong regime)")

# (d) yield frontier (dev seed)
ya = json.loads((ROOT / "results/logs_official/yield_analysis_bcoh_distmin.json").read_text())
P = ya["policies"]
pts = []
for a in [0.2, 0.3, 0.4]:
    p = P[f"signal_a{a}"]
    pts.append(("alarm α=%.1f" % a, p["median_handoff_frac"], p["retained_sr"], "#2166ac", "o"))
for key, lab in [("timeout_cov0.59", "timeout"), ("prefix8_a0.3", "prefix-8")]:
    p = P[key]
    if p.get("median_handoff_frac") is not None:
        pts.append(
            (
                lab,
                p["median_handoff_frac"],
                p["retained_sr"],
                "#b2182b" if "time" in key else "#8c510a",
                "s",
            )
        )
p = P["random_cov0.70"]
pts.append(("random", 0.5, p["retained_sr"], "#4d4d4d", "^"))
for lab, x, y, c, m in pts:
    ax[3].scatter(x, y * 100, color=c, marker=m, s=60)
    ax[3].annotate(lab, (x, y * 100), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax[3].axhline(64.0, color="gray", lw=0.8, ls="--")
ax[3].set_xlabel("median handoff time (fraction of episode)")
ax[3].set_ylabel("retained success (%)")
ax[3].set_title("(b) Yield: earliness vs selectivity\n(deferral policies, Push-T dev)")
ax[3].grid(alpha=0.3)

fig.tight_layout()
fig2.tight_layout()
fig.savefig(ROOT / "results/final/fig_signals.png", dpi=150)
fig.savefig(ROOT / "paper/aaai/fig_signals.pdf")
fig2.savefig(ROOT / "results/final/fig_protocol.png", dpi=150)
fig2.savefig(ROOT / "paper/aaai/fig_protocol.pdf")
print("saved fig_signals + fig_protocol")
