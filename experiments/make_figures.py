"""Publication figures from validated data: (a) failure-prediction AUROC with
bootstrap CIs, (b) risk-coverage, (c) conformal calibration + failure flagging.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/mvp_pusht_v1")
BCOH = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"]


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return np.nan
    o = np.argsort(np.concatenate([p, n]))
    r = np.empty(len(o))
    r[o] = np.arange(1, len(o) + 1)
    return (r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def main():
    data = pickle.load(open(RUN / "baseline_logs.pkl", "rb"))
    logs, success = data["logs"], data["success"]
    fail = {e: (not ok) for e, ok in enumerate(success)}
    recs = [r for r in logs if any(r[s] > 0 for s in BCOH)]
    eps = sorted({r["episode"] for r in recs})
    ep_fail = np.array([fail[e] for e in eps])

    sigs = [("bcoh_distmin", "boundary-coh"), ("disp_all", "dispersion")]
    fig, ax = plt.subplots(1, 4, figsize=(17.2, 3.6))

    # (a) AUROC with bootstrap CIs (episode max-agg)
    names, vals, los, his = [], [], [], []
    rng = np.random.default_rng(0)
    for key, label in [
        ("bcoh_distmin", "boundary\ncoherence"),
        ("bcoh_distmean", "boundary\n(mean)"),
        ("disp_all", "dispersion"),
        ("disp_first", "dispersion\n(first)"),
    ]:
        agg = np.array([max(r[key] for r in recs if r["episode"] == e) for e in eps])
        a = auroc(agg, ep_fail)
        bs = [
            auroc(agg[idx], ep_fail[idx])
            for idx in (rng.choice(len(eps), len(eps), replace=True) for _ in range(1500))
        ]
        bs = [x for x in bs if not np.isnan(x)]
        names.append(label)
        vals.append(a)
        los.append(a - np.percentile(bs, 2.5))
        his.append(np.percentile(bs, 97.5) - a)
    colors = ["#2166ac", "#4393c3", "#b2182b", "#d6604d"]
    ax[0].bar(names, vals, yerr=[los, his], color=colors, capsize=4)
    ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
    ax[0].set_ylim(0.4, 0.9)
    ax[0].set_ylabel("failure-prediction AUROC")
    ax[0].set_title("(a) Trigger quality")

    # (b) risk-coverage
    step_fail = np.array([fail[r["episode"]] for r in recs]).astype(float)
    for key, label, st in [
        ("bcoh_distmin", "boundary-coh", "-o"),
        ("disp_all", "dispersion", "--^"),
    ]:
        v = np.array([r[key] for r in recs])
        order = np.argsort(v)
        fs = step_fail[order]
        covs = np.linspace(1, 0.1, 19)
        risks = [fs[: max(1, int(len(v) * c))].mean() for c in covs]
        ax[1].plot(covs, risks, st, ms=3, label=label)
    ax[1].set_xlabel("coverage")
    ax[1].set_ylabel("failure rate (retained)")
    ax[1].set_title("(b) Risk-coverage")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # (c) calibration
    cal = (
        json.loads((RUN / "calibration.json").read_text())
        if (RUN / "calibration.json").exists()
        else None
    )
    if cal:
        alphas = [0.05, 0.10, 0.20]
        for key, label, c in [
            ("bcoh_distmin", "boundary-coh", "#2166ac"),
            ("disp_all", "dispersion", "#b2182b"),
        ]:
            fp = [cal["signals"][key][f"alpha_{a}"]["realized_fp"] for a in alphas]
            ff = [cal["signals"][key][f"alpha_{a}"]["flag_fail"] for a in alphas]
            ax[2].plot(alphas, fp, "--o", color=c, alpha=0.6, label=f"{label} FP(nominal)")
            ax[2].plot(alphas, ff, "-s", color=c, label=f"{label} flag(fail)")
        ax[2].plot([0, 0.2], [0, 0.2], "k:", lw=0.8, label="target")
        ax[2].set_xlabel("target intervention rate α")
        ax[2].set_ylabel("realized rate")
        ax[2].set_title("(c) Calibration & flagging")
        ax[2].legend(fontsize=6)

    # (d) selective autonomy: retained success vs coverage
    sel = (
        json.loads((RUN / "selective_autonomy.json").read_text())
        if (RUN / "selective_autonomy.json").exists()
        else None
    )
    if sel:
        covs = sel["coverages"]
        for key, label, st in [
            ("bcoh_distmin", "boundary-coh", "-o"),
            ("disp_all", "dispersion", "--^"),
            ("random", "random", ":s"),
        ]:
            ax[3].plot(covs, [v * 100 for v in sel["curves"][key]], st, ms=4, label=label)
        ax[3].axhline(sel["baseline_success"] * 100, color="gray", lw=0.8, ls="--")
        ax[3].set_xlabel("coverage (fraction executed)")
        ax[3].set_ylabel("retained success (%)")
        ax[3].set_title("(d) Selective autonomy")
        ax[3].legend(fontsize=7)
        ax[3].grid(alpha=0.3)
        ax[3].invert_xaxis()
    plt.tight_layout()
    plt.savefig(RUN / "fig_trigger.png", dpi=140)
    print("saved", RUN / "fig_trigger.png")


if __name__ == "__main__":
    main()
