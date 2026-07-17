"""Analyze an MVP run: bootstrap CIs on signal AUROCs + risk-coverage figure.

Usage: python experiments/analyze_mvp.py results/mvp_pusht_v1
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/mvp_pusht_v1")
SIGNALS = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin", "disp_all", "disp_first"]
BCOH = [s for s in SIGNALS if s.startswith("bcoh")]


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(scores, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    vals = []
    idx = np.arange(len(scores))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if labels[b].sum() == 0 or (~labels[b]).sum() == 0:
            continue
        vals.append(auroc(scores[b], labels[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    data = pickle.load(open(RUN / "baseline_logs.pkl", "rb"))
    logs, success = data["logs"], data["success"]
    fail_by_ep = {ep: (not ok) for ep, ok in enumerate(success)}
    recs = [r for r in logs if any(r[s] > 0 for s in BCOH)]
    step_fail = np.array([fail_by_ep[r["episode"]] for r in recs], bool)
    eps = sorted({r["episode"] for r in recs})
    ep_fail = np.array([fail_by_ep[ep] for ep in eps], bool)

    print(
        f"episodes={len(success)} success={np.mean(success) * 100:.1f}% "
        f"failed_eps_with_signal={int(ep_fail.sum())} replans={len(recs)}\n"
    )
    print(f"{'signal':16s} {'ep_auroc_max':>22s} {'ep_auroc_mean':>22s} {'step_auroc':>12s}")
    table = {}
    for s in SIGNALS:
        ep_max = np.array([max(r[s] for r in recs if r["episode"] == ep) for ep in eps])
        ep_mean = np.array([np.mean([r[s] for r in recs if r["episode"] == ep]) for ep in eps])
        step_vals = np.array([r[s] for r in recs])
        a_max, ci_max = auroc(ep_max, ep_fail), boot_ci(ep_max, ep_fail)
        a_mean, ci_mean = auroc(ep_mean, ep_fail), boot_ci(ep_mean, ep_fail)
        a_step = auroc(step_vals, step_fail)
        table[s] = dict(
            ep_auroc_max=a_max,
            ci_max=ci_max,
            ep_auroc_mean=a_mean,
            ci_mean=ci_mean,
            step_auroc=a_step,
        )
        print(
            f"{s:16s} {a_max:.3f} [{ci_max[0]:.2f},{ci_max[1]:.2f}]   "
            f"{a_mean:.3f} [{ci_mean[0]:.2f},{ci_mean[1]:.2f}]   {a_step:.3f}"
        )

    (RUN / "signal_analysis.json").write_text(json.dumps(table, indent=2))

    # risk-coverage figure: best bcoh vs best disp (step level)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def rc(sig):
            v = np.array([r[sig] for r in recs])
            order = np.argsort(v)
            fs = step_fail[order].astype(float)
            covs = np.linspace(1.0, 0.1, 19)
            risks = [fs[: max(1, int(len(v) * c))].mean() for c in covs]
            return covs, risks

        plt.figure(figsize=(5, 4))
        for sig, style in [("bcoh_distmean", "-o"), ("bcoh_distmin", "-s"), ("disp_all", "--^")]:
            covs, risks = rc(sig)
            plt.plot(covs, risks, style, ms=3, label=sig)
        plt.xlabel("coverage (fraction of steps retained)")
        plt.ylabel("failure rate among retained steps")
        plt.title("Risk-coverage: GUARD trigger vs dispersion")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RUN / "risk_coverage.png", dpi=130)
        print(f"\nsaved {RUN / 'risk_coverage.png'}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
