"""Signal laboratory: prototype detector variants OFFLINE on the dev seed.

Protocol discipline: this script is run on results/logs_official (DEV data) only.
The winning recipe is frozen, then confirmed once on held-out seeds + robomimic.

Evaluations:
  (a) episode-level AUROC, max-agg (with length-leakage caveat)
  (b) prefix-W conditional AUROC: at replan W, among episodes still running,
      predict eventual failure from the first W replan signals (no length leak)
  (c) prefix-W deferral: retained success among kept-and-still-running episodes

Signal variants (all from logged blocks; d_k = dist of fresh k to committed cont):
  bcoh_distmin        min_k d_k
  bcoh_distmean       mean_k d_k
  plan_nll            -log( (1/K) sum_k exp(-d_k^2 / h) )   [KDE likelihood of committed plan
                      under fresh distribution; h = median d_k^2 over calibration]
  bcoh_znorm          (distmin - mu_k(d)) / std_k(d)  [is committed an outlier vs fresh spread]
  bcoh_ratio          distmin / (mean pairwise fresh dist + eps)   [normalized by local diversity]
  bcoh_early          distance on FIRST overlap step only (immediate divergence)
  stac_energy / stac_mmd / stac_chamfer  (faithful fresh-vs-fresh)
  stac_energy_k2      energy distance with 2-sample subsets (cost-matched to cheap regime)
  disp_all, disp_first, ood_knn (context)
  ewma_<sig>          EWMA(0.3) temporal smoothing of a base signal (burst robustness)

Run: python experiments/signal_lab.py results/logs_official
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/logs_official")
WS = [4, 6, 8, 10, 12]


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return float("nan")
    o = np.argsort(np.concatenate([p, n]))
    r = np.empty(len(o))
    r[o] = np.arange(1, len(o) + 1)
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def boot_ci(s, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    idx = np.arange(len(s))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() and (~y[b]).sum():
            vals.append(auroc(s[b], y[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pdist_sets(A, B):
    return np.linalg.norm(A[:, None] - B[None], axis=-1).mean(-1)


def main():
    data = pickle.load(open(RUN / "rich_logs.pkl", "rb"))
    success, records = data["success"], data["records"]
    fail = {e: (not success[e]) for e in range(len(success))}
    by_ep = defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r)
    for e in by_ep:
        by_ep[e].sort(key=lambda r: r["t"])
    eps_all = sorted(by_ep)

    perm = list(eps_all)
    np.random.default_rng(0).shuffle(perm)
    half = len(perm) // 2
    cal_eps = [e for e in perm[:half] if not fail[e]]
    test_eps = sorted(perm[half:])

    # ---- KDE bandwidth + OOD bank from calibration ----
    d2_cal = []
    for e in cal_eps:
        rs = by_ep[e]
        for j in range(1, len(rs)):
            head = rs[j]["head_ov"].astype(np.float32)
            cc = rs[j - 1]["tail_ov"].astype(np.float32)[0]
            d2_cal.append((np.linalg.norm(head - cc[None], axis=-1).mean(-1) ** 2))
    h_kde = float(np.median(np.concatenate(d2_cal))) + 1e-12
    bank = np.concatenate(
        [np.stack([r["emb"].astype(np.float32) for r in by_ep[e]]) for e in cal_eps]
    )
    # MMD gamma
    gamma = 1.0 / h_kde  # same scale family; fine for comparison

    rng = np.random.default_rng(0)
    k2_subs = [rng.choice(16, 2, replace=False) for _ in range(10)]

    def derive(e):
        rs = by_ep[e]
        sig = defaultdict(list)
        for j in range(1, len(rs)):
            head = rs[j]["head_ov"].astype(np.float32)  # (K,m,a)
            tail_prev = rs[j - 1]["tail_ov"].astype(np.float32)
            cc = tail_prev[0]
            d_k = np.linalg.norm(head - cc[None], axis=-1).mean(-1)
            K = len(d_k)
            sig["bcoh_distmin"].append(float(d_k.min()))
            sig["bcoh_distmean"].append(float(d_k.mean()))
            sig["plan_nll"].append(float(-np.log(np.exp(-(d_k**2) / h_kde).mean() + 1e-12)))
            sig["bcoh_znorm"].append(float((d_k.min() - d_k.mean()) / (d_k.std() + 1e-8)))
            pw = pdist_sets(head, head)
            div = pw[np.triu_indices(K, 1)].mean()
            sig["bcoh_ratio"].append(float(d_k.min() / (div + 1e-8)))
            d_first = np.linalg.norm(head[:, 0, :] - cc[None, 0, :], axis=-1)
            sig["bcoh_early"].append(float(d_first.min()))
            d_ab = pdist_sets(tail_prev, head)
            d_aa = pdist_sets(tail_prev, tail_prev)
            d_bb = pw
            sig["stac_energy"].append(float(2 * d_ab.mean() - d_aa.mean() - d_bb.mean()))
            sig["stac_chamfer"].append(float(0.5 * (d_ab.min(1).mean() + d_ab.min(0).mean())))
            a2 = tail_prev.reshape(K, -1)
            b2 = head.reshape(K, -1)

            def rbf(x, y):
                return np.exp(-gamma * ((x[:, None] - y[None]) ** 2).sum(-1))

            sig["stac_mmd"].append(
                float(rbf(a2, a2).mean() + rbf(b2, b2).mean() - 2 * rbf(a2, b2).mean())
            )
            e2 = []
            for ss in k2_subs:
                A, B = tail_prev[ss], head[ss]
                e2.append(
                    2 * pdist_sets(A, B).mean() - pdist_sets(A, A).mean() - pdist_sets(B, B).mean()
                )
            sig["stac_energy_k2"].append(float(np.mean(e2)))
            sig["disp_all"].append(float(rs[j]["disp_all"]))
        # OOD
        E = np.stack([r["emb"].astype(np.float32) for r in by_ep[e]])[1:]
        d = np.linalg.norm(E[:, None, :] - bank[None], axis=-1)
        sig["ood_knn"] = np.sort(d, axis=1)[:, :5].mean(1).tolist()
        return sig

    print("deriving signals...", flush=True)
    derived = {e: derive(e) for e in eps_all}

    BASE = [
        "bcoh_distmin",
        "bcoh_distmean",
        "plan_nll",
        "bcoh_znorm",
        "bcoh_ratio",
        "bcoh_early",
        "stac_energy",
        "stac_mmd",
        "stac_chamfer",
        "stac_energy_k2",
        "disp_all",
        "ood_knn",
    ]

    # EWMA variants of top temporal signals
    def ewma(v, a=0.3):
        out, m = [], 0.0
        for i, x in enumerate(v):
            m = x if i == 0 else a * x + (1 - a) * m
            out.append(m)
        return out

    for s in ["bcoh_distmin", "plan_nll", "stac_energy", "stac_mmd"]:
        for e in eps_all:
            derived[e][f"ewma_{s}"] = ewma(derived[e][s])
    SIGNALS = BASE + [f"ewma_{s}" for s in ["bcoh_distmin", "plan_nll", "stac_energy", "stac_mmd"]]

    test_ok = [e for e in test_eps if len(derived[e]["bcoh_distmin"])]
    y_all = np.array([fail[e] for e in test_ok], bool)
    out = {"episode_max": {}, "prefix": {}}

    print(
        f"\n=== (a) episode-level max-agg AUROC (length-leak caveat) | "
        f"test n={len(test_ok)} fail={int(y_all.sum())} ==="
    )
    rows = []
    for s in SIGNALS:
        mx = np.array([max(derived[e][s]) for e in test_ok])
        a = auroc(mx, y_all)
        ci = boot_ci(mx, y_all)
        out["episode_max"][s] = {"auroc": a, "ci": ci}
        rows.append((a, f"{s:20s} {a:.3f} [{ci[0]:.2f},{ci[1]:.2f}]"))
    for _, line in sorted(rows, reverse=True):
        print(line)

    print("\n=== (b) prefix-W conditional AUROC (still-running episodes at replan W) ===")
    print(f"{'signal':20s} " + " ".join(f"W={w:<4d}" for w in WS))
    for s in SIGNALS:
        vals = []
        for w in WS:
            elig = [e for e in test_ok if len(derived[e][s]) >= w]
            yv = np.array([fail[e] for e in elig], bool)
            pv = np.array([max(derived[e][s][:w]) for e in elig])
            vals.append(auroc(pv, yv))
        out["prefix"][s] = dict(zip(map(str, WS), vals))
        print(f"{s:20s} " + " ".join(f"{v:.3f}" for v in vals))

    # eligible counts per W
    for w in WS:
        elig = [e for e in test_ok if len(derived[e]["bcoh_distmin"]) >= w]
        nf = sum(fail[e] for e in elig)
        print(f"  (W={w}: eligible={len(elig)}, fail={nf})")

    # (c) prefix-8 conformal deferral with retained success
    print("\n=== (c) prefix-8 conformal deferral (decide by ~64 env steps) ===")
    w = 8
    for s in ["bcoh_distmin", "plan_nll", "stac_mmd", "stac_energy", "ewma_plan_nll"]:
        cal_pref = [max(derived[e][s][:w]) for e in cal_eps if len(derived[e][s]) >= w]
        elig = [e for e in test_ok if len(derived[e][s]) >= w]
        yv = np.array([fail[e] for e in elig], bool)
        pv = np.array([max(derived[e][s][:w]) for e in elig])
        line = [f"{s:16s}"]
        for a_t in [0.1, 0.2, 0.3]:
            q = min(1.0, np.ceil((len(cal_pref) + 1) * (1 - a_t)) / len(cal_pref))
            tau = float(np.quantile(cal_pref, q))
            kept = pv < tau
            rs_ = float((~yv[kept]).mean()) if kept.sum() else float("nan")
            line.append(f"a={a_t}: cov={kept.mean():.2f} SR={rs_ * 100:.1f}%")
        print("  " + "  ".join(line))
    base_still = float((~y_all).mean())
    elig8 = [e for e in test_ok if len(derived[e]["bcoh_distmin"]) >= 8]
    y8 = np.array([fail[e] for e in elig8], bool)
    print(
        f"  (reference: SR among still-running-at-W8 = {(~y8).mean() * 100:.1f}%, overall test SR = {base_still * 100:.1f}%)"
    )

    (RUN / "signal_lab.json").write_text(json.dumps(out, indent=2))
    print("\nsaved", RUN / "signal_lab.json")


if __name__ == "__main__":
    main()
