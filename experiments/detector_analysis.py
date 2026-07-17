"""Unified detector analysis from rich logs (raw overlap blocks -> all signals offline).

Signal family (per replan):
  ours (committed-plan anchored):
    bcoh_distmin   min_k ||fresh_k - committed_cont||   (distance to nearest fresh mode)
    bcoh_distmean  mean_k ||fresh_k - committed_cont||
    bcoh_freshmean ||mean_k fresh_k - committed_cont||
  STAC-family (fresh-vs-fresh set distance, Sentinel/Agia et al. 2024):
    stac_energy    energy distance between consecutive fresh sets (~MMD, distance kernel)
    stac_mmd       MMD^2 with RBF kernel (median-heuristic bandwidth from calibration)
    stac_chamfer   symmetric Chamfer
  variance:  disp_all, disp_first
  OOD:       ood_knn (kNN dist of obs embedding to calibration-success bank)
  fusion:    fused_max = max(robust-z(bcoh_distmin), robust-z(ood_knn))
  null:      null_length (score = replan count; controls for episode-length leakage
             since successful Push-T episodes terminate early)

Aggregations per episode: max, mean, cusum (cumulative sum = online alarm semantics;
Sentinel uses cusum). All headline metrics on a held-out test split; calibration uses
ONLY successful episodes (split-conformal quantile with finite-sample correction).

Run: python experiments/detector_analysis.py results/logs_official
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/logs_official")
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
FPRS = [0.05, 0.10, 0.20]
KS = [2, 4, 8]
AGGS = ["max", "mean", "cusum"]


# ---------- metrics ----------
def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return float("nan")
    o = np.argsort(np.concatenate([p, n]))
    r = np.empty(len(o))
    r[o] = np.arange(1, len(o) + 1)
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def boot_ci(s, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    idx = np.arange(len(s))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() and (~y[b]).sum():
            vals.append(auroc(s[b], y[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def tpr_at_fpr(s, y, fpr):
    s, y = np.asarray(s, float), np.asarray(y).astype(bool)
    neg = np.sort(s[~y])[::-1]
    if not len(neg) or not y.sum():
        return float("nan")
    k = max(0, min(len(neg) - 1, int(np.floor(fpr * len(neg)))))
    return float((s[y] > neg[k]).mean())


# ---------- set distances ----------
def pdist_sets(A, B):
    """A:(K,m,a) B:(K,m,a) -> (K,K) mean-over-time L2."""
    return np.linalg.norm(A[:, None] - B[None], axis=-1).mean(-1)


def energy_dist(A, B):
    return float(2 * pdist_sets(A, B).mean() - pdist_sets(A, A).mean() - pdist_sets(B, B).mean())


def chamfer(A, B):
    d = pdist_sets(A, B)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def mmd_rbf(A, B, gamma):
    a = A.reshape(len(A), -1)
    b = B.reshape(len(B), -1)

    def k(x, y):
        d2 = ((x[:, None] - y[None]) ** 2).sum(-1)
        return np.exp(-gamma * d2)

    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def main():
    data = pickle.load(open(RUN / "rich_logs.pkl", "rb"))
    success, records = data["success"], data["records"]
    n_eps = len(success)
    fail = {e: (not success[e]) for e in range(n_eps)}

    by_ep = defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r)
    for e in by_ep:
        by_ep[e].sort(key=lambda r: r["t"])
    eps_all = sorted(by_ep)

    # ---- split (episode-level) ----
    perm = list(eps_all)
    np.random.default_rng(0).shuffle(perm)
    half = len(perm) // 2
    cal_eps = [e for e in perm[:half] if not fail[e]]
    test_eps = sorted(perm[half:])

    # ---- MMD bandwidth: median heuristic on calibration consecutive-set distances ----
    d2s = []
    for e in cal_eps[:20]:
        rs = by_ep[e]
        for p, c in zip(rs[:-1], rs[1:]):
            a = p["tail_ov"].astype(np.float32).reshape(len(p["tail_ov"]), -1)
            b = c["head_ov"].astype(np.float32).reshape(len(c["head_ov"]), -1)
            d2s.append(((a[:, None] - b[None]) ** 2).sum(-1).flatten())
    med_d2 = float(np.median(np.concatenate(d2s))) + 1e-12
    gamma = 1.0 / med_d2

    # ---- derive per-replan signal series per episode ----
    rng = np.random.default_rng(0)
    K_full = data.get("k", 16)
    subsets = {kk: [rng.choice(K_full, kk, replace=False) for _ in range(10)] for kk in KS}

    def derive(e):
        rs = by_ep[e]
        sig = defaultdict(list)
        for j, r in enumerate(rs):
            sig["disp_all"].append(float(r["disp_all"]))
            sig["disp_first"].append(float(r["disp_first"]))
            if j == 0:
                continue
            p = rs[j - 1]
            head = r["head_ov"].astype(np.float32)  # (K,m,a)
            tail_prev = p["tail_ov"].astype(np.float32)
            cc = tail_prev[0]  # committed = executed sample 0
            d_k = np.linalg.norm(head - cc[None], axis=-1).mean(-1)  # (K,)
            sig["bcoh_distmin"].append(float(d_k.min()))
            sig["bcoh_distmean"].append(float(d_k.mean()))
            sig["bcoh_freshmean"].append(float(np.linalg.norm(head.mean(0) - cc, axis=-1).mean()))
            for kk, subs in subsets.items():
                sig[f"bcoh_distmin_k{kk}"].append(float(np.mean([d_k[s].min() for s in subs])))
            sig["stac_energy"].append(energy_dist(tail_prev, head))
            sig["stac_chamfer"].append(chamfer(tail_prev, head))
            sig["stac_mmd"].append(mmd_rbf(tail_prev, head, gamma))
        sig["null_length"] = [float(j) for j in range(1, len(rs))]
        return sig

    derived = {e: derive(e) for e in eps_all}

    # ---- OOD from embeddings ----
    bank = np.concatenate(
        [np.stack([r["emb"].astype(np.float32) for r in by_ep[e]]) for e in cal_eps]
    )
    for e in eps_all:
        E = np.stack([r["emb"].astype(np.float32) for r in by_ep[e]])
        d = np.linalg.norm(E[:, None, :] - bank[None], axis=-1)
        derived[e]["ood_knn"] = np.sort(d, axis=1)[:, :5].mean(1).tolist()

    # ---- fusion (robust-z stats from calibration) ----
    def rz_stats(sig):
        v = np.concatenate([derived[e][sig] for e in cal_eps if len(derived[e][sig])])
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) + 1e-8
        return med, mad

    mb, sb = rz_stats("bcoh_distmin")
    mo, so = rz_stats("ood_knn")
    for e in eps_all:
        bs = derived[e]["bcoh_distmin"]
        os_ = derived[e]["ood_knn"][1 : len(bs) + 1]
        derived[e]["fused_max"] = [max((b - mb) / sb, (o - mo) / so) for b, o in zip(bs, os_)]

    SIGNALS = [
        "bcoh_distmin",
        "bcoh_distmean",
        "bcoh_freshmean",
        "stac_energy",
        "stac_mmd",
        "stac_chamfer",
        "disp_all",
        "disp_first",
        "ood_knn",
        "fused_max",
        "null_length",
    ] + [f"bcoh_distmin_k{kk}" for kk in KS]

    def agg_score(vals, how):
        if not len(vals):
            return None
        v = np.asarray(vals, float)
        return {"max": float(v.max()), "mean": float(v.mean()), "cusum": float(v.sum())}[how]

    test_ok = [e for e in test_eps if len(derived[e]["bcoh_distmin"])]
    y = np.array([fail[e] for e in test_ok], bool)

    out = {
        "n_episodes": n_eps,
        "success_rate": float(np.mean(success)),
        "n_cal_success_eps": len(cal_eps),
        "n_test_eps": len(test_ok),
        "n_test_fail": int(y.sum()),
        "mmd_gamma": gamma,
        "signals": {},
    }

    for sig in SIGNALS:
        sd = {}
        for how in AGGS:
            sc = np.array([agg_score(derived[e][sig], how) for e in test_ok], float)
            sd[f"auroc_{how}"] = auroc(sc, y)
            if how == "max":
                sd["auroc_max_ci"] = boot_ci(sc, y)
                sd["tpr_at_fpr"] = {str(f): tpr_at_fpr(sc, y, f) for f in FPRS}
        # conformal deferral on max-agg (running-max crossing semantics)
        cal_scores = np.array(
            [agg_score(derived[e][sig], "max") for e in cal_eps if len(derived[e][sig])], float
        )
        mx = np.array([agg_score(derived[e][sig], "max") for e in test_ok], float)
        defer = {}
        for a in ALPHAS:
            q = min(1.0, np.ceil((len(cal_scores) + 1) * (1 - a)) / len(cal_scores))
            tau = float(np.quantile(cal_scores, q))
            deferred = mx >= tau
            kept = ~deferred
            earl = []
            for e, d in zip(test_ok, deferred):
                if d and fail[e]:
                    s = derived[e][sig]
                    first = next(i for i, v in enumerate(s) if v >= tau)
                    earl.append(first / max(1, len(s) - 1))
            defer[str(a)] = dict(
                tau=tau,
                coverage=float(kept.mean()),
                false_defer_on_success=float(deferred[~y].mean()) if (~y).sum() else None,
                defer_on_fail=float(deferred[y].mean()) if y.sum() else None,
                retained_success=float((~y[kept]).mean()) if kept.sum() else None,
                earliness_frac=float(np.mean(earl)) if earl else None,
            )
        sd["conformal_defer"] = defer
        out["signals"][sig] = sd

    # calibration-set-size stability (bcoh_distmin, alpha=0.1)
    all_cal = [
        agg_score(derived[e]["bcoh_distmin"], "max")
        for e in cal_eps
        if len(derived[e]["bcoh_distmin"])
    ]
    mx_ref = np.array([agg_score(derived[e]["bcoh_distmin"], "max") for e in test_ok], float)
    stab = {}
    for n_cal in [10, 20, len(all_cal)]:
        taus, fps = [], []
        for rep in range(30):
            r2 = np.random.default_rng(100 + rep)
            sub = r2.choice(all_cal, min(n_cal, len(all_cal)), replace=False)
            q = min(1.0, np.ceil((len(sub) + 1) * 0.9) / len(sub))
            tau = float(np.quantile(sub, q))
            fps.append(float((mx_ref[~y] >= tau).mean()))
            taus.append(tau)
        stab[str(n_cal)] = dict(
            tau_mean=float(np.mean(taus)),
            tau_std=float(np.std(taus)),
            fp_mean=float(np.mean(fps)),
            fp_std=float(np.std(fps)),
        )
    out["cal_size_stability_alpha0.1"] = stab

    (RUN / "detector_analysis.json").write_text(json.dumps(out, indent=2))

    print(
        f"episodes={n_eps} SR={out['success_rate'] * 100:.1f}% | cal(succ)={len(cal_eps)} "
        f"test={len(test_ok)} (fail={int(y.sum())}) | mmd_gamma={gamma:.3g}\n"
    )
    print(f"{'signal':20s} {'AUROC(max)':>18s} {'(mean)':>7s} {'(cusum)':>8s} {'TPR@0.1':>8s}")
    for sig in SIGNALS:
        d = out["signals"][sig]
        ci = d["auroc_max_ci"]
        print(
            f"{sig:20s} {d['auroc_max']:.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
            f"{d['auroc_mean']:7.3f} {d['auroc_cusum']:8.3f} {d['tpr_at_fpr']['0.1']:8.2f}"
        )
    print("\nconformal defer (bcoh_distmin, max-agg):")
    for a, dd in out["signals"]["bcoh_distmin"]["conformal_defer"].items():
        rs_ = dd["retained_success"]
        print(
            f"  alpha={a}: cov={dd['coverage']:.2f} FPsucc={dd['false_defer_on_success']:.3f} "
            f"TPfail={dd['defer_on_fail']:.2f} retainedSR={rs_ * 100 if rs_ is not None else -1:.1f}% "
            f"earliness={dd['earliness_frac'] if dd['earliness_frac'] is not None else float('nan'):.2f}"
        )
    print("\nsaved", RUN / "detector_analysis.json")


if __name__ == "__main__":
    main()
