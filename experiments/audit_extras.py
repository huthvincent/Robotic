"""Reviewer-mandated extensions, all computed offline from logged data.

1. FIPER-style baseline (Roemer et al. 2025): RND on observation embeddings
   (predictor MLP trained to match a frozen random target on calibration-success
   embeddings) + ACE (action-chunk entropy proxy from the K logged samples via a
   Kozachenko-Leonenko nearest-neighbor estimate); combined score = max of
   robust-z (approximation of their dual-threshold rule; noted in paper).
2. ActProbe-style features (Huang et al. 2026): TCE (overlap MSE == freshmean
   boundary coherence) and ACM (chunk magnitude), unsupervised; plus a
   supervised logistic per-step upper bound with ORACLE labels (5-fold CV on the
   test split) - explicitly an upper bound, not a deployable monitor.
3. Time-stratified permutation null: shuffle scores across episodes WITHIN each
   replan index (preserves both lengths and time marginals).
4. Timeout frontier: all T0 thresholds -> (coverage, retained SR, handoff).
5. Risk-coverage-utility: per alpha -> coverage, retained, false-defer, deferred
   -failure recall, median handoff, replans saved, expected utility for fallback
   success probabilities {0.3, 0.6, 0.9}.
6. Calibration robustness: n_cal in {10,20,40,max} x 50 splits -> realized
   false-deferral mean +/- sd.
7. Push-T failure taxonomy (automatic): near_miss (max coverage>=0.8) / stalled
   (bottom-quartile late motion) / oscillating (top-quartile mean coherence) /
   other; per-type detection AUROC (type-vs-success).

Run: python experiments/audit_extras.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.final_report import load_run, auroc  # noqa: E402

RUNS = [
    ("pusht-strong:dev", "results/logs_official"),
    ("pusht-strong:s1000", "results/logs_seed1000"),
    ("pusht-strong:s2000", "results/logs_seed2000"),
    ("pusht-weak:nocrop", "results/logs_pusht_weak_nocrop"),
    ("square-weak:s1000", "results/logs_square_s1000"),
    ("square-weak:s2000", "results/logs_square_s2000"),
    # w2/w3 appended AFTER the original six so the shared torch RNG stream
    # (train_rnd inits) is identical for the first six runs and their numbers
    # reproduce exactly.
    ("pusht-weak:s3000", "results/logs_pusht_weak_s3000"),
    ("pusht-weak:s4000", "results/logs_pusht_weak_s4000"),
    # can-strong appended last (same RNG-preservation rule)
    ("can-strong:s1000", "results/logs_can_s1000"),
    ("pusht-mid:60k", "results/logs_pusht_mid60k"),
]
out = {
    "fiper_style": {},
    "actprobe_style": {},
    "tsnull": {},
    "timeout_frontier": {},
    "utility": {},
    "cal_robust": {},
    "taxonomy": {},
}
torch.manual_seed(0)


def train_rnd(bank_emb: np.ndarray, dim=64, epochs=300):
    tgt = nn.Sequential(nn.Linear(bank_emb.shape[1], 128), nn.ReLU(), nn.Linear(128, dim))
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(bank_emb.shape[1], 128), nn.ReLU(), nn.Linear(128, dim))
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
    X = torch.tensor(bank_emb, dtype=torch.float32)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    with torch.no_grad():
        Y = tgt(Xn)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((pred(Xn) - Y) ** 2).mean()
        loss.backward()
        opt.step()

    def score(e):
        xe = (torch.tensor(e, dtype=torch.float32) - mu) / sd
        with torch.no_grad():
            return ((pred(xe) - tgt(xe)) ** 2).mean(-1).numpy()

    return score


def ace_entropy(head_ov: np.ndarray) -> float:
    """KL nearest-neighbor entropy proxy from K flattened chunk samples."""
    x = head_ov.reshape(len(head_ov), -1).astype(np.float32)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.log(d.min(1) + 1e-9).mean())


def robust_z(v, med, mad):
    return (np.asarray(v) - med) / (mad + 1e-9)


for name, path in RUNS:
    if not (Path(path) / "rich_logs.pkl").exists():
        continue
    run = load_run(path)
    fail, derived, by_ep = run["fail"], run["derived"], run["by_ep"]
    cal_eps, test_eps = run["cal_eps"], run["test_eps"]
    test_ok = [e for e in test_eps if len(derived[e]["bcoh_distmin"])]
    y = np.array([fail[e] for e in test_ok], bool)
    bank_eps = cal_eps[::2]
    quant_eps = [e for e in cal_eps if e not in set(bank_eps)]

    # ---------- 1) FIPER-style ----------
    bank_emb = np.concatenate(
        [np.stack([r["emb"].astype(np.float32) for r in by_ep[e]]) for e in bank_eps]
    )
    rnd = train_rnd(bank_emb)

    def series_fiper(e):
        E = np.stack([r["emb"].astype(np.float32) for r in by_ep[e]])[1:]
        rnd_s = rnd(E)
        ace_s = np.array([ace_entropy(r["head_ov"].astype(np.float32)) for r in by_ep[e][1:]])
        return rnd_s, ace_s

    cal_rnd = np.concatenate([series_fiper(e)[0] for e in quant_eps])
    cal_ace = np.concatenate([series_fiper(e)[1] for e in quant_eps])
    mr, sr_ = np.median(cal_rnd), np.median(np.abs(cal_rnd - np.median(cal_rnd)))
    ma, sa = np.median(cal_ace), np.median(np.abs(cal_ace - np.median(cal_ace)))
    fiper_ser = {}
    for e in test_ok + quant_eps:
        r_s, a_s = series_fiper(e)
        fiper_ser[e] = np.maximum(robust_z(r_s, mr, sr_), robust_z(a_s, ma, sa))
    mx = np.array([fiper_ser[e].max() for e in test_ok])
    w8 = [e for e in test_ok if len(fiper_ser[e]) >= 8]
    pv = np.array([fiper_ser[e][:8].max() for e in w8])
    yv = np.array([fail[e] for e in w8], bool)
    # deferral at alpha=0.3
    calmax = np.array([fiper_ser[e].max() for e in quant_eps])
    q = min(1.0, np.ceil((len(calmax) + 1) * 0.7) / len(calmax))
    tau = float(np.quantile(calmax, q))
    kept = mx < tau
    out["fiper_style"][name] = dict(
        auroc_max=round(auroc(mx, y), 3),
        prefix_w8=round(auroc(pv, yv), 3),
        defer_a03=dict(
            coverage=round(float(kept.mean()), 2),
            retained=round(float((~y[kept]).mean()), 3) if kept.sum() else None,
            false_defer=round(float((mx[~y] >= tau).mean()), 3) if (~y).sum() else None,
        ),
    )

    # ---------- 2) ActProbe-style ----------
    def tce_acm(e):
        tce = np.asarray(
            derived[e]["bcoh_freshmean"]
            if "bcoh_freshmean" in derived[e]
            else derived[e]["bcoh_distmean"],
            float,
        )
        acm = np.array(
            [np.linalg.norm(r["head_ov"].astype(np.float32), axis=-1).mean() for r in by_ep[e][1:]]
        )
        return tce, acm

    mx_tce = np.array([tce_acm(e)[0].max() for e in test_ok])
    mx_acm = np.array([tce_acm(e)[1].max() for e in test_ok])
    # oracle-labeled supervised upper bound: per-step logistic, 5-fold CV over episodes
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    ep_arr = np.array(test_ok)
    ep_score = np.zeros(len(test_ok))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(ep_arr):
        Xtr, ytr = [], []
        for i in tr:
            t_, a_ = tce_acm(ep_arr[i])
            Xtr.append(np.stack([t_, a_], 1))
            ytr += [fail[ep_arr[i]]] * len(t_)
        clf = LogisticRegression(max_iter=1000).fit(np.concatenate(Xtr), np.array(ytr))
        for i in te:
            t_, a_ = tce_acm(ep_arr[i])
            ep_score[i] = clf.predict_proba(np.stack([t_, a_], 1))[:, 1].max()
    out["actprobe_style"][name] = dict(
        tce_max=round(auroc(mx_tce, y), 3),
        acm_max=round(auroc(mx_acm, y), 3),
        oracle_logistic_max=round(auroc(ep_score, y), 3),
    )

    # ---------- 3) time-stratified permutation null (bcoh + ood) ----------
    rng = np.random.default_rng(0)
    out["tsnull"][name] = {}
    for s in ("bcoh_distmin", "ood_knn"):
        series = [np.asarray(derived[e][s], float) for e in test_ok]
        max_t = max(len(v) for v in series)
        by_t = [[] for _ in range(max_t)]
        for v in series:
            for t, x in enumerate(v):
                by_t[t].append(x)
        nulls = []
        for _ in range(300):
            shuf = [rng.permutation(col) for col in by_t]
            ptr = [0] * max_t
            mxs = []
            for v in series:
                vals = [shuf[t][ptr[t]] for t in range(len(v))]
                for t in range(len(v)):
                    ptr[t] += 1
                mxs.append(max(vals))
            nulls.append(auroc(np.array(mxs), y))
        raw = auroc(np.array([v.max() for v in series]), y)
        out["tsnull"][name][s] = dict(
            raw=round(raw, 3),
            null_mean=round(float(np.mean(nulls)), 3),
            excess=round(raw - float(np.mean(nulls)), 3),
        )

    # ---------- 4-6) frontier / utility / calibration robustness (strong runs + weak ood) ----------
    sig = "ood_knn" if name.startswith(("pusht-weak", "pusht-mid", "square")) else "bcoh_distmin"
    ser = {e: np.asarray(derived[e][sig], float) for e in test_ok}
    n_rep = {e: len(ser[e]) for e in test_ok}
    # timeout frontier
    frontier = []
    for T0 in sorted(set(n_rep.values())):
        kept_m = np.array([n_rep[e] - 1 <= T0 for e in test_ok])
        if kept_m.sum() in (0, len(test_ok)):
            continue
        frontier.append(
            dict(
                T0=int(T0),
                coverage=round(float(kept_m.mean()), 2),
                retained=round(float((~y[kept_m]).mean()), 3),
            )
        )
    out["timeout_frontier"][name] = frontier
    # utility table
    calmax2 = np.array(
        [
            np.asarray(derived[e][sig], float).max()
            for e in (quant_eps if sig == "ood_knn" else cal_eps)
            if len(derived[e][sig])
        ]
    )
    rows = []
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        q = min(1.0, np.ceil((len(calmax2) + 1) * (1 - a)) / len(calmax2))
        tau = float(np.quantile(calmax2, q))
        mx2 = np.array([ser[e].max() for e in test_ok])
        kept_m = mx2 < tau
        first = {e: next((i for i, v in enumerate(ser[e]) if v >= tau), None) for e in test_ok}
        hf = [first[e] / max(1, n_rep[e] - 1) for e in test_ok if first[e] is not None and fail[e]]
        saved = [n_rep[e] - 1 - first[e] for e in test_ok if first[e] is not None and fail[e]]
        row = dict(
            alpha=a,
            coverage=round(float(kept_m.mean()), 2),
            retained=round(float((~y[kept_m]).mean()), 3) if kept_m.sum() else None,
            false_defer=round(float((mx2[~y] >= tau).mean()), 3),
            fail_recall=round(float((mx2[y] >= tau).mean()), 3),
            median_handoff=round(float(np.median(hf)), 2) if hf else None,
            replans_saved=round(float(np.mean(saved)), 1) if saved else None,
        )
        for pfb in (0.3, 0.6, 0.9):
            cov = row["coverage"]
            ret = row["retained"] or 0
            row[f"EU_pfb{pfb}"] = round(ret * cov + pfb * (1 - cov), 3)
        rows.append(row)
    out["utility"][name] = rows
    # calibration robustness
    all_cal = list(calmax2)
    rob = {}
    for n_cal in (10, 20, 40, len(all_cal)):
        fps = []
        for rep in range(50):
            r2 = np.random.default_rng(rep)
            sub = r2.choice(all_cal, min(n_cal, len(all_cal)), replace=False)
            q = min(1.0, np.ceil((len(sub) + 1) * 0.7) / len(sub))
            tau = float(np.quantile(sub, q))
            mx2 = np.array([ser[e].max() for e in test_ok])
            fps.append(float((mx2[~y] >= tau).mean()))
        rob[str(n_cal)] = dict(
            fp_mean=round(float(np.mean(fps)), 3), fp_sd=round(float(np.std(fps)), 3)
        )
    out["cal_robust"][name] = rob

    # ---------- 7) failure taxonomy (Push-T only) ----------
    if name.startswith("pusht"):
        types = {}
        succ_scores = {
            s2: [np.asarray(derived[e][s2], float).max() for e in test_ok if not fail[e]]
            for s2 in ("bcoh_distmin", "stac_mmd", "ood_knn")
        }
        feats = {}
        for e in test_ok:
            if not fail[e]:
                continue
            rs = by_ep[e]
            maxrew = max(r["reward"] for r in rs)
            pos = np.stack([r["agent_pos"] for r in rs])
            late = np.abs(np.diff(pos[-8:], axis=0)).sum()
            meanb = float(np.mean(derived[e]["bcoh_distmin"]))
            feats[e] = (maxrew, late, meanb)
        if feats:
            lates = np.array([v[1] for v in feats.values()])
            meanbs = np.array([v[2] for v in feats.values()])
            q_late = np.quantile(lates, 0.25)
            q_b = np.quantile(meanbs, 0.75)
            for e, (mr_, lt, mb) in feats.items():
                if mr_ >= 0.8:
                    t = "near_miss"
                elif lt <= q_late:
                    t = "stalled"
                elif mb >= q_b:
                    t = "oscillating"
                else:
                    t = "other"
                types.setdefault(t, []).append(e)
            tax = {}
            for t, eps_t in types.items():
                row = {"n": len(eps_t)}
                for s2 in ("bcoh_distmin", "stac_mmd", "ood_knn"):
                    sc = [np.asarray(derived[e][s2], float).max() for e in eps_t]
                    lab = np.array([1] * len(sc) + [0] * len(succ_scores[s2]))
                    row[s2] = round(auroc(np.array(sc + succ_scores[s2]), lab.astype(bool)), 3)
                tax[t] = row
            out["taxonomy"][name] = tax
    print(f"{name}: done", flush=True)

(ROOT / "results/final/extras.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/final/extras.json")
for k in ("fiper_style", "actprobe_style", "tsnull"):
    print(f"\n== {k} ==")
    for n, v in out[k].items():
        print(" ", n, v)
print("\n== taxonomy ==")
for n, v in out["taxonomy"].items():
    print(" ", n, v)
