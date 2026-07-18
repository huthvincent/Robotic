"""Final report: frozen-recipe analysis pooled across seeds and benchmarks.

Consumes rich_logs.pkl from multiple runs, emits:
  - results/final/report.json          (all numbers)
  - results/final/table_signals.tex    (per-seed + pooled AUROC table rows)
  - results/final/fig_main.png         (4-panel headline figure)
  - console summary

FROZEN RECIPE (decided on dev seed only): signals = bcoh_distmin, bcoh_distmean,
stac_energy, stac_mmd, stac_chamfer (+cusum variants reported for the leakage
exposé), disp_all, ood_knn, null_length; aggregation = episode max (headline),
prefix-W conditional (W=6,8); conformal deferral sweep on bcoh_distmin & stac_energy.

Run: python experiments/final_report.py \
       --runs pusht:dev:results/logs_official pusht:s1000:results/logs_seed1000 \
              pusht:s2000:results/logs_seed2000 square:s1000:results/logs_square_s1000 \
              square:s2000:results/logs_square_s2000
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final"
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
SIGNALS = [
    "bcoh_distmin",
    "bcoh_distmean",
    "stac_energy",
    "stac_mmd",
    "stac_chamfer",
    "disp_all",
    "ood_knn",
    "null_length",
]
PRETTY = {
    "bcoh_distmin": "Boundary coherence (min)",
    "bcoh_distmean": "Boundary coherence (mean)",
    "stac_energy": "STAC (energy)",
    "stac_mmd": "STAC (MMD)",
    "stac_chamfer": "STAC (Chamfer)",
    "disp_all": "Sample dispersion",
    "ood_knn": "Embedding OOD (kNN)",
    "null_length": "Episode length (null)",
}


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


def pdist_sets(A, B):
    return np.linalg.norm(A[:, None] - B[None], axis=-1).mean(-1)


def load_run(path):
    data = pickle.load(open(Path(path) / "rich_logs.pkl", "rb"))
    success, records = data["success"], data["records"]
    fail = {e: (not success[e]) for e in range(len(success))}
    by_ep = defaultdict(list)
    for r in records:
        by_ep[r["episode"]].append(r)
    for e in by_ep:
        by_ep[e].sort(key=lambda r: r["t"])
    eps = sorted(by_ep)
    perm = list(eps)
    np.random.default_rng(0).shuffle(perm)
    half = len(perm) // 2
    cal_eps = [e for e in perm[:half] if not fail[e]]
    test_eps = sorted(perm[half:])

    # MMD bandwidth from calibration
    d2s = []
    for e in cal_eps[:20]:
        rs = by_ep[e]
        for j in range(1, len(rs)):
            a = rs[j - 1]["tail_ov"].astype(np.float32).reshape(16, -1)
            b = rs[j]["head_ov"].astype(np.float32).reshape(16, -1)
            d2s.append(((a[:, None] - b[None]) ** 2).sum(-1).flatten())
    gamma = 1.0 / (float(np.median(np.concatenate(d2s))) + 1e-12)
    # OOD bank/quantile split: build the embedding bank from HALF the calibration
    # successes and calibrate quantiles on the OTHER half, so calibration-episode
    # distances are out-of-sample like test episodes (no in-sample bias).
    bank_eps = cal_eps[::2]
    quant_eps = [e for e in cal_eps if e not in set(bank_eps)]
    bank = np.concatenate(
        [np.stack([r["emb"].astype(np.float32) for r in by_ep[e]]) for e in bank_eps]
    )

    def derive(e):
        rs = by_ep[e]
        sig = defaultdict(list)
        for j in range(1, len(rs)):
            head = rs[j]["head_ov"].astype(np.float32)
            tail_prev = rs[j - 1]["tail_ov"].astype(np.float32)
            cc = tail_prev[0]
            d_k = np.linalg.norm(head - cc[None], axis=-1).mean(-1)
            sig["bcoh_distmin"].append(float(d_k.min()))
            sig["bcoh_distmean"].append(float(d_k.mean()))
            d_ab = pdist_sets(tail_prev, head)
            d_aa = pdist_sets(tail_prev, tail_prev)
            d_bb = pdist_sets(head, head)
            sig["stac_energy"].append(float(2 * d_ab.mean() - d_aa.mean() - d_bb.mean()))
            sig["stac_chamfer"].append(float(0.5 * (d_ab.min(1).mean() + d_ab.min(0).mean())))
            a2 = tail_prev.reshape(len(tail_prev), -1)
            b2 = head.reshape(len(head), -1)

            def rbf(x, y):
                return np.exp(-gamma * ((x[:, None] - y[None]) ** 2).sum(-1))

            sig["stac_mmd"].append(
                float(rbf(a2, a2).mean() + rbf(b2, b2).mean() - 2 * rbf(a2, b2).mean())
            )
            sig["disp_all"].append(float(rs[j]["disp_all"]))
        sig["null_length"] = [float(j) for j in range(1, len(rs))]
        E = np.stack([r["emb"].astype(np.float32) for r in by_ep[e]])[1:]
        d = np.linalg.norm(E[:, None, :] - bank[None], axis=-1)
        sig["ood_knn"] = np.sort(d, axis=1)[:, :5].mean(1).tolist()
        return sig

    derived = {e: derive(e) for e in eps}
    return dict(
        success=success,
        fail=fail,
        derived=derived,
        by_ep=by_ep,
        cal_eps=cal_eps,
        quant_eps=quant_eps,
        test_eps=test_eps,
    )


def analyze_run(run):
    fail, derived, by_ep = run["fail"], run["derived"], run["by_ep"]
    cal_eps, test_eps = run["cal_eps"], run["test_eps"]
    test_ok = [e for e in test_eps if len(derived[e]["bcoh_distmin"])]
    y = np.array([fail[e] for e in test_ok], bool)
    res = {
        "n_test": len(test_ok),
        "n_fail": int(y.sum()),
        "sr": float(np.mean(run["success"])),
        "signals": {},
        "prefix": {},
        "defer": {},
    }
    for s in SIGNALS:
        mx = np.array([max(derived[e][s]) for e in test_ok])
        cs = np.array([sum(derived[e][s]) for e in test_ok])
        res["signals"][s] = {
            "auroc_max": auroc(mx, y),
            "ci": boot_ci(mx, y),
            "auroc_cusum": auroc(cs, y),
        }
    # prefix-W sweep with CIs + eligible counts (defends against W cherry-picking)
    for s in ["bcoh_distmin", "stac_energy", "stac_mmd", "disp_all", "ood_knn"]:
        for w in (4, 6, 8, 10, 12):
            elig = [e for e in test_ok if len(derived[e][s]) >= w]
            yv = np.array([fail[e] for e in elig], bool)
            pv = np.array([max(derived[e][s][:w]) for e in elig])
            res["prefix"][f"{s}_W{w}"] = auroc(pv, yv)
            if s == "bcoh_distmin":
                res.setdefault("prefix_meta", {})[f"W{w}"] = dict(n=len(elig), n_fail=int(yv.sum()))
        res.setdefault("prefix_ci", {})[s] = {
            f"W{w}": boot_ci(
                np.array([max(derived[e][s][:w]) for e in test_ok if len(derived[e][s]) >= w]),
                np.array([fail[e] for e in test_ok if len(derived[e][s]) >= w], bool),
            )
            for w in (4, 6, 8, 10, 12)
        }
    # K-ablation with CI (dev-seed only meaningful; compute where dist arrays present)
    kabl = {}
    for kk in (2, 4, 8, 16):
        rng2 = np.random.default_rng(kk)
        subs = [rng2.choice(16, kk, replace=False) for _ in range(10)] if kk < 16 else None
        vals = []
        for e in test_ok:
            rs = by_ep[e]
            ep = []
            for j in range(1, len(rs)):
                head = rs[j]["head_ov"].astype(np.float32)
                cc = rs[j - 1]["tail_ov"].astype(np.float32)[0]
                d_k = np.linalg.norm(head - cc[None], axis=-1).mean(-1)
                if subs is None:
                    ep.append(float(d_k.min()))
                else:
                    ep.append(float(np.mean([d_k[ss].min() for ss in subs])))
            vals.append(max(ep) if ep else 0.0)
        vals = np.array(vals)
        kabl[f"K{kk}"] = {"auroc": auroc(vals, y), "ci": boot_ci(vals, y)}
    res["k_ablation"] = kabl
    for s in ["bcoh_distmin", "stac_energy", "ood_knn"]:
        cal_pool = run["quant_eps"] if s == "ood_knn" else cal_eps
        cal_scores = np.array([max(derived[e][s]) for e in cal_pool if len(derived[e][s])])
        mx = np.array([max(derived[e][s]) for e in test_ok])
        curve = []
        for a in ALPHAS:
            q = min(1.0, np.ceil((len(cal_scores) + 1) * (1 - a)) / len(cal_scores))
            tau = float(np.quantile(cal_scores, q))
            kept = mx < tau
            curve.append(
                dict(
                    alpha=a,
                    coverage=float(kept.mean()),
                    false_defer=float((mx[~y] >= tau).mean()) if (~y).sum() else None,
                    retained_sr=float((~y[kept]).mean()) if kept.sum() else None,
                )
            )
        res["defer"][s] = curve
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", required=True, help="entries like pusht:dev:results/logs_official"
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_res = {}
    pooled = defaultdict(lambda: dict(mx=defaultdict(list), y=[]))
    for entry in args.runs:
        bench, name, path = entry.split(":", 2)
        if not (Path(path) / "rich_logs.pkl").exists():
            print(f"SKIP {entry} (no rich_logs.pkl)")
            continue
        print(f"loading {entry} ...", flush=True)
        run = load_run(path)
        res = analyze_run(run)
        all_res[f"{bench}:{name}"] = res
        # pool test episodes across seeds per benchmark
        derived, fail = run["derived"], run["fail"]
        test_ok = [e for e in run["test_eps"] if len(derived[e]["bcoh_distmin"])]
        pooled[bench]["y"].extend([fail[e] for e in test_ok])
        for s in SIGNALS:
            pooled[bench]["mx"][s].extend([max(derived[e][s]) for e in test_ok])

    pooled_res = {}
    for bench, d in pooled.items():
        y = np.array(d["y"], bool)
        pooled_res[bench] = {}
        for s in SIGNALS:
            mx = np.array(d["mx"][s])
            pooled_res[bench][s] = {
                "auroc_max": auroc(mx, y),
                "ci": boot_ci(mx, y),
                "n": len(y),
                "n_fail": int(y.sum()),
            }

    # seed-level mean +/- std (avoids fake n=450 bootstrap; honest for multi-seed benches)
    seed_stats = {}
    by_bench_runs = defaultdict(list)
    for k in all_res:
        by_bench_runs[k.split(":", 1)[0]].append(k)
    for bench, keys in by_bench_runs.items():
        seed_stats[bench] = {"n_seeds": len(keys), "signals": {}}
        for s in SIGNALS:
            vals = [all_res[k]["signals"][s]["auroc_max"] for k in keys]
            seed_stats[bench]["signals"][s] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "per_seed": {k: round(all_res[k]["signals"][s]["auroc_max"], 3) for k in keys},
            }

    report = {"runs": all_res, "pooled": pooled_res, "seed_stats": seed_stats}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))

    # LaTeX table
    lines = []
    benches = sorted(pooled_res)
    for s in SIGNALS:
        cells = [PRETTY[s]]
        for b in benches:
            for key in all_res:
                pass
            p = pooled_res[b][s]
            cells.append(f"{p['auroc_max']:.3f} [{p['ci'][0]:.2f},{p['ci'][1]:.2f}]")
        lines.append(" & ".join(cells) + r" \\")
    (OUT / "table_signals.tex").write_text("\n".join(lines) + "\n")

    # console
    print("\n===== PER-RUN =====")
    for k, r in all_res.items():
        print(f"\n{k}: SR={r['sr'] * 100:.1f}% test={r['n_test']} fail={r['n_fail']}")
        for s in SIGNALS:
            d = r["signals"][s]
            print(
                f"  {s:16s} max={d['auroc_max']:.3f} [{d['ci'][0]:.2f},{d['ci'][1]:.2f}] "
                f"cusum={d['auroc_cusum']:.3f}"
            )
        print("  prefix:", {k2: round(v, 3) for k2, v in r["prefix"].items()})
        for s, curve in r["defer"].items():
            pts = [
                f"a={c['alpha']}: cov={c['coverage']:.2f} SR={c['retained_sr'] * 100:.1f}%"
                for c in curve
                if c["retained_sr"] is not None
            ]
            print(f"  defer[{s}]: " + " | ".join(pts[:4]))
    print("\n===== POOLED =====")
    for b in benches:
        print(
            f"\n[{b}] n={pooled_res[b]['bcoh_distmin']['n']} "
            f"fail={pooled_res[b]['bcoh_distmin']['n_fail']}"
        )
        for s in SIGNALS:
            p = pooled_res[b][s]
            print(f"  {s:16s} {p['auroc_max']:.3f} [{p['ci'][0]:.2f},{p['ci'][1]:.2f}]")
    print("\nsaved", OUT / "report.json", "and table_signals.tex")


if __name__ == "__main__":
    main()
