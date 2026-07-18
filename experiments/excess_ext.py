"""Reviewer-10 (2): tighter inference on the central quantitative claim.

Merges each strong Push-T run's original 300-episode collection with a fresh
600-episode extension (disjoint env seeds 300-899, disjoint sampling streams)
and recomputes the permutation-null excess for the coherence family on the
3x-larger set. Reported as a supplement-only robustness check; headline numbers
in the paper remain the original 300-episode protocol.

Episode ids are collection seeds, so records concatenate without renumbering
(original 0-299, extension 300-899) and the success list indexes align.

Output: results/final/excess_ext.json
Run: python experiments/excess_ext.py
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.final_report import load_run, auroc, boot_ci  # noqa: E402

PAIRS = [
    ("pusht-strong:dev", "results/logs_official", "results/logs_official_ext"),
    ("pusht-strong:s1000", "results/logs_seed1000", "results/logs_seed1000_ext"),
    ("pusht-strong:s2000", "results/logs_seed2000", "results/logs_seed2000_ext"),
]
SIGNALS = ["bcoh_distmin", "stac_energy", "stac_mmd"]
N_PERM = 500

out = {}
for name, base_dir, ext_dir in PAIRS:
    if not (Path(ext_dir) / "rich_logs.pkl").exists():
        print(f"skip {name}: extension not collected yet")
        continue
    a = pickle.load(open(Path(base_dir) / "rich_logs.pkl", "rb"))
    b = pickle.load(open(Path(ext_dir) / "rich_logs.pkl", "rb"))
    merged = dict(a)
    merged["success"] = list(a["success"]) + list(b["success"])
    merged["records"] = list(a["records"]) + list(b["records"])
    with tempfile.TemporaryDirectory() as td:
        pickle.dump(merged, open(Path(td) / "rich_logs.pkl", "wb"))
        run = load_run(td)
    fail, derived = run["fail"], run["derived"]
    test_ok = [e for e in run["test_eps"] if len(derived[e]["bcoh_distmin"])]
    y = np.array([fail[e] for e in test_ok], bool)
    rng = np.random.default_rng(0)
    res = {"n_test": len(test_ok), "n_fail": int(y.sum()), "sr": float(np.mean(merged["success"]))}
    for s in SIGNALS:
        series = [np.asarray(derived[e][s], float) for e in test_ok]
        lens = [len(v) for v in series]
        raw_max = np.array([v.max() for v in series])
        raw = auroc(raw_max, y)
        ci = boot_ci(raw_max, y)
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
        p = float((1 + int((nulls >= raw).sum())) / (1 + len(nulls)))
        res[s] = dict(
            auroc=round(raw, 3),
            ci=[round(ci[0], 3), round(ci[1], 3)],
            null_mean=round(float(nulls.mean()), 3),
            excess=round(raw - float(nulls.mean()), 3),
            p_perm=round(p, 4),
        )
        print(
            f"{name} {s}: {raw:.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
            f"null {nulls.mean():.3f} excess {raw - nulls.mean():+.3f} p={p:.3f}",
            flush=True,
        )
    out[name] = res

(ROOT / "results/final/excess_ext.json").write_text(json.dumps(out, indent=2))
print("saved results/final/excess_ext.json")
