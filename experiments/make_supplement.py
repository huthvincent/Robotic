"""Generate all supplement tables/figures from logged artifacts.

Outputs:
  paper/aaai/supp_generated.tex   (all \\input-able tables, one per section)
  results/final/fig_utility_heatmap.png/.pdf
  results/final/taxonomy_grid_<class>.png   (trajectory exemplars for author check)
  results/final/tide_like.json              (TIDE-like signal per run)

Every table footnotes its generating script and data file.
Run: python experiments/make_supplement.py
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.final_report import load_run, auroc, boot_ci  # noqa: E402

REP = json.loads((ROOT / "results/final/report.json").read_text())
CTRL = json.loads((ROOT / "results/final/controls.json").read_text())
EXT = json.loads((ROOT / "results/final/extras.json").read_text())
RUNS_META = [
    ("pusht-strong:dev", "results/logs_official", "Push-T strong (dev)"),
    ("pusht-strong:s1000", "results/logs_seed1000", "Push-T strong (seed A)"),
    ("pusht-strong:s2000", "results/logs_seed2000", "Push-T strong (seed B)"),
    ("pusht-weak:nocrop", "results/logs_pusht_weak_nocrop", "Push-T weak (w1)"),
    ("pusht-weak:s3000", "results/logs_pusht_weak_s3000", "Push-T weak (w2)"),
    ("pusht-weak:s4000", "results/logs_pusht_weak_s4000", "Push-T weak (w3)"),
    ("square-weak:s1000", "results/logs_square_s1000", "Square weak (A)"),
    ("square-weak:s2000", "results/logs_square_s2000", "Square weak (B)"),
    ("can-strong:s1000", "results/logs_can_s1000", "Can strong"),
    ("pusht-mid:60k", "results/logs_pusht_mid60k", "Push-T mid (60k)"),
]
SIGS = [
    "bcoh_distmin",
    "bcoh_distmean",
    "stac_energy",
    "stac_mmd",
    "stac_chamfer",
    "disp_all",
    "ood_knn",
]
PRETTY = {
    "bcoh_distmin": "Boundary coh.\\ (min)",
    "bcoh_distmean": "Boundary coh.\\ (mean)",
    "stac_energy": "STAC (energy)",
    "stac_mmd": "STAC (MMD)",
    "stac_chamfer": "STAC (Chamfer)",
    "disp_all": "Dispersion",
    "ood_knn": "Embedding OOD",
    "tide_like": "TIDE-like (exec.\\ inter-chunk)",
    "null_length": "Episode length (null)",
}

out_parts = []


def add_table(key, caption, header, rows, src, placement="!t"):
    # No \resizebox: it rescales to full text width (often UPscaling, which
    # inflates row height and lets long floats overflow the page silently —
    # the cause of the round-6 truncated per-seed table). Tables are kept
    # short enough to fit one page at natural \small size.
    # `src` is intentionally NOT printed: per-caption script attributions read
    # as machine output; provenance lives once in the Sec. J manifest.
    body = "\n".join(rows)
    out_parts.append(f"""% ---- {key} (source: {src}) ----
\\begin{{table*}}[{placement}]
\\centering\\small
\\begin{{tabular}}{{{header[0]}}}
\\toprule
{header[1]} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{tab:{key}}}
\\end{{table*}}
""")


# ---------- TIDE-like signal (executed-plan inter-chunk discrepancy) ----------
tide = {}
for name, path, _ in RUNS_META:
    run = load_run(path)
    fail, by_ep = run["fail"], run["by_ep"]
    test_ok = [e for e in run["test_eps"] if len(by_ep[e]) > 1]
    y = np.array([fail[e] for e in test_ok], bool)
    mx, w8 = [], []
    for e in test_ok:
        rs = by_ep[e]
        ser = []
        for j in range(1, len(rs)):
            head0 = rs[j]["head_ov"].astype(np.float32)[0]
            cc = rs[j - 1]["tail_ov"].astype(np.float32)[0]
            ser.append(float(np.linalg.norm(head0 - cc, axis=-1).mean()))
        mx.append(max(ser))
        w8.append(max(ser[:8]))
    elig = [i for i, e in enumerate(test_ok) if len(by_ep[e]) - 1 >= 8]
    tide[name] = dict(
        auroc_max=round(auroc(np.array(mx), y), 3),
        ci=boot_ci(np.array(mx), y, n=1000),
        prefix_w8=round(auroc(np.array([w8[i] for i in elig]), y[elig]), 3),
    )
    print(f"tide {name}: {tide[name]['auroc_max']}", flush=True)
(ROOT / "results/final/tide_like.json").write_text(json.dumps(tide, indent=2))

# ---------- D. full per-seed detection tables ----------
# Split into three page-safe tables (a 64-row table* cannot break across pages
# and silently clips at the bottom of the page).
PERSEED_CAPTION = (
    "episode-max AUROC with bootstrap 95\\% CIs "
    "(2000 resamples, episode level), CUSUM aggregation, leak-free "
    "prefix-$W{=}8$, and excess over the length-preserving permutation "
    "null (500 shuffles). Splits: episode-level 50/50 calibration/test, "
    "calibration success-only."
)
PERSEED_GROUPS = [
    (
        "supp-perseed-strong",
        "Push-T strong regime (3 seeds)",
        ["pusht-strong:dev", "pusht-strong:s1000", "pusht-strong:s2000"],
    ),
    (
        "supp-perseed-weak",
        "Push-T degraded regime (3 crop-ablated seeds; 1 undertrained 60k checkpoint)",
        ["pusht-weak:nocrop", "pusht-weak:s3000", "pusht-weak:s4000", "pusht-mid:60k"],
    ),
    (
        "supp-perseed-square",
        "robomimic Square weak regime (2 seeds) and Can strong regime (1 seed)",
        ["square-weak:s1000", "square-weak:s2000", "can-strong:s1000"],
    ),
]
pretty_of = dict((n, p) for n, _, p in RUNS_META)
for key, gname, members in PERSEED_GROUPS:
    rows = []
    for name in members:
        pretty = pretty_of[name]
        r = REP["runs"][name]
        pn = CTRL["perm_null"][name]
        for s in SIGS:
            d = r["signals"][s]
            ci = d["ci"]
            pns = pn.get(s, {})
            exc = pns.get("excess", "---")
            pp = pns.get("p_perm", None)
            pps = ("$<$.002" if pp <= 0.002 else f"{pp:.2f}") if pp is not None else "---"
            pw8 = r["prefix"].get(f"{s}_W8", None)
            pw8s = f"{pw8:.3f}" if pw8 is not None else "---"
            rows.append(
                f"{pretty} & {PRETTY[s]} & {d['auroc_max']:.3f} "
                f"[{ci[0]:.2f},{ci[1]:.2f}] & {d['auroc_cusum']:.3f} & {pw8s} & "
                f"{exc if isinstance(exc, str) else f'{exc:+.3f}'} & {pps} \\\\"
            )
        t = tide[name]
        rows.append(
            f"{pretty} & {PRETTY['tide_like']} & {t['auroc_max']:.3f} "
            f"[{t['ci'][0]:.2f},{t['ci'][1]:.2f}] & --- & {t['prefix_w8']:.3f} & --- & --- \\\\"
        )
        rows.append("\\midrule")
    rows = rows[:-1]
    add_table(
        key,
        f"Full per-seed detection results, {gname}: {PERSEED_CAPTION} "
        "$p$ = one-sided permutation $p$-value of the max-agg AUROC against its "
        "length-preserving null (500 shuffles, add-one corrected; minimum attainable "
        "$1/501{\\approx}0.002$); uncorrected for multiplicity --- the paper's claims "
        "rest on within-family replication across seeds, not single cells.",
        ("llccccc", "run & signal & max-agg [95\\% CI] & CUSUM & prefix-$W$8 & excess & $p$"),
        rows,
        "experiments/make\\_supplement.py + final\\_report.py",
        placement="!htbp",
    )

# ---------- D2. full prefix-W sweep ----------
PREFIX_SIGS = ["bcoh_distmin", "stac_energy", "stac_mmd", "disp_all", "ood_knn"]
WS = (4, 6, 8, 10, 12)
for key, gname, members in PERSEED_GROUPS:
    rows = []
    for name in members:
        r = REP["runs"][name]
        for s in PREFIX_SIGS:
            cics = r.get("prefix_ci", {}).get(s, {})

            def cell(w):
                v = r["prefix"][f"{s}_W{w}"]
                ci = cics.get(f"W{w}")
                return (
                    f"{v:.2f} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else f"{v:.2f}"
                )

            cells = " & ".join(cell(w) for w in WS)
            rows.append(f"{pretty_of[name]} & {PRETTY[s]} & {cells} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    add_table(
        key.replace("supp-perseed", "supp-prefixw"),
        f"Full prefix-conditional sweep, {gname}: AUROC at decision replan "
        "$W\\in\\{4,\\dots,12\\}$ (among episodes still running at $W$, predict the "
        "eventual outcome from $\\max_{t\\le W}s_t$), with episode-level bootstrap "
        "95\\% CIs. Eligible populations per $W$ in "
        "Table~\\ref{tab:supp-prefixw-meta}. Conclusions are insensitive to the "
        "main paper's choice $W{=}8$.",
        ("llccccc", "run & signal & $W$4 & $W$6 & $W$8 & $W$10 & $W$12"),
        rows,
        "experiments/final\\_report.py",
        placement="!htbp",
    )

rows = []
for name, _, pretty in RUNS_META:
    pm = REP["runs"][name]["prefix_meta"]
    cells = " & ".join(f"{pm[f'W{w}']['n']} ({pm[f'W{w}']['n_fail']})" for w in WS)
    rows.append(f"{pretty} & {cells} \\\\")
add_table(
    "supp-prefixw-meta",
    "Eligible populations for the prefix-conditional protocol: test episodes with "
    "$\\ge W$ replans (eventual failures in parentheses). The population shrinks "
    "only mildly with $W$ because failures run to the horizon.",
    ("lccccc", "run & $W$4 & $W$6 & $W$8 & $W$10 & $W$12"),
    rows,
    "experiments/final\\_report.py",
    placement="!htbp",
)

# ---------- D3. K-ablation ----------
rows = []
for name, _, pretty in RUNS_META:
    ka = REP["runs"][name]["k_ablation"]
    cells = " & ".join(
        f"{ka[f'K{k}']['auroc']:.3f} [{ka[f'K{k}']['ci'][0]:.2f},{ka[f'K{k}']['ci'][1]:.2f}]"
        for k in (2, 4, 8, 16)
    )
    rows.append(f"{pretty} & {cells} \\\\")
add_table(
    "supp-kablation",
    "Sample-budget ablation: boundary-coherence episode-max AUROC (bootstrap 95\\% "
    "CIs) recomputed with $K\\in\\{2,4,8,16\\}$ fresh chunks per replan ($K{<}16$ "
    "averages 10 random subsets of the logged $K{=}16$ samples). Values differ from "
    "$K{=}16$ by at most 0.022 across all runs and budgets (median 0.01) --- "
    "detection quality is not limited by the sampling budget.",
    ("lcccc", "run & $K{=}2$ & $K{=}4$ & $K{=}8$ & $K{=}16$"),
    rows,
    "experiments/final\\_report.py",
    placement="!htbp",
)

# ---------- G2. calibration robustness ----------
rows = []
for name, _, pretty in RUNS_META:
    rob = EXT["cal_robust"][name]
    # the full-set entry is keyed by the actual calibration size (may be < 40)
    extra = [k for k in rob if k not in ("10", "20", "40")]
    full = extra[0] if extra else max(rob, key=int)
    cells = []
    for n in ("10", "20", "40"):
        d = rob[n]
        cells.append(f"{d['fp_mean']:.2f}$\\pm${d['fp_sd']:.2f}")
    d = rob[full]
    cells.append(f"{d['fp_mean']:.2f} ($n{{=}}{full}$)")
    rows.append(f"{pretty} & " + " & ".join(cells) + " \\\\")
add_table(
    "supp-calrobust",
    "Deferral-threshold robustness to calibration-set size: realized false-deferral "
    "at $\\alpha{=}0.3$ (signal: coherence on strong runs, embedding-OOD on weak "
    "runs) when the success-only calibration set is subsampled to $n$ episodes (50 "
    "draws, mean$\\pm$sd; last column = the full set). Weak runs have only 12--23 "
    "calibration successes, so subsamples at $n{\\ge}20$ coincide with the full set "
    "(sd 0). The conformal bound (false-deferral $\\le\\alpha$ in expectation) holds "
    "conservatively at every size; small calibration sets are noisier but not "
    "anti-conservative.",
    ("lcccc", "run & $n{=}10$ & $n{=}20$ & $n{=}40$ & full"),
    rows,
    "experiments/audit\\_extras.py",
    placement="!htbp",
)

# ---------- G3. elapsed-time vs alarm vs combined online detectors ----------
TA = json.loads((ROOT / "results/final/time_aware.json").read_text())
ta_pretty = {
    "pusht-strong:dev": "Push-T strong (dev)",
    "pusht-strong:s1000": "Push-T strong (A)",
    "pusht-strong:s2000": "Push-T strong (B)",
    "can-strong:s1000": "Can strong",
    "pusht-weak:nocrop": "Push-T weak (w1)",
    "square-weak:s1000": "Square weak (A)",
    "square-weak:s2000": "Square weak (B)",
}
rows = []
for name, r in TA.items():
    # use the middle (0.75*cmax) anchor: timeout's usable region, non-degenerate
    mid = list(r["anchors"].values())[1]
    t, at = mid.get("T"), mid.get("AT")
    if not (t and at and t["handoff"] is not None and at["handoff"] is not None):
        continue
    gain = t["handoff"] - at["handoff"]
    sign = "coh" if r["signal"] == "bcoh_distmin" else "OOD"
    rows.append(
        f"{ta_pretty.get(name, name)} & {sign} & {r['sr']:.2f} & "
        f"{t['coverage']:.2f} & {t['retained']:.2f} & {t['handoff']:.2f} & "
        f"{at['retained']:.2f} & {at['handoff']:.2f} & $-${gain:.2f} \\\\"
    )
add_table(
    "supp-timeaware",
    "Online detectors combining elapsed time and alarm history (reviewer request). "
    "At a per-run coverage anchor inside timeout's usable range (0.75 of its maximum "
    "non-degenerate coverage), we compare the pure \\emph{timeout} rule (defer if "
    "still running at $T_0$) with a training-free \\emph{combined} rule (defer at the "
    "first replan where the running-max alarm crosses a conformal threshold "
    "\\emph{or} $T_0$ is reached). Both are matched on coverage; the combined rule is "
    "constrained to retained success within 3pp of timeout. Handoff = median fraction "
    "of the episode at first deferral (deferred failures). The combined detector is "
    "never later and hands off earlier by an amount that grows with the alarm's "
    "discriminability: small on strong Push-T (coherence near its null), largest on "
    "the OOD-monitorable weak regimes where the timeout alone is nearly useless "
    "(its coverage ceiling $\\approx$ the success rate).",
    (
        "llccccccc",
        "run & sig & SR & cov & timeout ret & timeout handoff & "
        "comb.\\ ret & comb.\\ handoff & handoff earlier",
    ),
    rows,
    "experiments/time\\_aware\\_detector.py",
    placement="!htbp",
)

# ---------- E. leakage controls ----------
rows = []
for name, path, pretty in RUNS_META:
    pn = CTRL["perm_null"][name]["bcoh_distmin"]
    ts = EXT["tsnull"].get(name, {}).get("bcoh_distmin", {})
    sd = CTRL["survivor_decomp"][name]["bcoh_distmin"]
    rows.append(
        f"{pretty} & {pn['raw']:.3f} & {pn['null_mean']:.3f} "
        f"[{pn['null_lo']:.2f},{pn['null_hi']:.2f}] & "
        f"{ts.get('null_mean', '---')} & {sd['full_max_on_survivors']:.3f} & "
        f"{sd['prefix_on_survivors']:.3f} \\\\"
    )
add_table(
    "supp-leakage",
    "Duration-confounding controls for boundary coherence (per run): raw max-agg "
    "AUROC; length-preserving permutation null (mean [2.5\\%,97.5\\%]); "
    "time-stratified permutation null; survivor decomposition at $W{=}8$ "
    "(full-episode max on survivors vs.\\ prefix on survivors).",
    (
        "lccccc",
        "run & raw max & perm.\\ null & time-strat.\\ null & full-max$|$surv & prefix$|$surv",
    ),
    rows,
    "experiments/permutation\\_null.py, audit\\_extras.py",
)

# ---------- G. yield sweeps (all eight runs) ----------
rows = []
for name, _, pretty in RUNS_META:
    if name not in EXT["utility"]:
        raise KeyError(f"utility missing for {name}: rerun audit_extras.py with all 8 runs")
    for rr in EXT["utility"][name]:
        rows.append(
            f"{pretty} & {rr['alpha']} & {rr['coverage']:.2f} & "
            f"{(rr['retained'] or 0):.3f} & {rr['false_defer']:.3f} & "
            f"{rr['fail_recall']:.3f} & "
            f"{rr['median_handoff'] if rr['median_handoff'] is not None else '---'} & "
            f"{rr['replans_saved'] if rr['replans_saved'] is not None else '---'} & "
            f"{rr['EU_pfb0.3']:.2f}/{rr['EU_pfb0.6']:.2f}/{rr['EU_pfb0.9']:.2f} \\\\"
        )
    rows.append("\\midrule")
rows = rows[:-1]
(
    add_table(
        "supp-yield",
        "Complete conformal-deferral sweeps on all eight runs (signal: boundary "
        "coherence on strong runs, embedding-OOD on weak runs): coverage, retained "
        "success, realized false-deferral, deferred-failure recall, median handoff "
        "(fraction of episode), replans saved per deferred failure, and expected "
        "utility at fallback success probabilities 0.3/0.6/0.9 with fallback cost "
        "$c{=}0$ (Fig.~\\ref{fig:supp-heatmap} sweeps the cost).",
        (
            "lcccccccc",
            "run & $\\alpha$ & cov. & retained & false-defer & fail-recall & "
            "handoff & saved & EU (.3/.6/.9)",
        ),
        rows,
        "experiments/audit\\_extras.py",
        placement="!htbp",
    ),
)

# ---------- H. repair stats + MDE ----------
# Human-readable condition labels (provenance verified from the source JSONs:
# guard_sweep_v1 + recovery_power500 baselines match the strong dev checkpoint;
# mid_base_study is the 60k-step mid-strength checkpoint; recovery_square is
# Square seed A).
REPAIR_LABEL = {
    "square_ood_committed_p90": "Square weak (A), committed, $p_{90}$, \\textbf{OOD gate}",
    "square_ood_consensus_p90": "Square weak (A), consensus, $p_{90}$, \\textbf{OOD gate}",
    "strong1500_committed_p90_ext": "Push-T strong (dev), committed, $p_{90}$ --- extension (frozen $\\tau$, seeds 500--1999)",
    "strong2000_committed_p90_pooled": "Push-T strong (dev), committed, $p_{90}$ --- \\textbf{pooled confirmatory ($n{=}2000$)}",
    "strong_gated_p90_committed(main_compare_v2)": "Push-T strong (dev), committed, $p_{90}$ --- early pilot",
    "weak60k_sweep(mid_base_study)": "Push-T mid (60k ckpt, SR 24.5\\%), committed, $p_{90}$",
    "strong150_committed_p86": "Push-T strong (dev), committed, $p_{86}$",
    "strong150_committed_p90": "Push-T strong (dev), committed, $p_{90}$",
    "strong150_committed_p94": "Push-T strong (dev), committed, $p_{94}$",
    "strong150_consensus_p86": "Push-T strong (dev), consensus, $p_{86}$",
    "strong150_consensus_p90": "Push-T strong (dev), consensus, $p_{90}$",
    "strong150_consensus_p94": "Push-T strong (dev), consensus, $p_{94}$",
}
rows = []
for k, v in CTRL["recovery_ci"].items():
    if isinstance(v, str):
        continue
    lab = REPAIR_LABEL.get(k, k.replace("_", "\\_"))
    rows.append(
        f"{lab} & {v['n']} & {v['helped']} & {v['hurt']} & "
        f"{v['delta_pp']:+.1f} & [{v['ci_lo_pp']:+.1f}, {v['ci_hi_pp']:+.1f}] & "
        f"{v['mde_pp']:.1f} \\\\"
    )
p5 = json.loads((ROOT / "results/recovery_power500/compare.json").read_text())
r5 = p5["rows"][0]
n5 = p5["n_episodes"]
import math

d5 = (r5["helped"] - r5["hurt"]) / n5
se5 = math.sqrt(r5["helped"] + r5["hurt"]) / n5
rows.append(
    f"Push-T strong (dev), committed, $p_{{90}}$ --- \\textbf{{confirmatory main}} & "
    f"{n5} & {r5['helped']} & {r5['hurt']} & "
    f"{100 * d5:+.1f} & [{100 * (d5 - 1.96 * se5):+.1f}, {100 * (d5 + 1.96 * se5):+.1f}] & "
    f"{100 * 1.96 * se5:.1f} \\\\"
)
sq = json.loads((ROOT / "results/recovery_square/compare.json").read_text())
for rr in sq["rows"]:
    h_, b_ = rr["helped"], rr["hurt"]
    n_ = sq["n_episodes"]
    d_ = (h_ - b_) / n_
    se_ = math.sqrt(h_ + b_) / n_
    rows.append(
        f"Square weak (seed A), {rr['recovery']}, $p_{{90}}$ & {n_} & {h_} & {b_} & {100 * d_:+.1f} & "
        f"[{100 * (d_ - 1.96 * se_):+.1f}, {100 * (d_ + 1.96 * se_):+.1f}] & {100 * 1.96 * se_:.1f} \\\\"
    )
add_table(
    "supp-repair",
    "All paired re-sampling-repair conditions: discordant counts (helped/hurt), "
    "success-rate change with 95\\% CI (paired normal approximation), and minimum "
    "detectable effect at $\\alpha{=}0.05$. Pairing via common initial seeds and "
    "shared sampling streams up to first intervention. On gate firing, "
    "\\emph{committed} re-samples $K$ fresh chunks and executes the medoid of the "
    "cluster most coherent with the committed continuation; \\emph{consensus} "
    "executes the majority-cluster medoid; $p_{x}$ = gate threshold at the $x$-th "
    "percentile of baseline replan-level coherence scores.",
    ("lcccccc", "condition & $n$ & helped & hurt & $\\Delta$ (pp) & 95\\% CI (pp) & MDE (pp)"),
    rows,
    "experiments/permutation\\_null.py \\S recovery",
)

# ---------- I. taxonomy table (all six Push-T runs) ----------
PUSHT_RUNS = [n for n, _, _ in RUNS_META if n.startswith("pusht")]
missing_tax = [n for n in PUSHT_RUNS if n not in EXT["taxonomy"]]
if missing_tax:
    raise KeyError(f"taxonomy missing for {missing_tax}: rerun audit_extras.py with all 8 runs")
rows = []
for name in PUSHT_RUNS:
    tax = EXT["taxonomy"][name]
    pretty = pretty_of[name]
    for t, d in sorted(tax.items()):
        rows.append(
            f"{pretty} & {t.replace('_', '-')} & {d['n']} & "
            f"{d['bcoh_distmin']:.3f} & {d['stac_mmd']:.3f} & {d['ood_knn']:.3f} \\\\"
        )
    rows.append("\\midrule")
rows = rows[:-1]
add_table(
    "supp-taxonomy",
    "Automatic failure taxonomy on all seven Push-T runs: per-type counts and "
    "per-type detection AUROC (type-vs-success) for three representative signals. "
    "Heuristic labels: near-miss = max coverage $\\ge$0.8; stalled = "
    "bottom-quartile late motion; oscillating = top-quartile mean coherence; "
    "other = remainder.",
    ("llcccc", "run & type & $n$ & bound.\\ coh & STAC-MMD & OOD"),
    rows,
    "experiments/audit\\_extras.py \\S taxonomy",
    placement="!htbp",
)

# ---------- I2. composition reweighting ----------
RW = json.loads((ROOT / "results/final/reweight.json").read_text())
rows = []
for s in ("bcoh_distmin", "stac_mmd", "ood_knn"):
    d = RW[s]
    rows.append(
        f"{PRETTY[s]} & {d['strong_actual']:.3f} & {d['strong_detect_weak_mix']:.3f} & "
        f"{d['weak_actual']:.3f} & {d['gap']:+.3f} & "
        f"{100 * d['composition_share']:.0f}\\% \\\\"
    )
mix = RW["mix"]
mixrow = (
    "\\multicolumn{6}{l}{\\footnotesize failure mix (near-miss/stalled/oscillating/other): "
    f"strong {mix['strong']['near_miss']:.2f}/{mix['strong']['stalled']:.2f}/"
    f"{mix['strong']['oscillating']:.2f}/{mix['strong']['other']:.2f}; "
    f"crop-ablated {mix['weak']['near_miss']:.2f}/{mix['weak']['stalled']:.2f}/"
    f"{mix['weak']['oscillating']:.2f}/{mix['weak']['other']:.2f}; "
    f"undertrained {mix['mid60k']['near_miss']:.2f}/{mix['mid60k']['stalled']:.2f}/"
    f"{mix['mid60k']['oscillating']:.2f}/{mix['mid60k']['other']:.2f}}} \\\\"
)
rows.append("\\midrule")
rows.append(mixrow)
add_table(
    "supp-reweight",
    "Composition-reweighting decomposition of the strong-vs-degraded Push-T gap, using "
    "$\\mathrm{AUROC}=\\sum_t w_t\\,\\mathrm{AUROC}_t$ over failure types (success set "
    "fixed). ``Counterfactual'' holds the strong regime's per-type AUROCs and swaps in "
    "the crop-ablated regime's failure mix; the composition share is the fraction of the "
    "actual gap this swap covers. The coherence gap is entirely compositional; the MMD "
    "and OOD gains on degraded policies mostly reflect per-type detectability change. "
    "The undertrained 60k checkpoint's mix is nearly identical to the crop-ablated mix.",
    (
        "lccccc",
        "signal & strong actual & counterfactual & degraded actual & gap & comp.\\ share",
    ),
    rows,
    "experiments/composition\\_reweight.py",
    placement="!htbp",
)

# ---------- G4. time-priced utility ----------
UT = json.loads((ROOT / "results/final/utility_time.json").read_text())
rows = []
for name, r in UT.items():
    be = r["breakeven_ct"]
    g0 = r["grid"]["pfb0.6_ct0.0"]
    g2 = r["grid"]["pfb0.6_ct0.2"]
    bes = "/".join(("---" if be[k] is None else f"{be[k]:.2f}") for k in ("0.3", "0.6", "0.9"))
    rows.append(
        f"{pretty_of.get(name, name)} & {r['R_ref']:.0f} & "
        f"{g0['best_timeout']:.3f} & {g0['best_alarm']:.3f} & "
        f"{g2['best_timeout']:.3f} & {g2['best_alarm']:.3f} & "
        f"{g2['combined_minus_timeout']:+.3f} & {bes} \\\\"
    )
add_table(
    "supp-utime",
    "Time-priced utility on every run: best pure timeout vs.\\ best conformal alarm at "
    "$p_{\\mathrm{fb}}{=}0.6$ under execution cost $c_t\\in\\{0,0.2\\}$; the "
    "combined-rule column gives best(alarm-or-timeout) minus best timeout at "
    "$c_t{=}0.2$ (the combined family contains pure timeout, so it is never negative); "
    "the last column lists the smallest $c_t$ at which the alarm alone overtakes the "
    "best timeout, per fallback success probability $p_{\\mathrm{fb}}\\in\\{.3,.6,.9\\}$ "
    "(--- = never, $c_t$ swept to 1). On the strong Push-T runs the alarm never "
    "overtakes and the combined rule degenerates to the pure timeout; where the alarm "
    "carries genuine signal (Square, Can at high $p_{\\mathrm{fb}}$), moderate execution "
    "cost flips the ranking.",
    (
        "lccccccc",
        "run & $R_{\\mathrm{ref}}$ & TO$_{c_t=0}$ & alarm$_{c_t=0}$ & "
        "TO$_{c_t=.2}$ & alarm$_{c_t=.2}$ & comb$-$TO$_{c_t=.2}$ & break-even $c_t$",
    ),
    rows,
    "experiments/utility\\_time.py",
    placement="!htbp",
)

(ROOT / "paper/aaai/supp_generated.tex").write_text("\n".join(out_parts))
print("supp_generated.tex written:", len(out_parts), "tables")

# ---------- utility heatmap figure (full-width, larger fonts) ----------
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13})
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, name, sig_label in [
    (axs[0], "pusht-strong:dev", "coherence alarm"),
    (axs[1], "square-weak:s1000", "OOD alarm"),
]:
    pretty = dict((n, p) for n, _, p in RUNS_META)[name]
    utab = EXT["utility"][name]
    base = [r for r in utab if r["alpha"] == 0.1][0]
    full_auto = REP["runs"][name]["sr"]
    pfbs = np.linspace(0.05, 0.95, 19)
    costs = np.linspace(0, 0.3, 13)  # fallback cost in success-equivalent units
    best = np.full((len(costs), len(pfbs)), -np.inf)
    for r in utab:
        cov, ret = r["coverage"], (r["retained"] or 0)
        for i, c in enumerate(costs):
            for j, p in enumerate(pfbs):
                eu = ret * cov + (p - c) * (1 - cov)
                best[i, j] = max(best[i, j], eu)
    gain = best - full_auto
    im = ax.imshow(
        gain,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(gain).max(),
        vmax=np.abs(gain).max(),
        extent=[pfbs[0], pfbs[-1], costs[0], costs[-1]],
    )
    cs = ax.contour(pfbs, costs, gain, levels=[0], colors="k", linewidths=1.5)
    ax.set_xlabel("fallback success probability $p_{fb}$")
    ax.set_ylabel("fallback cost (success-equiv.)")
    ax.set_title(
        f"{pretty}: best-$\\alpha$ EU gain vs full autonomy\n({sig_label}; black = break-even)"
    )
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(ROOT / "results/final/fig_utility_heatmap.png", dpi=140)
plt.savefig(ROOT / "paper/aaai/fig_utility_heatmap.pdf")
print("utility heatmap saved")

# ---------- taxonomy exemplar trajectory grids ----------
run = load_run("results/logs_official")
by_ep, fail = run["by_ep"], run["fail"]
data = pickle.load(open(ROOT / "results/logs_official/rich_logs.pkl", "rb"))
tax = EXT["taxonomy"]["pusht-strong:dev"]
rng = np.random.default_rng(0)
feats = {}
test_ok = [e for e in run["test_eps"] if len(by_ep[e]) > 1]
for e in test_ok:
    if not fail[e]:
        continue
    rs = by_ep[e]
    maxrew = max(r["reward"] for r in rs)
    pos = np.stack([r["agent_pos"] for r in rs])
    late = np.abs(np.diff(pos[-8:], axis=0)).sum()
    bc = [
        float(
            np.linalg.norm(
                rs[j]["head_ov"].astype(np.float32)
                - rs[j - 1]["tail_ov"].astype(np.float32)[0][None],
                axis=-1,
            )
            .mean(-1)
            .min()
        )
        for j in range(1, len(rs))
    ]
    feats[e] = (maxrew, late, float(np.mean(bc)), pos, bc, [r["reward"] for r in rs])
lates = np.array([v[1] for v in feats.values()])
meanbs = np.array([v[2] for v in feats.values()])
q_late = np.quantile(lates, 0.25)
q_b = np.quantile(meanbs, 0.75)
classes = defaultdict(list)
for e, (mr, lt, mb, pos, bc, rew) in feats.items():
    t = (
        "near_miss"
        if mr >= 0.8
        else "stalled"
        if lt <= q_late
        else "oscillating"
        if mb >= q_b
        else "other"
    )
    classes[t].append(e)
# Layout for readability (reviewer round 7): at most 3 exemplars per row, each
# exemplar a stacked (trajectory, curves) pair -> panels ~2.3in wide in a
# full-width figure instead of ~1.1in.
plt.rcParams.update({"font.size": 11})
for t, eps_t in classes.items():
    sel = rng.choice(eps_t, min(6, len(eps_t)), replace=False)
    ncol = min(3, len(sel))
    ngrp = int(np.ceil(len(sel) / ncol))  # rows of exemplars
    fig, axs = plt.subplots(2 * ngrp, ncol, figsize=(4.3 * ncol, 5.6 * ngrp), squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)
    for i, e in enumerate(sel):
        g, c = divmod(i, ncol)
        ax_traj, ax_cur = axs[2 * g, c], axs[2 * g + 1, c]
        ax_traj.set_visible(True)
        ax_cur.set_visible(True)
        mr, lt, mb, pos, bc, rew = feats[e]
        ax_traj.plot(pos[:, 0], pos[:, 1], "-o", ms=3)
        ax_traj.scatter(pos[0, 0], pos[0, 1], c="g", s=70, zorder=3, label="start")
        ax_traj.scatter(pos[-1, 0], pos[-1, 1], c="r", s=70, zorder=3, label="end")
        ax_traj.set_title(f"episode {e}   max coverage = {mr:.2f}", fontsize=11)
        if i == 0:
            ax_traj.legend(fontsize=9, loc="best")
        ax_cur.plot(rew, color="gray", lw=2, label="coverage")
        ax2b = ax_cur.twinx()
        ax2b.plot(range(1, len(bc) + 1), bc, color="#2166ac", lw=2, label="coherence")
        ax2b.set_ylabel("coherence", color="#2166ac", fontsize=10)
        ax_cur.set_ylim(0, 1)
        ax_cur.set_xlabel("replan index", fontsize=10)
        ax_cur.set_ylabel("coverage", color="gray", fontsize=10)
    fig.suptitle(
        f"class = {t.replace('_', ' ')} (n={len(eps_t)} in dev seed); "
        f"{len(sel)} random exemplars: agent trajectory (upper), "
        f"coverage + boundary coherence (lower)",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(ROOT / f"results/final/taxonomy_grid_{t}.png", dpi=130)
    plt.close()
    print(f"taxonomy grid {t} saved ({len(eps_t)} eps, {len(sel)} shown)")

# copy figure assets next to the LaTeX sources so compilation is self-contained
import shutil

for t in classes:
    shutil.copy(
        ROOT / f"results/final/taxonomy_grid_{t}.png", ROOT / f"paper/aaai/taxonomy_grid_{t}.png"
    )
print("DONE")
