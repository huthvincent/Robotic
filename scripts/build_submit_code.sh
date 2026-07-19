#!/usr/bin/env bash
# Build the ANONYMIZED code-and-data supplement for AAAI submission at
# submit_code/. Differences from the public-release ZIP (make_release.sh):
#   - hard anonymization: machine-specific conda paths in shell scripts are
#     replaced with generic activation lines, and the tree is grep-verified
#     free of usernames/hostnames/absolute paths before packaging
#   - includes the development run's raw rollout logs (results/logs_official,
#     ~13 MB) so reviewers can execute the full analysis pipeline end-to-end
#     without any collection
#   - reviewer-facing README
# Usage: bash scripts/build_submit_code.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DST=submit_code
rm -rf "$DST"
mkdir -p "$DST"

# --- code ---
cp -r guard envs baselines "$DST"/
mkdir -p "$DST/experiments" "$DST/scripts"
cp experiments/*.py "$DST/experiments/"
cp scripts/can_pipeline.sh scripts/night_pipeline.sh "$DST/scripts/" 2>/dev/null || true
cp requirements.freeze.txt "$DST"/

# --- anonymize machine-specific paths in shell scripts ---
python3 - "$DST" <<'PY'
import re
import sys
from pathlib import Path

dst = Path(sys.argv[1])
targets = list((dst / "scripts").glob("*.sh"))
for p in targets:
    s = p.read_text()
    s = re.sub(
        r"source /\S*conda\.sh && conda activate \S+",
        "conda activate robo  # activate the environment created by setup instructions",
        s,
    )
    s = re.sub(r"source /\S*conda\.sh\n *conda activate \S+",
               "conda activate robo", s)
    s = s.replace("/data2/zhu11/robotic/guard-testtime-recovery", ".")
    p.write_text(s)
print("anonymized:", [t.name for t in targets])
PY

# --- generic environment setup (replaces the machine-specific setup_env.sh) ---
cat > "$DST/setup_env.sh" <<'EOF'
#!/usr/bin/env bash
# Create the pinned environment. Version-sensitive pins that matter:
#   pymunk 6.11.1 (7.x removes add_collision_handler, breaking gym-pusht)
#   mujoco 2.3.7  (3.x breaks robosuite 1.4.1)
#   lerobot 0.4.4 (pass --policy.crop_shape="[84,84]" explicitly for Push-T)
set -e
conda create -y -n robo python=3.10
conda activate robo
pip install -r requirements.freeze.txt
EOF

# --- data: analysis JSONs (all reported numbers) + dev-run raw logs ---
mkdir -p "$DST/results/final" "$DST/results/logs_official"
cp results/final/report.json results/final/controls.json results/final/extras.json \
   results/final/tide_like.json results/final/deferral_ci.json \
   results/final/deferral_lift_ci.json results/final/overhead.json \
   results/final/time_aware.json results/final/utility_time.json \
   results/final/reweight.json results/final/excess_ext.json results/final/taxonomy_sensitivity.json \
   "$DST/results/final/"
cp results/logs_official/rich_logs.pkl "$DST/results/logs_official/"
for p in results/guard_sweep_v1/sweep.json results/recovery_power500/compare.json \
         results/recovery_power2000_ext/compare.json results/recovery_square/compare.json \
         results/recovery_square_ood/compare.json results/main_compare_v2/main.json \
         results/mid_base_study/study.json results/bid_faithful/bid.json; do
  mkdir -p "$DST/$(dirname "$p")" && cp "$p" "$DST/$p"
done

# --- reviewer README ---
cat > "$DST/README.md" <<'EOF'
# Code and Data Supplement (anonymized)

Code, analysis artifacts, and one complete raw-log set for the paper
"Knowing When to Yield: An Anatomy of Runtime Failure Alarms for Chunked
Diffusion Policies".

## What is here

```
guard/          signal library: rich-log collection, boundary coherence, STAC
                variants, dispersion, kNN-OOD, re-sampling recovery controllers
envs/           rollout harnesses: LeRobot Push-T and robosuite/robomimic
                (Square, Can), one shared interface
experiments/    every collection, analysis, and table/figure script; each
                docstring states what it produces and how to run it
baselines/      dense-replanning and BID-style selection baselines
scripts/        unattended train+collect pipelines
results/final/  the analysis JSONs from which EVERY number in the paper and
                supplement derives -- auditable without running anything
results/logs_official/rich_logs.pkl
                the development run's complete raw logs (300 episodes,
                per-replan overlap sample blocks + policy embeddings), so the
                full analysis pipeline can be executed end-to-end
results/*/      paired-repair comparison artifacts (compare/sweep/study JSONs)
```

## Auditing the paper's numbers (no execution needed)

Every table maps to a JSON in `results/`: detection tables to
`final/report.json` + `final/controls.json`; learned probes, yield sweeps,
time-stratified nulls, taxonomy to `final/extras.json`; deferral CIs to
`final/deferral_ci.json` / `final/deferral_lift_ci.json`; time-priced utility
to `final/utility_time.json`; elapsed-vs-alarm detectors to
`final/time_aware.json`; composition reweighting to `final/reweight.json`;
enlarged-sample check to `final/excess_ext.json`; repair conditions to the
`recovery_*/` and related JSONs; monitoring overhead to `final/overhead.json`.

## Running the pipeline

Environment: `bash setup_env.sh` (Python 3.10; pins in
requirements.freeze.txt; single GPU; tested with PyTorch 2.10/CUDA 12.8).

End-to-end on the included dev-run logs (no training or collection needed):

```
python experiments/final_report.py --runs pusht-strong:dev:results/logs_official
python experiments/permutation_null.py     # nulls + p-values (edit RUNS to available logs)
python experiments/audit_extras.py         # probes, yield, taxonomy (same note)
```

Full regeneration for any other run is a documented four-command chain:
train (LeRobot, commands below) -> collect (`experiments/collect_logs.py`)
-> analyze (the three scripts above) -> tables
(`experiments/make_supplement.py`).

Training commands (identical recipe per seed; ~0.5-14 GPU-h each):

```
python -m lerobot.scripts.lerobot_train --policy.type=diffusion \
  --policy.crop_shape="[84,84]" --dataset.repo_id=lerobot/pusht \
  --steps=200000 --batch_size=64 --seed=1000 --output_dir=checkpoints/dp_pusht_seed1000
# robomimic Square/Can: --dataset.repo_id=ankile/robomimic-ph-{square,can}-image
#                       --policy.crop_shape="[76,76]"
```

Datasets: `lerobot/pusht`, `ankile/robomimic-ph-square-image`,
`ankile/robomimic-ph-can-image` (all public on the Hugging Face Hub; the
robomimic sets are converted to LeRobot v3.0 with the stock LeRobot converter).

All content is anonymized for double-blind review.
EOF

find "$DST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# --- verify anonymity and size ---
echo "--- anonymity scan (must be empty) ---"
grep -rln "zhu11\|huthvincent\|huthruiz\|/data2\|/home/" "$DST" || echo "(clean)"
echo "--- size ---"
du -sh "$DST"
echo "built $DST/"
