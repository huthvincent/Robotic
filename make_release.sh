#!/usr/bin/env bash
# Build the AAAI supplementary ZIP exactly as described in the supplement's
# submitted-package manifest (supplement.tex, Sec. J): code + environment lock
# + final JSON artifacts + README. Checkpoints (41 GB) and rich logs (217 MB)
# are excluded by design; the manifest documents the one-command regeneration
# path for each.
#
# Usage: bash make_release.sh          -> paper/aaai/supplementary_material.zip
set -euo pipefail
cd "$(dirname "$0")"

OUT=paper/aaai/supplementary_material.zip
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
PKG="$STAGE/guard_supplementary"
mkdir -p "$PKG"

cp README.md setup_env.sh requirements.freeze.txt "$PKG"/
cp -r guard envs "$PKG"/
mkdir -p "$PKG/experiments"
cp experiments/*.py "$PKG/experiments/"
mkdir -p "$PKG/results/final"
cp results/final/report.json results/final/controls.json \
   results/final/extras.json results/final/tide_like.json \
   results/final/deferral_ci.json results/final/deferral_lift_ci.json \
   results/final/overhead.json results/final/time_aware.json \
   results/final/utility_time.json results/final/reweight.json \
   results/final/excess_ext.json \
   "$PKG/results/final/" 2>/dev/null || true
# repair-condition source JSONs referenced by the supplement's repair table
for p in results/guard_sweep_v1/sweep.json results/recovery_power500/compare.json \
         results/recovery_square/compare.json results/main_compare_v2/main.json \
         results/mid_base_study/study.json results/bid_faithful/bid.json; do
  mkdir -p "$PKG/$(dirname "$p")" && cp "$p" "$PKG/$p"
done
cp paper/aaai/supplement.pdf "$PKG/" 2>/dev/null || true
find "$PKG" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

rm -f "$OUT"
(cd "$STAGE" && zip -qr - guard_supplementary) > "$OUT"
echo "built $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | tail -3
