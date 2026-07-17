#!/usr/bin/env bash
# Reproducible environment setup for GUARD (single-GPU, sim-only).
# Verified 2026-07-05 on H200 NVL, CUDA 13.2 driver, Ubuntu.
set -euo pipefail

source /data2/zhu11/miniconda3/etc/profile.d/conda.sh
conda create -y -n robo python=3.10
conda activate robo

# LeRobot pulls a recent torch (cu128) that works on Hopper.
pip install "lerobot==0.4.4" "gym-pusht==0.1.6"

# gym-pusht needs the pymunk 6.x collision-handler API (pymunk 7.x removed it).
pip install "pymunk>=6.4,<7.0"

# Frozen Push-T Diffusion Policy: the public checkpoint predates the 0.4.x
# processor format, so migrate it once into checkpoints/diffusion_pusht/.
python -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/diffusion_pusht \
  --output-dir checkpoints/diffusion_pusht

echo "Sanity eval (expect ~65-75% success):"
python -m lerobot.scripts.lerobot_eval \
  --policy.path=checkpoints/diffusion_pusht --env.type=pusht \
  --eval.n_episodes=30 --eval.batch_size=15 --policy.device=cuda \
  --output_dir=results/eval_check
