#!/usr/bin/env bash
# Reviewer-8 M1: second-task strong policy. Download + convert Can-ph dataset,
# train DP (identical recipe to Square: crop 76x76, 200k steps, batch 64),
# collect 300-episode rich logs. Self-sequencing; run under nohup.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data2/zhu11/miniconda3/etc/profile.d/conda.sh
conda activate robo
LOG=scratch/can_pipeline.log
echo "=== can pipeline start $(date) ===" >> $LOG

DS=/home/zhu11/.cache/huggingface/lerobot/ankile/robomimic-ph-can-image

# 1) download (mirror fallback, retries inside)
if [ ! -f "$DS/meta/info.json" ] && [ ! -f "$DS/meta/info.json" ]; then
  python scratch/download_can.py > scratch/download_can.log 2>&1 \
    || { echo "DOWNLOAD FAILED $(date)" >> $LOG; exit 1; }
fi
echo "download ok $(date)" >> $LOG

# 2) convert v2.1 -> v3.0 in place (converter manages _old/_v30 swap)
if ! grep -q '"codebase_version": "v3.0"' "$DS/meta/info.json" 2>/dev/null; then
  HF_ENDPOINT=https://hf-mirror.com python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id=ankile/robomimic-ph-can-image > scratch/convert_can.log 2>&1
  grep -q '"codebase_version": "v3.0"' "$DS/meta/info.json" 2>/dev/null \
    || { echo "CONVERSION FAILED $(date)" >> $LOG; exit 1; }
fi
echo "conversion ok $(date)" >> $LOG

# 3) train (same recipe as Square seeds)
if [ ! -d checkpoints/dp_can_seed1000/checkpoints/200000 ]; then
  echo "can training start $(date)" >> $LOG
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion --policy.push_to_hub=false \
    --policy.crop_shape="[76,76]" \
    --dataset.repo_id=ankile/robomimic-ph-can-image \
    --steps=200000 --batch_size=64 --seed=1000 \
    --save_freq=100000 --eval_freq=0 --log_freq=20000 \
    --policy.device=cuda --wandb.enable=false \
    --output_dir=checkpoints/dp_can_seed1000 \
    > scratch/train_can_s1000.log 2>&1
  echo "can training done $(date)" >> $LOG
fi
[ -d checkpoints/dp_can_seed1000/checkpoints/200000 ] \
  || { echo "TRAINING FAILED $(date)" >> $LOG; exit 1; }

# 4) collect 300-episode rich logs
if [ ! -f results/logs_can_s1000/rich_logs.pkl ]; then
  mkdir -p results/logs_can_s1000
  python -u experiments/collect_logs.py --benchmark can \
    --ckpt checkpoints/dp_can_seed1000/checkpoints/200000/pretrained_model \
    --n_episodes 300 --n_envs 10 --k 16 --out results/logs_can_s1000 \
    > scratch/collect_can_s1000.log 2>&1
fi
[ -f results/logs_can_s1000/rich_logs.pkl ] \
  || { echo "COLLECTION FAILED $(date)" >> $LOG; exit 1; }

echo "=== can pipeline ALL DONE $(date) ===" >> $LOG
touch scratch/CAN_PIPELINE_DONE
