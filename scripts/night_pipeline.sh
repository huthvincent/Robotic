#!/usr/bin/env bash
# Overnight pipeline: wait for FINAL (200k) checkpoints, collect detector logs per
# seed/benchmark, queue the second square seed. Idempotent; safe to re-run.
set -u
cd /data2/zhu11/robotic/guard-testtime-recovery
source /data2/zhu11/miniconda3/etc/profile.d/conda.sh
conda activate robo
export MUJOCO_GL=egl
LOG=scratch/night_pipeline.log
echo "=== night pipeline start $(date) ===" >> $LOG

wait_for_dir () {  # poll every 5 min
  while [ ! -d "$1" ]; do sleep 300; done
}

collect_if_missing () {  # <ckpt> <benchmark> <out> <n_envs>
  if [ ! -f "$3/rich_logs.pkl" ]; then
    echo "collect $3 start $(date)" >> $LOG
    python -u experiments/collect_logs.py --ckpt "$1" --benchmark "$2" \
      --n_episodes 300 --n_envs "$4" --k 16 --out "$3" > "$3.collect.log" 2>&1
    echo "collect $3 done $(date): $(tail -1 $3.collect.log)" >> $LOG
  fi
}

# 1) Push-T seeds (FINAL checkpoints = step 200000)
wait_for_dir checkpoints/dp_pusht_seed1000/checkpoints/200000
mkdir -p results/logs_seed1000
collect_if_missing checkpoints/dp_pusht_seed1000/checkpoints/200000/pretrained_model \
  pusht results/logs_seed1000 25

wait_for_dir checkpoints/dp_pusht_seed2000/checkpoints/200000
mkdir -p results/logs_seed2000
collect_if_missing checkpoints/dp_pusht_seed2000/checkpoints/200000/pretrained_model \
  pusht results/logs_seed2000 25

# 2) Square seed 1000
wait_for_dir checkpoints/dp_square_seed1000/checkpoints/200000
mkdir -p results/logs_square_s1000
collect_if_missing checkpoints/dp_square_seed1000/checkpoints/200000/pretrained_model \
  square results/logs_square_s1000 10

# 3) Queue square seed 2000 training then collect
if [ ! -d checkpoints/dp_square_seed2000/checkpoints/200000 ]; then
  echo "square seed2000 training start $(date)" >> $LOG
  python -u -m lerobot.scripts.lerobot_train \
    --policy.type=diffusion --policy.push_to_hub=false \
    --policy.crop_shape="[76,76]" \
    --dataset.repo_id=ankile/robomimic-ph-square-image \
    --steps=200000 --batch_size=64 --seed=2000 \
    --save_freq=100000 --eval_freq=0 --log_freq=20000 \
    --policy.device=cuda --wandb.enable=false \
    --output_dir=checkpoints/dp_square_seed2000 \
    > scratch/train_square_s2000.log 2>&1
  echo "square seed2000 training done $(date)" >> $LOG
fi
mkdir -p results/logs_square_s2000
collect_if_missing checkpoints/dp_square_seed2000/checkpoints/200000/pretrained_model \
  square results/logs_square_s2000 10

echo "=== night pipeline ALL DONE $(date) ===" >> $LOG
touch scratch/NIGHT_PIPELINE_DONE
