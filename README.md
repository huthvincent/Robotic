# Knowing When to Yield: An Anatomy of Runtime Failure Alarms for Chunked Diffusion Policies

Research code for a controlled **audit of training-free runtime failure alarms** for chunked
generative visuomotor policies (Diffusion Policy). This is not a new-detector project: it asks
what existing alarm families actually measure, how they should be evaluated, and which
downstream decisions they support.

> 中文读者：[overview.md](overview.md) 是由浅入深的完整项目故事；[docs/project_summary.md](docs/project_summary.md) 是逐轮迭代的详细档案。

## Key findings

1. **Duration leakage.** On benchmarks where success terminates the episode, episode length
   alone is a perfect "detector" (AUROC 1.0), and rollout-level aggregations (CUSUM, episode-max)
   inherit much of that. We measure *monitorability* as AUROC excess over a length-preserving
   permutation null, and evaluate decisions with a leak-free **prefix-conditional protocol**
   (the landmarking construction from survival analysis).
2. **Monitorability depends on the failure regime, not policy competence alone.** Strong Push-T
   failures (in-distribution near-misses) carry little signal beyond duration for every tested
   statistic, learned probes included; a strong Can policy exposes genuine rollout-level
   coherence signal; degraded-policy regimes expose signal through different families again
   (distributional STAC-MMD within-task, embedding-OOD on Square).
3. **Alarms support yielding, not naive repair.** A conformally calibrated alarm acts as a
   better-timed timeout for selective handoff (deferring at ~53% of the episode instead of
   94–97%), while alarm-triggered re-sampling repair shows no effect detectable above a 2.7pp
   MDE in 2,000 paired episodes.

## Repository layout

```
guard/          Signal library: rich-log collection, boundary coherence, STAC variants,
                dispersion, kNN-OOD, recovery controllers (committed / consensus re-sampling)
envs/           Rollout harnesses: LeRobot Push-T and robosuite/robomimic (Square, Can),
                duck-typed to one interface
experiments/    Every collection, training-support, analysis, and table/figure script
                (each file's docstring states what it produces and how to run it)
baselines/      Dense-replanning baselines (naive closed-loop, BID-style selection)
scripts/        Unattended pipelines (overnight train+collect, Can-ph end-to-end)
results/        Analysis artifacts (small JSONs are versioned; raw rollout logs are not — see
                "Data" below)
docs/           Project documentation (Chinese)
DESIGN.md       Original design document (historical; the project pivoted from recovery to audit)
PROGRESS.md     Task timeline (historical)
```

## Installation

Tested on Ubuntu 24.04, Python 3.10, a single NVIDIA GPU (any ≥24 GB card works; we used one
H200). Version pins that matter are captured in `requirements.freeze.txt`; the load-bearing ones:

| pin | why |
|---|---|
| `lerobot==0.4.4` | flat namespace; pass `--policy.crop_shape="[84,84]"` explicitly for Push-T — the 0.4.4 default (no crop) silently trains a much weaker policy |
| `pymunk==6.11.1` | 7.x removes `add_collision_handler`, breaking gym-pusht |
| `mujoco==2.3.7` | 3.x breaks robosuite 1.4.1 |
| `robosuite==1.4.1` | robomimic-compatible envs |

```bash
bash setup_env.sh          # creates the conda env and installs pinned dependencies
```

## Reproduction pipeline

Four stages, each a single command; every table in the analysis derives from the JSON artifacts
the last two stages write.

```bash
# 1) Train policies (LeRobot; commands for each checkpoint in scripts/ and the docstrings)
python -m lerobot.scripts.lerobot_train --policy.type=diffusion \
    --policy.crop_shape="[84,84]" --dataset.repo_id=lerobot/pusht \
    --steps=200000 --batch_size=64 --seed=1000 --output_dir=checkpoints/dp_pusht_seed1000

# 2) Collect 300-episode rich logs per policy (K=16 fresh chunks per replan, raw overlap
#    blocks + policy embeddings logged, so every signal is recomputable offline)
python experiments/collect_logs.py --benchmark pusht \
    --ckpt checkpoints/dp_pusht_seed1000/checkpoints/200000/pretrained_model \
    --n_episodes 300 --out results/logs_seed1000

# 3) Detection analysis: per-signal AUROCs, prefix-W sweep, K-ablation, permutation nulls
python experiments/final_report.py --runs pusht-strong:s1000:results/logs_seed1000 ...
python experiments/permutation_null.py
python experiments/audit_extras.py          # learned probes, yield sweeps, taxonomy, ts-null
python experiments/deferral_ci.py           # bootstrap CIs for deferral operating points
python experiments/time_aware_detector.py   # elapsed-time vs alarm vs combined detectors

# 4) Paired repair experiments (baseline and gated arms share seeds and sampling streams)
python experiments/recovery_compare.py --n_episodes 500 --pct 90 --recoveries committed
```

Analysis order matters: `final_report.py` → `permutation_null.py` → `audit_extras.py`
(each reads its predecessors' outputs). **Note:** `final_report.py` rewrites
`results/final/report.json` wholesale — always pass the full run list.

## Data

Raw rollout logs (`rich_logs.pkl`, ~15–50 MB per run) and policy checkpoints (~1–6 GB each)
are not versioned. Both are regenerated by stages 1–2 above; the versioned
`results/**/*.json` artifacts are sufficient to audit every reported number without re-running
anything.

## Hardware and cost

Single-GPU throughout. Policy training is 0.5–14 GPU-hours per checkpoint; a 300-episode
collection is ~0.5–1 GPU-hour. Monitoring overhead at deployment is +3.5% wall-clock at K=2
samples per replan (measured; see `experiments/bench_overhead.py`).

## License and citation

Code is released for research use. A citation entry will be added once the accompanying
manuscript is public.
