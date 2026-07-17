# GUARD — Progress Tracker

Legend: [ ] todo · [~] in progress · [x] done · [!] blocked/risk

## Phase 0 — Setup
- [x] Recon server (H200 143GB, CUDA 13.2, internet OK, uv+conda)
- [x] Topic selection (multi-agent survey) → GUARD chosen
- [x] Scaffold repo + DESIGN.md + README.md
- [x] Install LeRobot 0.4.4 + gym-pusht (pymunk pinned 6.11.1; migrated old ckpt to 0.4.x format)
- [x] Load frozen DP checkpoint on Push-T; baseline eval = 73.3% success (30 eps), 1.8 s/ep
- [x] Validate custom chunk-level rollout path (K-sample batched diffusion, embedding, unnormalize, step)

## Phase 1 — Minimal-viable result (Push-T, 1 frozen DP)  ✅ GO
- [x] Policy wrapper: sample K action chunks; expose obs-embedding; vary denoising steps
- [x] Trigger term (i): backward-coherence (executing chunk vs fresh sample on overlap)
- [x] Baseline signal: action-dispersion / sample variance (for risk-coverage comparison)
- [x] Recovery A1: mode-committed re-sample (K=16, k-means, medoid)
- [x] Eval harness: per-step failure labels, risk-coverage curve, success at low intervention rate
- [x] **MVP VERDICT (150 eps): GO.** Baseline 64% (matches published). Boundary-coherence
      predicts failure MUCH better than dispersion: bcoh_distmin ep-AUROC 0.741 [0.66,0.82]
      vs disp 0.598 [0.51,0.69] (mean-agg CIs disjoint). Gated A1 at 9% intervention = +6.7pp
      (64→70.7%); over-triggering hurts → motivates better gate (bcoh_distmin), abstain, calibration.

## Phase 1b — Strengthen closing-the-loop
- [x] Gate on bcoh_distmin; recovery variants. Best = committed re-sample, +6.0pp @9% interv (p=0.25),
      but CHURNS (helps~29/hurts~20). hold/extend CASCADE (disagreement trigger re-fires vs a recovery
      that fights the policy) → informative negative result.
- [x] Paired McNemar harness (deterministic); recovery_compare + guard_sweep done.
- [x] **Calibration (conformal) holds**: threshold from successful rollouts → realized FP tracks target
      (0.10→0.121); temporal trigger separates fail/success (1.4x) while dispersion collapses (~1.0x @α=0.2).

## Phase 1c — Main comparison (done, 75 eps)
- [x] receding baseline 65.3% (27.7 calls/ep) is a STRONG default. GUARD 68.0% (+2.7pp, helped15/hurt13,
      p=0.85 = wash on strong base). Closed-loop COLLAPSES: baseline_cl 34.7%, temporal_ensemble 41.3%,
      BID 21.3% — replan-every-step w/ independent samples = constant mode-jumps (dramatically CONFIRMS
      the mode-jump premise). BID was buggy (forward-contrast self-reference → picks outliers); fixed to
      consensus-mean anchors, needs re-run.
- [x] **Key takeaway:** on a well-trained base, recovery has little headroom (failures aren't mostly
      re-sampleable mode-jumps). Trigger/calibration result stands regardless.

## Phase 1d — Recovery in the multimodal regime → ROBUST NEGATIVE
- [x] Trained 15k (10% SR) and 60k (~25% SR) Push-T DP (DP needs way more steps than budgeted for
      mid-competence). On the weak base too, gated re-sample recovery is a WASH (helped≈hurt, all p>0.1).
- [x] **Conclusion: gated re-sampling recovery does NOT reliably close the loop on Push-T (strong OR weak).**

## Phase 2 — Reframe: detector + selective autonomy (POSITIVE) ✅
- [x] **Selective autonomy WORKS**: defer high-trigger episodes → retained success 64%→76.2% @70% cov
      (+12.2pp), vs dispersion +2.7pp, random flat. This is the paper's positive contribution.
- [x] 4-panel publication figure (trigger AUROC / risk-coverage / calibration / selective autonomy).
- [x] Paper reframed (paper/DRAFT.md): calibrated temporal detector + selective autonomy + honest
      recovery-negative. Title: "Knowing When to Yield".

## Phase 3b — Solidify sprint (2026-07-05 evening, in progress)
- [x] Official Push-T train config recovered (200k steps / batch 64 / lr 1e-4) — explains why 60k→25%
- [~] Multi-seed Push-T bases: seeds 1000+2000 training (200k each, nohup-safe); official ckpt = seed 3
- [x] Sentinel/STAC + FAIL-Detect facts verified from papers → positioning rewritten (DESIGN.md 6c,
      paper/DRAFT.md). Key: success-only quantile calibration is THEIRS; our novelty = execution-anchored
      signal + what-to-do-with-alarm study + negative result.
- [x] collect.py v2: logs raw overlap blocks (head_ov/tail_ov f16) + embeddings + per-sample distances
      → ALL signals computable offline (faithful STAC energy/MMD/chamfer + CUSUM, OOD-kNN, fusion, K-ablation)
- [x] detector_analysis.py v2: AUROC+CI / TPR@FPR / AURC / conformal defer curves (finite-sample corrected)
      / earliness / cal-set-size stability / null_length (length-leakage control)
- [~] 300-ep rich collection on official ckpt (running)
- [x] robomimic bridge DONE & smoke-tested: robosuite 1.4.1 + mujoco 2.3.7 (pymunk-style version pin),
      ankile/robomimic-ph-square-image converted to v3.0, state=eef_pos+quat+gripper verified,
      FLIP images verified (corr 0.987 flipped vs -0.108 raw), RobomimicHarness duck-types RolloutHarness
- [~] Square DP training seed1000 (200k, crop 76×76, nohup)

## Phase 3 — Remaining for a full AAAI submission (roadmap)
- [ ] Second benchmark (robomimic via robosuite, policy-agnostic controller) to clear ≥2-benchmark bar
- [ ] OOD + contact trigger terms + fusion ablation; multi-seed base checkpoints; fixed-BID Pareto
- [ ] Convert paper/DRAFT.md → AAAI LaTeX; full ablation tables
- Note: strong-base recovery being a wash is a ROBUST finding — do not force a positive; the honest
  detector+selective-autonomy framing is the defensible paper.

## Phase 2 — Full method
- [ ] OOD term (ii) RND on successful-rollout embeddings
- [ ] Contact/phase gate (iii)
- [ ] 3-param logistic fusion + conformal calibration
- [ ] Recovery A2 (extra deliberation), A3 (abstain/hold)
- [ ] Rules-based gating policy + learned-threshold variant

## Phase 3 — Baselines
- [ ] open-loop / closed-loop fixed-H
- [ ] Temporal Ensembling
- [ ] BID (reimplement, match at compute)
- [ ] RTC (via LeRobot)
- [ ] FIPER detect-only

## Phase 4 — Benchmarks & scale-out
- [ ] Push-T full (3 seeds × 50 inits)
- [ ] Robomimic Lift/Can/Square/Transport (ph, +mh slice)
- [ ] LIBERO Spatial/Object/Goal/Long
- [ ] Second base: flow / rectified-flow policy (cross-family transfer)

## Phase 5 — Analysis & paper
- [ ] All ablations
- [ ] Calibration (ECE, risk-coverage, AUROC vs entropy)
- [ ] Success-vs-compute Pareto
- [ ] Tables + figures
- [ ] LaTeX write-up + honest limitations

---
### Log
- 2026-07-05: Project created. Topic = GUARD. Env `robo` (py3.10, torch2.5.1+cu124) ready; installing LeRobot.
