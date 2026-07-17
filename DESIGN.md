# GUARD: Gated Uncertainty-Aware Recovery at Deployment

> A training-free, inference-time layer that turns *inert failure detection* into
> *deployment-time success* for frozen chunked generative visuomotor policies
> (Diffusion Policy, flow/consistency policies, action-chunking transformers).

Target venue: **AAAI-2027** (full-paper deadline ~mid-Aug 2026). Sim-only, single H200.

---

## 1. Problem

Modern generative visuomotor policies fail at deployment mainly through
**compounding error** and **chunk-boundary mode-jumps**. A rich 2025–2026 literature
can now *detect* these failures at runtime (FIPER, ActProbe, VLA-FAIL, VLAConf), but
detection is **inert**: almost no method *closes the loop* by triggering a training-free
recovery and honestly reporting the resulting success-rate change. Worse, the reused
detection signals (action entropy, denoising variance, attention confidence) are weakly
calibrated (failure AUROC ≈ 0.71) and ignore compounding error and contact-rich phases,
so naïve "intervene when uncertain" both fires at the wrong times and gives no principled
choice of *what* to do.

**The open problem:** a *calibrated*, *phase/contact-aware* per-step trigger that decides,
over a **frozen** policy, whether to (A0) commit-and-continue, (A1) re-sample the chunk
committed to a single action mode, (A2) spend extra deliberation compute, or (A3)
abstain/hold — with gains shown in **success rate AND calibration**, not detection AUROC alone.

## 2. Method (all inference-only; base policy stays frozen)

### 2.1 Calibrated phase-aware trigger `s_t ∈ [0,1]`
Fuse three per-step terms:
- **(i) Backward-coherence / boundary risk** — disagreement between the currently-executing
  chunk and a freshly sampled chunk on their overlapping horizon (mode-jump / seam risk).
- **(ii) OOD term** — RND-style distance in the policy's own observation-embedding space to a
  small buffer of embeddings from **successful** calibration rollouts (no failure data needed).
- **(iii) Contact/phase gate** — a cheap scalar (predicted end-effector speed valley or
  gripper-state transition) that up-weights the trigger in contact-rich phases where
  compounding error bites.

Fuse via a 3-parameter logistic; set the operating threshold by **conformal calibration** on
~20–50 successful calibration rollouts to guarantee a target intervention-rate / risk-coverage
tradeoff. Crucially the signal is **validated as calibrated** (ECE, risk-coverage), not merely used.

### 2.2 Recovery menu (training-free, frozen base)
- **A0 Commit-and-continue** (default; most steps).
- **A1 Mode-committed re-sample** — draw K chunks, cluster them (k-means on the first H action
  steps), pick the cluster whose mode is most coherent with the already-committed prefix,
  execute its medoid. Explicitly commits to **one** mode at the seam (distinct from BID's
  backward+forward-contrast per-step scoring and from RTC inpainting).
- **A2 Extra deliberation** — increase denoising / integration substeps for this control step only.
- **A3 Abstain / hold** — freeze at last safe pose (short receding-horizon re-plan) rather than
  execute an OOD action.

### 2.3 Gating policy
Mostly rules-based map from `(s_t, which term dominates)` → menu:
high boundary term → A1; high OOD term → A2 then A3 if still high; contact-gated → prefer A1/A3.
Ablation: a tiny learned-threshold variant (logistic regression on calibration rollouts,
<1k params, **no** policy retraining).

**Thesis:** a properly *calibrated*, *phase-aware* trigger lets a *small* menu of training-free
recoveries convert detection into success, and the abstain option makes the system honest about
when *not* to act.

## 3. Why novel (vs the saturated watchouts)
- **NOT** entropy/variance → chunk-length (AAC/PACE/DVAC/AutoHorizon/HiPolicy/SCALE): GUARD does
  not schedule chunk length; it selects among heterogeneous recovery actions incl. abstain.
- **NOT** best-of-N / verifier-free selection (BID, MG-Select, VOTE, TAS): GUARD is *gated*
  (most steps do nothing) and mode-*committed* via clustering + abstain; no weak reference policy,
  no external verifier.
- **NOT** boundary smoothing (RTC / temporal ensemble): GUARD commits to a single discrete mode at
  the seam instead of averaging/inpainting, and only when triggered.
- **NOT** a plain detector (FIPER/ActProbe/VLA-FAIL): GUARD *acts* and reports success-rate deltas.
- **NOT** trained-verifier scaling (RoboMonkey) or VLM/activation steering (VLS/COAST/CAG).

Two things reviewers can least attack: the trigger is **validated as calibrated**, and the first
controlled **composition/interaction** study of trigger-gated recovery vs uniform intervention.

## 4. Experimental design

### Bases (frozen)
- **Diffusion Policy** (CNN + Transformer) — primary base.
- **A flow / rectified-flow policy** — second heterogeneous base (cross-family transfer test).

### Benchmarks
- **Push-T** (gym-pusht / lerobot ref checkpoint) — canonical multimodality + mode-jump probe.
- **Robomimic** Lift/Can/Square/Transport (ph; + mh slice for shift).
- **LIBERO** Spatial/Object/Goal/Long (Long = compounding-error stress; 3rd benchmark > the ≥2 bar).

### Baselines
open-loop; closed-loop fixed receding-horizon; Temporal Ensembling; **BID**; **RTC**;
**FIPER (detect-only, same trigger budget, no recovery)**; GUARD-uniform-intervention.

### Metrics
- Success rate (Robomimic/LIBERO) / max target-coverage ≥95% (Push-T); mean±std over **≥3 seeds
  × 50 env inits**, avg of last 10 checkpoints (published DP protocol).
- Δ-success vs each baseline with CIs ("Beyond Binary Success" statistically-rigorous protocol).
- **Trigger calibration**: ECE + risk-coverage / selective-risk curves for `s_t` predicting per-step
  failure; failure AUROC vs entropy/variance (must beat ≈0.71).
- Intervention rate + per-action usage (sparsity).
- Compute: extra NFE / wall-clock per episode; success-vs-compute Pareto vs BID(N=64) and RTC.
- Abstain quality: success on retained episodes vs abstain rate (selective prediction).

### Ablations
remove each trigger term; rules vs learned gating; menu ablation (A1-only / A2-only / A1+A3 / full);
conformal calibration on/off; gated vs uniform at matched budget; composition study; cross-family
transfer; sensitivity to K and calibration-set size; OOD/shift stress (Robomimic-mh, LIBERO-Long).

## 5. Minimal-viable result (validate before full build-out)
One frozen DP checkpoint on Push-T. Implement **only** the boundary-coherence trigger term and
**only** A1 (mode-committed re-sample, K=16), single fixed threshold. Over 200–500 episodes show:
(a) the trigger's per-step score separates about-to-fail from benign steps — **risk-coverage curve
better than action-entropy** on the same rollouts; and (b) triggered A1 **raises coverage-success vs
plain closed-loop** while firing on only a small fraction of steps. Positive on both → proceed.

## 6. Compute budget
Base training ~60 runs × 0.5–2 H200-GPU-h ≈ 30–90 GPU-h (days, overlappable). Inference eval is
CPU-sim-bound, parallel over 192 threads; GUARD adds only K forward passes on *triggered* steps.
Full grid is single-digit GPU-days over ~5 weeks with re-run headroom. Peak VRAM tiny (DP/flow ≪ 143 GB).

## 6b. Benchmark plan (revised after tooling recon, 2026-07-05)
Harness is LeRobot-native. LeRobot ships a Push-T DP checkpoint and envs for
pusht/aloha/libero/metaworld but **no robomimic env**. Decision:
- **Primary (deep): Push-T** — full method, all baselines, all ablations, multi-seed
  base checkpoints. MVP already validates the core here.
- **Second: robomimic (Lift/Can/Square[/Transport])** via a *policy-agnostic* GUARD
  controller: train DP on lerobot-format robomimic datasets (e.g. `ankile/robomimic-ph-*-image`),
  evaluate through robosuite. Canonical DP suite → directly comparable to BID/RTC/DP papers.
- **Stretch third: LIBERO-Long** (LeRobot env supported) as the compounding-error stress test.

Empirical finding to date (Push-T, 150 eps, frozen DP, baseline 64% = published number):
boundary-coherence predicts episode failure at AUROC **0.74 [0.66,0.82]** vs dispersion/variance
**0.60 [0.51,0.69]** (mean-aggregate CIs disjoint) — the calibrated-trigger claim holds.

## 6c. Positioning vs Sentinel/STAC (verified from paper, 2026-07-05)
Sentinel (Agia et al., CoRL 2024, arXiv:2410.04640) is the closest prior work and MUST be
compared faithfully. Facts: STAC compares the marginal distributions of temporally overlapping
action samples at consecutive inference steps (fresh-vs-fresh), with forward/reverse KL or MMD,
scores a rollout by the CUMULATIVE SUM of distances, and calibrates the alarm threshold as the
(1-δ)-quantile over M successful rollouts (M=50 sim) with an FPR≤δ guarantee (their Prop. 1).
They evaluate on PushT + mobile-manipulation tasks, report TPR/TNR/accuracy/detection-time, and
do NOT act on the alarm (no recovery, no retained-success analysis).
**Implications for our claims:**
- "Conformal-style calibration on success-only rollouts" is NOT novel alone — Sentinel does this.
- Our novelty rests on: (1) the COMMITTED-PLAN-anchored statistic (execution-anchored
  boundary-coherence, esp. distance-to-nearest-fresh-mode) vs their fresh-vs-fresh distribution
  distance — must WIN head-to-head on the same data (we implement STAC faithfully: energy/MMD/
  chamfer + cusum aggregation); (2) the first systematic "what do you do with the alarm" study:
  quantified selective autonomy (retained-success vs coverage) + the recovery negative result;
  (3) multi-seed, multi-benchmark, length-leakage-controlled protocol (null_length baseline).
- Cite Sentinel prominently; frame as building on their calibration paradigm.

## 6d. PIVOT after dev-seed signal lab (2026-07-05 ~21:00) — FINAL PAPER SHAPE
Dev-seed (official ckpt, 300 eps) results across 16 signal variants:
- Episode-max AUROC: ALL temporal-coherence signals tie at 0.67-0.70 (ewma_stac_energy 0.700,
  stac_energy_k2 0.691, bcoh_distmean 0.674, bcoh_distmin 0.668 — CIs overlap). New variants
  (plan_nll KDE, ratio, znorm, early) did NOT help. Dispersion 0.60, OOD-kNN 0.51 (useless),
  fusion no help. K=2 saturates for BOTH families.
- null_length achieves AUROC 1.0 (success ends early on Push-T) → cusum-style scores (~0.92)
  are length-inflated; a methodological exposé worth reporting.
- Prefix-W conditional AUROC (no length leak): EVERYTHING collapses to 0.55-0.64 → these
  signals DETECT failure-in-progress (earliness ~0.5), they do NOT PREDICT failure early.
- Prefix-8 deferral adds ~nothing; full-episode running-max deferral (mid-episode handoff)
  remains the real value: 64→76% retained @0.7 coverage.
**Paper reframed as a rigorous anatomy/study** ("What do runtime failure alarms buy you?"):
(C1) protocol fixes: length-leakage exposé + prefix-W conditional evaluation + conformal
     deferral with retained-success curves; (C2) family-saturation finding (anchored bcoh ≈
     faithful STAC; K=2 suffices; OOD/dispersion below); (C3) detection≠prediction, quantified;
(C4) the asymmetry: alarms → calibrated mid-episode handoff works; alarms → re-sampling
     recovery does not (+ dense replanning collapses). All × 3 Push-T seeds × robomimic Square.
DISCIPLINE: official seed = dev (variants explored); seeds 1000/2000 + Square = confirmation,
run once with the frozen recipe. Open question for Square: failure structure differs (grasp/
insert misses) — does early predictability change? Either answer is a finding.

## 6e. Training-recipe gotcha + weak-policy regime finding (2026-07-06 00:30)
**Root cause of weak self-trained Push-T policies found:** lerobot 0.4.4 default
`policy.crop_shape=None` (no random crop), while the official diffusion_pusht was trained with
`crop_shape=[84,84]`. Without random crop: 200k steps → 25.3% SR (also explains the earlier
60k→25%). Seeds retrained with `--policy.crop_shape="[84,84]"`; sanity auto-eval at the 100k
checkpoint armed. Weak-policy artifacts KEPT (checkpoints/dp_pusht_seed1000_nocrop,
results/logs_pusht_weak_nocrop) because they revealed a potential finding:
**signal ranking is policy-quality dependent** — on the 25% base, stac_mmd jumps to 0.851 and
embedding-OOD to 0.711 (useless 0.51 on the strong base) while dispersion INVERTS to 0.371;
prefix-W stays weak (~0.55). Interpretation: weak-policy failures ARE distribution-shift-like
(OOD works), strong-policy failures are in-distribution indecision (only coherence works).
If it replicates, this regime-dependence becomes a paper subsection.

## 6f. Regime finding REPLICATED + honesty-checked (07-06 ~04:00) — co-headline
Square seed1000 (trained w/ correct crop-76 recipe) still only ~25% SR → likely lerobot-DP-
without-EMA reproduction gap (config verified: both cameras, crop76, std horizons; controller
matches robomimic env_args exactly; action stats match dataset). DECISION: do not chase a
strong square policy; square serves as the weak-regime cross-task replication. Pipeline still
trains square seed2000 (2nd weak seed). Findings after honesty check (prefix-W on OOD too):
- Weak-regime signal flip REPLICATES cross-task: square ood_knn 0.946 max / **0.832 prefix-W8**
  (real EARLY prediction, first in the study!); pusht-weak ood 0.711/0.65; dispersion INVERTS
  on both weak policies (0.30-0.37); coherence family dies (~0.5-0.6).
- Refined law: **failure predictability tracks failure type** — incompetence failures (weak
  policies) are OOD-visible EARLY; indecision failures (strong policies) are only detectable
  in-progress by coherence, and nothing predicts them early. Explains contradictory conclusions
  across the detection literature (each paper's regime differs).
- Paper claims to update: "detection≠prediction" scoped to strong/in-distribution regime;
  regime-dependence promoted to co-headline; ood added to deferral sweep (weak regimes).

## 7. Honest scope
Simulation / offline only. Framed as an ML decision-theoretic method evaluated on robot benchmarks.
No hardware claims. Report where GUARD abstains or fails.
