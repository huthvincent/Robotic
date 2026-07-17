"""Closed-loop (replan-every-step) test-time baselines on the shared harness:
  - baseline_cl : receding horizon with replan every step, execute sample 0
  - temporal_ensemble : ACT-style exponentially-weighted average of overlapping
    predictions for the current timestep
  - bid : Bidirectional Decoding — best-of-N with backward coherence + forward
    contrast against a weak (low-NFE) reference set

These replan every step, so they cost ~n_action_steps x more diffusion calls than
the receding-horizon baseline; that compute is measured for the success-vs-compute
Pareto against GUARD (which intervenes sparsely).
"""

from __future__ import annotations

import numpy as np
import torch

from baselines.bid import bid_select


def closed_loop_rollout(
    h, seeds, mode="baseline_cl", k=8, k_weak=4, weak_steps=4, te_decay=0.1, base_seed=0
):
    """Return (ep_success, n_diffusion_calls_per_env_mean)."""
    device = h.device
    E = len(seeds)
    exec_start, Hh = h.exec_start, h.horizon
    act = h.cfg.action_feature.shape[0]
    raw = h.reset(seeds)
    done = np.zeros(E, bool)
    ep_success = np.zeros(E, bool)
    prev_plan = None  # (E, Hh, act) last selected full-horizon plan
    te_buffer = []  # list of (age, (E,Hh,act)) for temporal ensembling
    calls = np.zeros(E)

    for t in range(h.max_steps):
        stacked = h.observe(raw)
        gen = torch.Generator(device=device).manual_seed(base_seed * 100003 + t)
        if mode == "bid":
            strong, _ = h.sample_chunks(stacked, k, generator=gen)  # (E,k,Hh,act)
            gw = torch.Generator(device=device).manual_seed(base_seed * 100003 + t + 7)
            weak, _ = h.sample_chunks(stacked, k_weak, generator=gw, num_inference_steps=weak_steps)
            calls += k + k_weak
            strong_np = strong.cpu().numpy()
            weak_np = weak.cpu().numpy()
            chosen = np.zeros((E, Hh, act), np.float32)
            for i in range(E):
                cands = strong_np[i, :, exec_start:, :]  # (k, Hh-exec_start, act)
                # forward contrast anchored to consensus means (avoids self-reference:
                # candidates must be near the STRONG consensus and far from the WEAK one)
                pos = strong_np[i, :, exec_start:, :].mean(0, keepdims=True)
                neg = weak_np[i, :, exec_start:, :].mean(0, keepdims=True)
                if prev_plan is not None:
                    prev_ov = prev_plan[i, exec_start + 1 :, :].cpu().numpy()  # shift by 1 step
                else:
                    prev_ov = None
                idx = bid_select(cands, prev_ov, pos, neg)
                chosen[i] = strong_np[i, idx]
            chosen_t = torch.from_numpy(chosen).to(device)
            a_norm = chosen_t[:, exec_start, :]
            prev_plan = chosen_t
        elif mode == "temporal_ensemble":
            chunks, _ = h.sample_chunks(stacked, 1, generator=gen)  # (E,1,Hh,act)
            calls += 1
            plan = chunks[:, 0]  # (E,Hh,act)
            te_buffer.append(plan)
            te_buffer[:] = te_buffer[-Hh:]  # keep last Hh plans
            # action for current step = weighted avg of plan[age] predicting this step
            num = torch.zeros((E, act), device=device)
            den = 0.0
            for age, p in enumerate(reversed(te_buffer)):  # age 0 = newest
                j = exec_start + age
                if j < Hh:
                    w = float(np.exp(-te_decay * age))
                    num += w * p[:, j, :]
                    den += w
            a_norm = num / den
            prev_plan = plan
        else:  # baseline_cl
            chunks, _ = h.sample_chunks(stacked, 1, generator=gen)
            calls += 1
            a_norm = chunks[:, 0, exec_start, :]
            prev_plan = chunks[:, 0]

        a_real = h.to_real(a_norm)
        res = h.step(a_real)
        raw = res.obs
        succ = h.successes(res.info, E)
        ep_success = ep_success | (succ & ~done)
        done = done | res.terminated | res.truncated
        if done.all():
            break
    return ep_success, float(calls.mean())
