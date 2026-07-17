"""Unified GUARD rollout controller over a frozen chunked policy.

One receding-horizon loop (replan every n_action_steps) that supports the
baseline and several test-time recovery actions, so every method is evaluated
on the SAME harness with paired seeds. Determinism is enforced for reproducible
paired comparisons.

Recoveries (fire only when the calibrated trigger exceeds the threshold):
  committed  : re-sample K, execute cluster medoid most coherent w/ committed continuation (A1)
  consensus  : re-sample K, execute majority-cluster medoid (ablation of A1)
  extend     : DON'T jump — execute the committed continuation of the previous plan (low-variance A1')
  hold       : abstain — freeze the end-effector at its current pose (A3)
"""

from __future__ import annotations

import numpy as np
import torch

from guard.recovery import _kmeans
from guard.signals import boundary_coherence, overlap_regions, sample_dispersion
from lerobot.utils.constants import OBS_STATE


def _coherent_or_consensus_idx(exec_i, committed_overlap, recovery, seed):
    k, L, act = exec_i.shape
    flat = exec_i.reshape(k, L * act)
    labels, _ = _kmeans(flat, min(3, k), seed=seed)
    clusters, sizes = {}, {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = flat[idx]
        clusters[c] = idx[np.argmin(((pts - pts.mean(0, keepdims=True)) ** 2).sum(1))]
        sizes[c] = len(idx)
    if recovery == "consensus" or committed_overlap is None:
        return int(clusters[max(sizes, key=sizes.get)])
    m = committed_overlap.shape[0]
    best_c, best_d = None, np.inf
    for c, med in clusters.items():
        d = np.linalg.norm(exec_i[med, :m, :] - committed_overlap, axis=-1).mean()
        if d < best_d:
            best_d, best_c = d, c
    return int(clusters[best_c])


def rollout_batch(
    h,
    seeds,
    mode="baseline",
    recovery="committed",
    gate_signal="bcoh_distmin",
    threshold=None,
    k=16,
    base_seed=0,
    ood_bank=None,
):
    """ood_bank: (N, D) float32 array of policy conditioning embeddings from
    successful calibration rollouts. Required when gate_signal == "ood_knn"
    (the informative signal on Square); the gate is then the mean distance to
    the 5 nearest bank embeddings, matching the offline detector."""
    device = h.device
    bank_t = None
    if ood_bank is not None:
        bank_t = torch.as_tensor(np.asarray(ood_bank, np.float32), device=device)
    E = len(seeds)
    L, exec_start, H = h.n_action_steps, h.exec_start, h.horizon
    act = h.cfg.action_feature.shape[0]
    m = H - exec_start - L
    raw = h.reset(seeds)
    done = np.zeros(E, bool)
    ep_success = np.zeros(E, bool)
    committed_full = None  # (E, H, act) torch
    exec_buf = np.zeros((E, L, act), np.float32)
    raw_hold = np.zeros((E, L, act), np.float32)  # raw-space hold actions
    hold_mask = np.zeros((E, L), bool)
    buf_idx = np.full(E, L)
    logs = []

    def raw_agent_pos(raw_obs):
        # Push-T raw obs: agent position lives in the "agent_pos" field.
        for key in ("agent_pos", "observation.state", "pixels_agent_pos"):
            if isinstance(raw_obs, dict) and key in raw_obs:
                v = np.asarray(raw_obs[key], np.float32)
                return v[:, :act] if v.ndim == 2 else v
        return None

    for t in range(h.max_steps):
        stacked = h.observe(raw)
        if buf_idx.min() >= L:
            gen = torch.Generator(device=device).manual_seed(base_seed * 100003 + t)
            chunks, gcond = h.sample_chunks(stacked, k, generator=gen)  # (E,K,H,act), (E,D)
            exec_all = chunks[:, :, exec_start : exec_start + L, :]
            sig = {
                "disp_all": sample_dispersion(exec_all).cpu().numpy(),
                "disp_first": exec_all[:, :, 0, :].std(dim=1).mean(-1).cpu().numpy(),
            }
            if committed_full is not None:
                fresh_ov = chunks[:, :, exec_start : exec_start + m, :]
                cc = committed_full[:, exec_start + L : H, :].unsqueeze(1)
                dist = torch.linalg.vector_norm(fresh_ov - cc, dim=-1).mean(-1)  # (E,K)
                fo2, cc2 = overlap_regions(chunks.mean(1), committed_full, exec_start, L)
                sig["bcoh_freshmean"] = boundary_coherence(fo2, cc2).cpu().numpy()
                sig["bcoh_distmean"] = dist.mean(1).cpu().numpy()
                sig["bcoh_distmin"] = dist.min(1).values.cpu().numpy()
            else:
                for key in ("bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"):
                    sig[key] = np.zeros(E, np.float32)
            if bank_t is not None:
                # same conditioning vector collect.py logs as `emb`, so the online
                # gate and the offline kNN-OOD detector see identical embeddings
                d = torch.cdist(gcond.float(), bank_t)  # (E, N)
                sig["ood_knn"] = d.topk(5, dim=1, largest=False).values.mean(1).cpu().numpy()

            gate = sig[gate_signal]
            exec_np = exec_all.cpu().numpy()
            full_np = chunks.cpu().numpy()
            pos = raw_agent_pos(raw)
            new_committed = full_np[:, 0].copy()
            hold_mask[:] = False
            intervened = np.zeros(E, bool)
            for i in range(E):
                exec_buf[i] = exec_np[i, 0]
                fire = mode == "guard" and committed_full is not None and gate[i] >= threshold
                if not fire:
                    continue
                intervened[i] = True
                comm_ov = committed_full[i, exec_start + L : H, :].cpu().numpy()
                if recovery in ("committed", "consensus"):
                    idx = _coherent_or_consensus_idx(exec_np[i], comm_ov, recovery, seed=i)
                    exec_buf[i] = exec_np[i, idx]
                    new_committed[i] = full_np[i, idx]
                elif recovery == "extend":
                    cont = committed_full[i, exec_start + L : H, :].cpu().numpy()  # (m,act)
                    pad = np.repeat(cont[-1:], L - m, axis=0) if L > m else np.zeros((0, act))
                    win = np.concatenate([cont, pad], 0)[:L]
                    exec_buf[i] = win
                    new_committed[i] = np.concatenate([win, np.repeat(win[-1:], H - L, 0)], 0)
                elif recovery == "hold" and pos is not None:
                    raw_hold[i] = np.repeat(pos[i : i + 1], L, axis=0)
                    hold_mask[i] = True
                    new_committed[i] = full_np[i, 0]
            committed_full = torch.from_numpy(new_committed).to(device)
            buf_idx = np.zeros(E, int)
            for i in range(E):
                if not done[i]:
                    rec = dict(episode=int(seeds[i]), t=int(t), intervened=bool(intervened[i]))
                    rec.update({key: float(val[i]) for key, val in sig.items()})
                    logs.append(rec)

        a_norm = exec_buf[np.arange(E), buf_idx]
        a_real = h.to_real(torch.from_numpy(a_norm).to(device))
        held = hold_mask[np.arange(E), buf_idx]
        if held.any():
            pos_now = raw_agent_pos(raw)
            if pos_now is not None:
                a_real = np.where(held[:, None], pos_now[:, :act], a_real)
        res = h.step(a_real)
        raw = res.obs
        succ = h.successes(res.info, E)
        ep_success = ep_success | (succ & ~done)
        done = done | res.terminated | res.truncated
        buf_idx += 1
        if done.all():
            break
    return ep_success, logs
