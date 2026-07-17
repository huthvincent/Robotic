"""Rich detector-signal collection during plain receding-horizon rollouts.

Runs the BASELINE policy (execute fresh sample 0 every replan) and logs, at each
replan, everything needed to evaluate ALL detector signals offline:

  - dist:      (K,) per-fresh-sample overlap distance to the COMMITTED continuation
               (ours; distmin/distmean for any K' computable offline by subsampling)
  - bcoh_freshmean: scalar, distance of the mean-of-K fresh overlap to committed cont.
  - stac_chamfer / stac_energy: fresh-vs-fresh SET distances between the previous
    replan's K chunks and the current K chunks on their time-aligned overlap
    (Sentinel/STAC-style temporal consistency baseline — no committed plan used)
  - disp_all / disp_first: sample-dispersion baselines
  - emb:       (D,) policy observation embedding (global conditioning), float16,
               for offline kNN-OOD signal (FAIL-Detect-style embedding monitor)
  - agent_pos: (2,) raw agent position (for speed/phase analysis)
  - reward:    last env reward before this replan (ground-truth progress; for
               analysis only, NOT a deployable signal)

Baseline execution is identical to prior experiments: chunk 0 executed, chunk 0
committed. Logging adds no behavioral change.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.utils.constants import OBS_STATE


def _set_chamfer(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two sets of trajectories.
    A: (E,K,m,act), B: (E,K,m,act) -> (E,)"""
    # pairwise (E,K,K): mean-over-time L2 between A_i and B_j
    d = torch.linalg.vector_norm(A.unsqueeze(2) - B.unsqueeze(1), dim=-1).mean(-1)  # (E,K,K)
    return 0.5 * (d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1))


def _set_energy(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Energy distance between two sets. A,B: (E,K,m,act) -> (E,)"""
    dab = torch.linalg.vector_norm(A.unsqueeze(2) - B.unsqueeze(1), dim=-1).mean(-1)  # (E,K,K)
    daa = torch.linalg.vector_norm(A.unsqueeze(2) - A.unsqueeze(1), dim=-1).mean(-1)
    dbb = torch.linalg.vector_norm(B.unsqueeze(2) - B.unsqueeze(1), dim=-1).mean(-1)
    return 2 * dab.mean(dim=(1, 2)) - daa.mean(dim=(1, 2)) - dbb.mean(dim=(1, 2))


def collect_batch(h, seeds, k=16, base_seed=0):
    """Roll out one batch of baseline episodes with rich logging.

    Returns (ep_success: (E,) bool, records: list of per-replan dicts)."""
    device = h.device
    E = len(seeds)
    L, exec_start, H = h.n_action_steps, h.exec_start, h.horizon
    act = h.cfg.action_feature.shape[0]
    m = H - exec_start - L
    raw = h.reset(seeds)
    done = np.zeros(E, bool)
    ep_success = np.zeros(E, bool)
    committed_full = None  # (E,H,act)
    prev_fresh = None  # (E,K,H,act) previous replan's fresh chunks
    exec_buf = np.zeros((E, L, act), np.float32)
    buf_idx = np.full(E, L)
    last_reward = np.zeros(E, np.float32)
    records = []

    def agent_pos(raw_obs):
        for key in ("agent_pos", "observation.state"):
            if isinstance(raw_obs, dict) and key in raw_obs:
                v = np.asarray(raw_obs[key], np.float32)
                return v[:, :2] if v.ndim == 2 else v
        return np.zeros((E, 2), np.float32)

    for t in range(h.max_steps):
        stacked = h.observe(raw)
        if buf_idx.min() >= L:
            gen = torch.Generator(device=device).manual_seed(base_seed * 100003 + t)
            chunks, gcond = h.sample_chunks(stacked, k, generator=gen)  # (E,K,H,act), (E,D)

            rec_common = {}
            exec_all = chunks[:, :, exec_start : exec_start + L, :]
            rec_common["disp_all"] = exec_all.std(dim=1).mean(dim=(-1, -2)).cpu().numpy()
            rec_common["disp_first"] = exec_all[:, :, 0, :].std(dim=1).mean(-1).cpu().numpy()

            # raw overlap blocks (float16) so ALL set statistics are computable offline:
            #   head_ov: this replan's fresh chunks on the incoming overlap region
            #   tail_ov: this replan's fresh chunks on the outgoing overlap region
            # dist/bcoh vs committed plan: committed = sample 0 -> prev tail_ov[0].
            # STAC fresh-vs-fresh: D(prev tail_ov, cur head_ov).
            head_ov = chunks[:, :, exec_start : exec_start + m, :]  # (E,K,m,act)
            tail_ov = chunks[:, :, exec_start + L : H, :]  # (E,K,m,act)
            rec_common["head_ov"] = head_ov.cpu().numpy().astype(np.float16)
            rec_common["tail_ov"] = tail_ov.cpu().numpy().astype(np.float16)
            if committed_full is not None:
                cc = committed_full[:, exec_start + L : H, :].unsqueeze(1)  # (E,1,m,act)
                dist = torch.linalg.vector_norm(head_ov - cc, dim=-1).mean(-1)  # (E,K)
                rec_common["dist"] = dist.cpu().numpy().astype(np.float32)
                fm = head_ov.mean(dim=1)  # (E,m,act)
                rec_common["bcoh_freshmean"] = (
                    torch.linalg.vector_norm(fm - cc.squeeze(1), dim=-1).mean(-1).cpu().numpy()
                )
            if prev_fresh is not None:
                A = prev_fresh[:, :, exec_start + L : H, :]  # (E,K,m,act)
                B = head_ov
                rec_common["stac_chamfer"] = _set_chamfer(A, B).cpu().numpy()
                rec_common["stac_energy"] = _set_energy(A, B).cpu().numpy()

            emb = gcond.detach().cpu().numpy().astype(np.float16)  # (E,D)
            pos = agent_pos(raw)

            exec_np = exec_all.cpu().numpy()
            exec_buf = exec_np[:, 0]
            committed_full = chunks[:, 0]
            prev_fresh = chunks
            buf_idx = np.zeros(E, int)

            for i in range(E):
                if done[i]:
                    continue
                rec = dict(
                    episode=int(seeds[i]),
                    t=int(t),
                    emb=emb[i],
                    agent_pos=pos[i].copy(),
                    reward=float(last_reward[i]),
                )
                for key, val in rec_common.items():
                    rec[key] = val[i].copy() if val.ndim > 1 else float(val[i])
                records.append(rec)

        a_norm = exec_buf[np.arange(E), buf_idx]
        a_real = h.to_real(torch.from_numpy(a_norm).to(device))
        res = h.step(a_real)
        raw = res.obs
        last_reward = np.asarray(res.reward, np.float32)
        succ = h.successes(res.info, E)
        ep_success = ep_success | (succ & ~done)
        done = done | res.terminated | res.truncated
        buf_idx += 1
        if done.all():
            break
    return ep_success, records
