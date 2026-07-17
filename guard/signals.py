"""GUARD trigger signals (computed in the policy's normalized action space).

For the MVP we implement:
  - boundary_coherence: disagreement between a freshly sampled chunk and the
    *committed continuation* of the previous plan on their overlapping horizon.
    This is GUARD's distinctive TEMPORAL signal (mode-jump / seam risk).
  - sample_dispersion: cross-sample disagreement of K fresh chunks at the same
    state (the standard entropy/variance baseline signal, NOT temporal).

All inputs are torch tensors in normalized action space.
"""

from __future__ import annotations

import torch


def exec_window(chunks: torch.Tensor, exec_start: int, n_action_steps: int) -> torch.Tensor:
    """Slice the executable window from a full-horizon chunk.

    chunks: (..., horizon, act) -> (..., n_action_steps, act)
    """
    return chunks[..., exec_start : exec_start + n_action_steps, :]


def overlap_regions(fresh: torch.Tensor, prev: torch.Tensor, exec_start: int, n_action_steps: int):
    """Return the temporally-aligned overlap between a fresh plan and the previous
    plan's committed continuation, assuming the previous executable window
    (n_action_steps long) was fully executed before replanning.

    fresh, prev: (E, horizon, act)
    returns (fresh_overlap, committed_cont): each (E, m, act) with
      m = horizon - exec_start - n_action_steps.
    """
    horizon = fresh.shape[-2]
    m = horizon - exec_start - n_action_steps
    if m <= 0:
        raise ValueError(
            f"no overlap: horizon={horizon} exec_start={exec_start} n_action_steps={n_action_steps}"
        )
    fresh_overlap = fresh[:, exec_start : exec_start + m, :]
    committed_cont = prev[:, exec_start + n_action_steps : horizon, :]
    return fresh_overlap, committed_cont


def boundary_coherence(fresh_overlap: torch.Tensor, committed_cont: torch.Tensor) -> torch.Tensor:
    """Per-env mean L2 distance over the overlap. Higher = larger mode-jump risk.

    fresh_overlap, committed_cont: (E, m, act) -> (E,)
    """
    d = torch.linalg.vector_norm(fresh_overlap - committed_cont, dim=-1)  # (E, m)
    return d.mean(dim=-1)


def sample_dispersion(chunks_exec: torch.Tensor) -> torch.Tensor:
    """Cross-sample disagreement of K chunks at one state (variance baseline).

    chunks_exec: (E, K, L, act) -> (E,)  (mean over L,act of per-coordinate std across K).
    """
    std = chunks_exec.std(dim=1)  # (E, L, act)
    return std.mean(dim=(-1, -2))  # (E,)
