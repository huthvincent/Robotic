"""Bidirectional Decoding (BID) — the strongest training-free sample-selection
competitor for chunked policies (Liu et al., 2024). Reimplemented harness-agnostic.

BID selects, among N freshly sampled candidate chunks, the one that is both
  - backward-coherent: close to the previously predicted plan on the overlap
    (temporal consistency), and
  - forward-contrastive: close to a strong (positive) reference set and far from
    a weak (negative) reference set on the action horizon.

We approximate the weak policy with a low-NFE sampler (fewer denoising steps),
which the caller supplies as `neg_refs`. All arrays are numpy in normalized
action space. Returns the selected candidate index.

NOTE: this is our reimplementation for fair in-harness comparison; hyperparameters
(N, weak-NFE, overlap length) are matched to GUARD's compute where relevant.
"""

from __future__ import annotations

import numpy as np


def _min_dist(cands: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """For each candidate, min L2 (mean over time) distance to any reference.

    cands: (N, L, act), refs: (M, L, act) -> (N,)
    """
    if refs.shape[0] == 0:
        return np.zeros(cands.shape[0])
    # (N, M) mean-over-(L,act) L2
    d = np.linalg.norm(cands[:, None] - refs[None], axis=-1).mean(-1)  # (N, M)
    return d.min(1)


def bid_select(
    cands: np.ndarray,  # (N, L, act) candidate executable windows
    prev_overlap: np.ndarray | None,  # (m, act) previous committed continuation, or None
    pos_refs: np.ndarray,  # (Ns, L, act) strong reference samples
    neg_refs: np.ndarray,  # (Nw, L, act) weak reference samples
    w_backward: float = 1.0,
    w_forward: float = 1.0,
) -> int:
    n = cands.shape[0]
    # backward coherence: distance to previous plan on the overlap (lower = better)
    if prev_overlap is not None:
        m = prev_overlap.shape[0]
        backward = np.linalg.norm(cands[:, :m, :] - prev_overlap[None], axis=-1).mean(-1)  # (N,)
    else:
        backward = np.zeros(n)
    # forward contrast: near strong refs, far from weak refs (lower is better)
    d_pos = _min_dist(cands, pos_refs)
    d_neg = _min_dist(cands, neg_refs)
    forward = d_pos - d_neg
    score = w_backward * backward + w_forward * forward
    return int(np.argmin(score))
