"""GUARD recovery actions (training-free, frozen base).

MVP implements A1 = mode-committed re-sample:
  draw K chunks, cluster them (k-means on the flattened executable window), take
  each cluster's medoid, and execute the medoid of the cluster whose overlap
  region is most coherent with the already-committed continuation. This commits
  to a SINGLE action mode at the seam rather than averaging across modes.
"""

from __future__ import annotations

import numpy as np


def _kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0):
    """Minimal k-means. x: (N, D) -> labels (N,), centers (k, D). Deterministic init."""
    n = x.shape[0]
    k = min(k, n)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (N, k)
        new = d.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            labels = new
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = x[m].mean(0)
    return labels, centers


def mode_committed_resample(
    chunks_exec: np.ndarray,  # (K, L, act) executable windows for ONE env
    committed_overlap: np.ndarray | None,  # (m, act) previous committed continuation, or None
    n_clusters: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Return the chosen executable window (L, act) for ONE env."""
    k, L, act = chunks_exec.shape
    flat = chunks_exec.reshape(k, L * act)
    labels, _ = _kmeans(flat, n_clusters, seed=seed)

    # medoid per cluster = sample closest to its cluster mean
    clusters = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = flat[idx]
        centroid = pts.mean(0, keepdims=True)
        medoid = idx[np.argmin(((pts - centroid) ** 2).sum(1))]
        clusters[c] = medoid

    if committed_overlap is None:
        # no history: pick medoid of the largest cluster
        biggest = max(np.unique(labels), key=lambda c: (labels == c).sum())
        return chunks_exec[clusters[biggest]]

    # pick the cluster whose medoid overlap best matches the committed continuation
    m = committed_overlap.shape[0]
    best_c, best_d = None, np.inf
    for c, medoid in clusters.items():
        cand_overlap = chunks_exec[medoid, :m, :]  # (m, act)
        d = np.linalg.norm(cand_overlap - committed_overlap, axis=-1).mean()
        if d < best_d:
            best_d, best_c = d, c
    return chunks_exec[clusters[best_c]]
