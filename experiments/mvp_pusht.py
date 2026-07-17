"""GUARD minimal-viable experiment on Push-T (single frozen Diffusion Policy).

Validates the two core claims before the full build-out:
  (a) the boundary-coherence trigger separates about-to-fail from benign steps
      BETTER than the standard sample-dispersion (variance) baseline signal
      (risk-coverage / AUROC), and
  (b) trigger-gated A1 (mode-committed re-sample) raises success rate vs plain
      receding-horizon execution while firing on only a small fraction of steps.

Run:
  python experiments/mvp_pusht.py --n_episodes 100 --n_envs 25 --k 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.harness import RolloutHarness  # noqa: E402
from guard.recovery import mode_committed_resample  # noqa: E402
from guard.signals import boundary_coherence, overlap_regions, sample_dispersion  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402

CKPT = str(ROOT / "checkpoints" / "diffusion_pusht")


def run_batch(
    h: RolloutHarness,
    seeds,
    mode,
    threshold=None,
    k=16,
    base_seed=0,
    gate_signal="bcoh_distmean",
    recovery="committed",
):
    """Roll out one batch of episodes (one per env). Returns (ep_success, logs).

    logs: list of per-replan dicts {episode, t, bcoh, disp, intervened}."""
    device = h.device
    E = len(seeds)
    L, exec_start, horizon = h.n_action_steps, h.exec_start, h.horizon
    raw = h.reset(seeds)
    done = np.zeros(E, bool)
    ep_success = np.zeros(E, bool)
    committed_full = None  # (E, horizon, act) torch: previously committed plan
    exec_buf = np.zeros((E, L, h.cfg.action_feature.shape[0]), np.float32)
    buf_idx = np.full(E, L)  # force replan at t=0
    logs = []

    for t in range(h.max_steps):
        stacked = h.observe(raw)
        if buf_idx.min() >= L:  # lockstep replan every L env-steps
            gen = torch.Generator(device=device).manual_seed(base_seed * 100003 + t)
            chunks, _ = h.sample_chunks(stacked, k, generator=gen)  # (E,K,H,act)
            exec_all = chunks[:, :, exec_start : exec_start + L, :]  # (E,K,L,act)
            m = horizon - exec_start - L
            sig = {
                "disp_all": sample_dispersion(exec_all).cpu().numpy(),
                "disp_first": exec_all[:, :, 0, :].std(dim=1).mean(-1).cpu().numpy(),
            }
            if committed_full is not None:
                # per-sample overlap distances to the committed continuation: (E,K)
                fresh_ov = chunks[:, :, exec_start : exec_start + m, :]  # (E,K,m,act)
                cc = committed_full[:, exec_start + L : horizon, :].unsqueeze(1)  # (E,1,m,act)
                dist = torch.linalg.vector_norm(fresh_ov - cc, dim=-1).mean(-1)  # (E,K)
                fo2, cc2 = overlap_regions(chunks.mean(dim=1), committed_full, exec_start, L)
                sig["bcoh_freshmean"] = boundary_coherence(fo2, cc2).cpu().numpy()
                sig["bcoh_distmean"] = dist.mean(dim=1).cpu().numpy()
                sig["bcoh_distmin"] = dist.min(dim=1).values.cpu().numpy()
            else:
                for key in ("bcoh_freshmean", "bcoh_distmean", "bcoh_distmin"):
                    sig[key] = np.zeros(E, np.float32)

            gate = sig[gate_signal]
            exec_np = exec_all.cpu().numpy()
            chosen_idx = np.zeros(E, int)
            intervened = np.zeros(E, bool)
            for i in range(E):
                if mode == "guard" and committed_full is not None and gate[i] >= threshold:
                    committed_overlap = committed_full[i, exec_start + L : horizon, :].cpu().numpy()
                    idx = mode_committed_resample_idx(
                        exec_np[i], committed_overlap, seed=i, recovery=recovery
                    )
                    chosen_idx[i] = idx
                    intervened[i] = True
            exec_buf = exec_np[np.arange(E), chosen_idx]  # (E,L,act)
            committed_full = chunks[torch.arange(E), torch.tensor(chosen_idx, device=device)]
            buf_idx = np.zeros(E, int)

            for i in range(E):
                if not done[i]:
                    rec = dict(episode=int(seeds[i]), t=int(t), intervened=bool(intervened[i]))
                    for key, val in sig.items():
                        rec[key] = float(val[i])
                    logs.append(rec)

        a_norm = exec_buf[np.arange(E), buf_idx]  # (E,act)
        a_real = h.to_real(torch.from_numpy(a_norm).to(device))
        res = h.step(a_real)
        raw = res.obs
        succ = h.successes(res.info, E)
        ep_success = ep_success | (succ & ~done)
        done = done | res.terminated | res.truncated
        buf_idx += 1
        if done.all():
            break
    return ep_success, logs


def mode_committed_resample_idx(
    chunks_exec, committed_overlap, n_clusters=3, seed=0, recovery="committed"
):
    """Return the chosen K-index for a recovery.

    recovery="committed": cluster medoid most COHERENT with the committed
        continuation (temporal consistency; GUARD's A1).
    recovery="consensus": medoid of the LARGEST cluster (majority mode;
        ignores history) — an ablation of what drives any recovery gain.
    """
    from guard.recovery import _kmeans

    k, L, act = chunks_exec.shape
    flat = chunks_exec.reshape(k, L * act)
    labels, _ = _kmeans(flat, n_clusters, seed=seed)
    clusters, sizes = {}, {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = flat[idx]
        centroid = pts.mean(0, keepdims=True)
        clusters[c] = idx[np.argmin(((pts - centroid) ** 2).sum(1))]
        sizes[c] = len(idx)
    if recovery == "consensus" or committed_overlap is None:
        return int(clusters[max(sizes, key=sizes.get)])
    m = committed_overlap.shape[0]
    best_c, best_d = None, np.inf
    for c, medoid in clusters.items():
        d = np.linalg.norm(chunks_exec[medoid, :m, :] - committed_overlap, axis=-1).mean()
        if d < best_d:
            best_d, best_c = d, c
    return int(clusters[best_c])


# ---------- metrics ----------
def auroc(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def risk_coverage(scores, fail_labels, n_points=21):
    """Selective-risk curve: reject highest-score steps first; report failure rate
    among retained (kept) steps as coverage decreases. Returns AURC (lower=better)."""
    scores = np.asarray(scores)
    fail = np.asarray(fail_labels).astype(float)
    order = np.argsort(scores)  # ascending: keep low-score (safe) first
    fail_sorted = fail[order]
    covs, risks = [], []
    for frac in np.linspace(1.0, 0.1, n_points):
        keep = max(1, int(len(scores) * frac))
        covs.append(frac)
        risks.append(fail_sorted[:keep].mean())
    return float(np.trapz(risks[::-1], covs[::-1])), covs, risks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--percentiles", type=float, nargs="+", default=[70.0, 80.0, 90.0])
    ap.add_argument("--gate", type=str, default="bcoh_distmean")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "mvp_pusht"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    env_cfg = PushtEnv()
    h = RolloutHarness(env_cfg, CKPT, n_envs=args.n_envs)

    n_batches = (args.n_episodes + args.n_envs - 1) // args.n_envs
    # ---- BASELINE (log signals) ----
    print(f"[baseline] {n_batches} batches x {args.n_envs} envs")
    t0 = time.time()
    base_succ, base_logs = [], []
    for b in range(n_batches):
        seeds = list(range(b * args.n_envs, (b + 1) * args.n_envs))
        s, lg = run_batch(h, seeds, "baseline", k=args.k, base_seed=b + 1)
        base_succ.extend(s.tolist())
        base_logs.extend(lg)
        print(f"  batch {b}: succ={np.mean(s) * 100:.1f}%  ({time.time() - t0:.0f}s)")
    base_sr = float(np.mean(base_succ))
    fail_by_ep = {ep: (not ok) for ep, ok in enumerate(base_succ)}

    SIGNALS = ["bcoh_freshmean", "bcoh_distmean", "bcoh_distmin", "disp_all", "disp_first"]
    BCOH = [s for s in SIGNALS if s.startswith("bcoh")]

    # a replan record is "valid" once a committed plan exists (all bcoh signals nonzero)
    def is_valid(r):
        return any(r[s] > 0 for s in BCOH)

    recs = [r for r in base_logs if is_valid(r)]
    step_fail = np.array([fail_by_ep[r["episode"]] for r in recs], bool)

    # per-episode aggregates (max & mean over the episode's valid replans)
    eps = sorted({r["episode"] for r in recs})
    ep_fail = np.array([fail_by_ep[ep] for ep in eps], bool)
    sq = {}
    for s in SIGNALS:
        step_vals = np.array([r[s] for r in recs])
        ep_max = np.array([max(r[s] for r in recs if r["episode"] == ep) for ep in eps])
        ep_mean = np.array([np.mean([r[s] for r in recs if r["episode"] == ep]) for ep in eps])
        aurc, _, _ = risk_coverage(step_vals, step_fail)
        sq[s] = {
            "step_auroc": auroc(step_vals, step_fail),
            "step_aurc": aurc,
            "ep_auroc_max": auroc(ep_max, ep_fail),
            "ep_auroc_mean": auroc(ep_mean, ep_fail),
        }

    results = {
        "baseline_success_rate": base_sr,
        "n_episodes": len(base_succ),
        "n_replan_records": len(recs),
        "n_failed_episodes": int(ep_fail.sum()),
        "signal_quality": sq,
        "gate_signal": args.gate,
    }

    # ---- GUARD at thresholds derived from the chosen gate signal ----
    gate_step_vals = np.array([r[args.gate] for r in recs])
    results["guard"] = {}
    for pct in args.percentiles:
        thr = float(np.percentile(gate_step_vals, pct))
        g_succ, g_logs = [], []
        for b in range(n_batches):
            seeds = list(range(b * args.n_envs, (b + 1) * args.n_envs))
            s, lg = run_batch(
                h, seeds, "guard", threshold=thr, k=args.k, base_seed=b + 1, gate_signal=args.gate
            )
            g_succ.extend(s.tolist())
            g_logs.extend(lg)
        interv_rate = float(np.mean([r["intervened"] for r in g_logs]))
        results["guard"][f"p{pct}"] = {
            "threshold": thr,
            "success_rate": float(np.mean(g_succ)),
            "intervention_rate": interv_rate,
            "delta_vs_baseline": float(np.mean(g_succ) - base_sr),
        }
        print(
            f"  guard[{args.gate}] p{pct}: succ={np.mean(g_succ) * 100:.1f}%  "
            f"interv={interv_rate * 100:.1f}%  delta={(np.mean(g_succ) - base_sr) * 100:+.1f}pp"
        )

    import pickle

    with open(out / "baseline_logs.pkl", "wb") as f:
        pickle.dump({"logs": base_logs, "success": base_succ}, f)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("\n==== RESULTS ====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
