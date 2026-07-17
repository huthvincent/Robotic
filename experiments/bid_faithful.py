"""Faithful(er) BID positive control (Liu et al., ICLR 2025).

Setting follows the paper's core recipe on Push-T:
  - closed-loop execution (replan every step, execute first action)
  - N strong candidates from the CURRENT policy (full NFE)
  - weak negative references from an EARLIER CHECKPOINT of the SAME training run
    (we use seed-1000 @100k as weak vs @200k as strong)
  - selection = backward coherence (distance to previously selected plan on the
    overlap) + forward contrast (avg distance to strong references minus avg
    distance to weak references), lower is better.
Differences from the released implementation are listed in the paper appendix.

Run: python experiments/bid_faithful.py --n_episodes 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.harness import RolloutHarness  # noqa: E402
from lerobot.envs.configs import PushtEnv  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402

STRONG = str(ROOT / "checkpoints/dp_pusht_seed1000/checkpoints/200000/pretrained_model")
WEAK = str(ROOT / "checkpoints/dp_pusht_seed1000/checkpoints/100000/pretrained_model")


@torch.no_grad()
def sample_with(policy, stacked, k, horizon, act_dim, device, generator=None):
    gcond = policy.diffusion._prepare_global_conditioning(stacked)
    gc = gcond.repeat_interleave(k, dim=0)
    noise = None
    if generator is not None:
        noise = torch.randn(
            (gcond.shape[0] * k, horizon, act_dim),
            device=device,
            dtype=gcond.dtype,
            generator=generator,
        )
    ch = policy.diffusion.conditional_sample(gcond.shape[0] * k, global_cond=gc, noise=noise)
    return ch.view(gcond.shape[0], k, horizon, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=150)
    ap.add_argument("--n_envs", type=int, default=25)
    ap.add_argument("--n_strong", type=int, default=8)
    ap.add_argument("--n_weak", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(ROOT / "results/bid_faithful"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    h = RolloutHarness(PushtEnv(), STRONG, n_envs=args.n_envs)
    weak = DiffusionPolicy.from_pretrained(WEAK).to(h.device).eval()
    E = args.n_envs
    es, H = h.exec_start, h.horizon
    act = h.cfg.action_feature.shape[0]
    nb = (args.n_episodes + E - 1) // E

    def run(mode):
        succ_all = []
        for b in range(nb):
            seeds = list(range(b * E, (b + 1) * E))
            raw = h.reset(seeds)
            done = np.zeros(E, bool)
            ep_s = np.zeros(E, bool)
            prev = None
            for t in range(h.max_steps):
                stacked = h.observe(raw)
                gen = torch.Generator(device=h.device).manual_seed((b + 1) * 100003 + t)
                strong = sample_with(h.policy, stacked, args.n_strong, H, act, h.device, gen)
                if mode == "bid":
                    gw = torch.Generator(device=h.device).manual_seed((b + 1) * 100003 + t + 7)
                    weak_s = sample_with(weak, stacked, args.n_weak, H, act, h.device, gw)
                    s_np = strong.cpu().numpy()
                    w_np = weak_s.cpu().numpy()
                    picks = np.zeros(E, int)
                    for i in range(E):
                        cands = s_np[i, :, es:, :]  # (N,H-es,a)
                        # backward: distance to previously selected plan, shifted 1 step
                        if prev is not None:
                            pv = prev[i, es + 1 :, :]
                            back = np.linalg.norm(
                                cands[:, : pv.shape[0], :] - pv[None], axis=-1
                            ).mean(-1)
                        else:
                            back = np.zeros(len(cands))
                        # forward contrast: avg dist to strong-others minus avg dist to weak
                        d_ss = np.linalg.norm(cands[:, None] - cands[None], axis=-1).mean(
                            -1
                        )  # (N,N)
                        np.fill_diagonal(d_ss, np.nan)
                        d_pos = np.nanmean(d_ss, axis=1)
                        wv = w_np[i, :, es:, :]
                        d_neg = np.linalg.norm(cands[:, None] - wv[None], axis=-1).mean(-1).mean(1)
                        score = back + (d_pos - d_neg)
                        picks[i] = int(np.argmin(score))
                    chosen = strong[torch.arange(E), torch.tensor(picks, device=h.device)]
                else:  # closed-loop baseline: first sample
                    chosen = strong[:, 0]
                a = h.to_real(chosen[:, es, :])
                prev = chosen.cpu().numpy()
                res = h.step(a)
                raw = res.obs
                s = h.successes(res.info, E)
                ep_s |= s & ~done
                done |= res.terminated | res.truncated
                if done.all():
                    break
            succ_all.extend(ep_s.tolist())
            print(f"  {mode} batch {b + 1}/{nb}: cum SR={np.mean(succ_all) * 100:.1f}%", flush=True)
        return succ_all

    res = {}
    for mode in ("bid", "closed_loop_1"):
        print(f"[{mode}]")
        s = run(mode)
        res[mode] = dict(sr=float(np.mean(s)), n=len(s))
    res["receding_baseline_ref"] = dict(sr=0.523, note="seed1000 300-ep collection")
    (out / "bid.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
