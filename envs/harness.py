"""Rollout harness: wrap a frozen LeRobot policy + env + processors so we can
drive chunk-level, receding-horizon rollouts with full control over sampling.

Exposes the pieces GUARD needs that `policy.select_action` hides:
  - per-env observation history management (queues)
  - K full-horizon action chunks sampled in ONE batched diffusion call
  - the policy's global conditioning embedding (for the OOD trigger term)
  - action unnormalization back to env space

Currently specialized to Diffusion Policy (the primary base); the sampling hook
is the only policy-specific part and can be swapped for a flow policy later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


@dataclass
class StepResult:
    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


class RolloutHarness:
    def __init__(self, env_cfg, ckpt: str, n_envs: int, device: str = "cuda"):
        self.device = device
        self.policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        self.cfg = self.policy.config
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.cfg
        )
        env_dict = make_env(env_cfg, n_envs=n_envs)
        suite = next(iter(env_dict))
        self.env = env_dict[suite][next(iter(env_dict[suite]))]
        self.n_envs = n_envs
        self.max_steps = int(self.env.call("_max_episode_steps")[0])
        self.horizon = self.cfg.horizon
        self.n_action_steps = self.cfg.n_action_steps
        self.n_obs_steps = self.cfg.n_obs_steps
        self.exec_start = self.n_obs_steps - 1  # first executable index in the full horizon
        self._obs_keys = [OBS_STATE, OBS_IMAGES]
        if self.cfg.env_state_feature:
            self._obs_keys.append(OBS_ENV_STATE)

    # ---- episode lifecycle ----
    def reset(self, seeds):
        self.policy.reset()
        obs, info = self.env.reset(seed=list(seeds))
        return obs

    def observe(self, raw_obs) -> dict:
        """Feed one env observation, update history queues, return the stacked
        obs batch used for conditioning: {key: (E, n_obs_steps, ...)}."""
        o = preprocess_observation(raw_obs)
        o = add_envs_task(self.env, o)
        o = self.env_pre(o)
        o = self.pre(o)
        batch = dict(o)
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.cfg.image_features], dim=-4)
        self.policy._queues = populate_queues(self.policy._queues, batch, exclude_keys=[ACTION])
        return {k: torch.stack(list(self.policy._queues[k]), dim=1) for k in self._obs_keys}

    # ---- sampling ----
    @torch.no_grad()
    def global_cond(self, stacked: dict) -> torch.Tensor:
        return self.policy.diffusion._prepare_global_conditioning(stacked)  # (E, D)

    @torch.no_grad()
    def sample_chunks(
        self,
        stacked: dict,
        k: int,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ):
        """Return (E, k, horizon, act_dim) normalized chunks + (E, D) global cond.

        All k samples per env are drawn in a single batched diffusion call.
        `num_inference_steps` overrides the sampler step count (for A2 deliberation).
        """
        gcond = self.policy.diffusion._prepare_global_conditioning(stacked)  # (E, D)
        e, d = gcond.shape
        gc = gcond.repeat_interleave(k, dim=0)  # (E*k, D)
        diff = self.policy.diffusion
        if num_inference_steps is not None:
            saved = diff.num_inference_steps
            diff.num_inference_steps = num_inference_steps
        try:
            noise = None
            if generator is not None:
                noise = torch.randn(
                    (e * k, self.horizon, self.cfg.action_feature.shape[0]),
                    device=self.device,
                    dtype=gcond.dtype,
                    generator=generator,
                )
            chunks = diff.conditional_sample(e * k, global_cond=gc, noise=noise)
        finally:
            if num_inference_steps is not None:
                diff.num_inference_steps = saved
        return chunks.view(e, k, self.horizon, -1), gcond

    def to_real(self, a_norm: torch.Tensor) -> np.ndarray:
        return self.post(a_norm).to("cpu").numpy()

    def step(self, a_np: np.ndarray) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(a_np)
        return StepResult(
            obs, np.asarray(reward), np.asarray(terminated), np.asarray(truncated), info
        )

    @staticmethod
    def successes(info: dict, n_envs: int):
        if "final_info" in info and isinstance(info["final_info"], dict):
            return np.asarray(info["final_info"]["is_success"], dtype=bool)
        return np.zeros(n_envs, dtype=bool)
