"""Robomimic (robosuite) evaluation harness with the same duck-typed interface
as envs.harness.RolloutHarness, so guard.collect / guard.controller run unchanged.

Matches the `ankile/robomimic-ph-square-image` lerobot dataset:
  observation.state                      (9,)  = eef_pos(3) + eef_quat(4) + gripper_qpos(2)
  observation.images.agentview           (84,84,3) uint8
  observation.images.robot0_eye_in_hand  (84,84,3) uint8
  action                                 (7,)  OSC_POSE delta + gripper, in [-1,1]

Notes:
  - robosuite offscreen renders are vertically flipped vs the robomimic dataset
    convention; we flip them upright (set FLIP_IMAGES=False if dataset probing
    shows otherwise).
  - No native vector env: we hold E independent envs and step them in a loop.
    Success = env._check_success() at any step. Horizon 400 (robomimic standard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

FLIP_IMAGES = True
CAMERAS = ["agentview", "robot0_eye_in_hand"]
IMG_KEYS = [f"observation.images.{c}" for c in CAMERAS]


@dataclass
class StepResult:
    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


def _make_env(task="NutAssemblySquare"):
    import robosuite
    from robosuite.controllers import load_controller_config

    cfg = load_controller_config(default_controller="OSC_POSE")
    return robosuite.make(
        task,
        robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERAS,
        camera_heights=84,
        camera_widths=84,
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
        hard_reset=False,
    )


class RobomimicHarness:
    def __init__(
        self,
        ckpt: str,
        n_envs: int,
        device: str = "cuda",
        task: str = "NutAssemblySquare",
        max_steps: int = 400,
    ):
        self.device = device
        self.policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        self.cfg = self.policy.config
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.envs = [_make_env(task) for _ in range(n_envs)]
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.horizon = self.cfg.horizon
        self.n_action_steps = self.cfg.n_action_steps
        self.n_obs_steps = self.cfg.n_obs_steps
        self.exec_start = self.n_obs_steps - 1
        self._obs_keys = [OBS_STATE, OBS_IMAGES]
        self._succeeded = np.zeros(n_envs, bool)
        self._steps = 0

    # ---- raw robosuite obs -> lerobot-keyed numpy dict ----
    def _convert(self, obs_list) -> dict:
        state = np.stack(
            [
                np.concatenate(
                    [o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"]]
                )
                for o in obs_list
            ]
        ).astype(np.float32)
        out = {"agent_pos": state}
        images = {}
        for cam in CAMERAS:
            im = np.stack([o[f"{cam}_image"] for o in obs_list])  # (E,84,84,3) uint8
            if FLIP_IMAGES:
                im = im[:, ::-1].copy()
            images[cam] = im
        out["pixels"] = images
        return out

    # ---- episode lifecycle (mirrors RolloutHarness API) ----
    def reset(self, seeds):
        self.policy.reset()
        self._succeeded[:] = False
        self._steps = 0
        obs_list = []
        for env, s in zip(self.envs, seeds):
            np.random.seed(int(s) + 7919)  # robosuite randomizes via global numpy rng
            obs_list.append(env.reset())
        return self._convert(obs_list)

    def observe(self, raw_obs) -> dict:
        state = torch.from_numpy(raw_obs["agent_pos"]).to(self.device)
        batch = {OBS_STATE: state}
        for cam in CAMERAS:
            im = torch.from_numpy(raw_obs["pixels"][cam]).to(self.device)
            im = im.permute(0, 3, 1, 2).float() / 255.0  # (E,3,84,84)
            batch[f"observation.images.{cam}"] = im
        batch = self.pre(batch)
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in self.cfg.image_features], dim=-4)
        self.policy._queues = populate_queues(self.policy._queues, batch, exclude_keys=[ACTION])
        return {k: torch.stack(list(self.policy._queues[k]), dim=1) for k in self._obs_keys}

    @torch.no_grad()
    def sample_chunks(self, stacked, k, generator=None, num_inference_steps=None):
        gcond = self.policy.diffusion._prepare_global_conditioning(stacked)
        e, _ = gcond.shape
        gc = gcond.repeat_interleave(k, dim=0)
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
        obs_list, rewards = [], []
        for i, env in enumerate(self.envs):
            o, r, _, _ = env.step(a_np[i].astype(np.float64))
            obs_list.append(o)
            rewards.append(r)
            if env._check_success():
                self._succeeded[i] = True
        self._steps += 1
        term = self._succeeded.copy()  # stop successful episodes
        trunc = np.full(self.n_envs, self._steps >= self.max_steps)
        info = (
            {"final_info": {"is_success": self._succeeded.copy()}}
            if (term.any() or trunc.any())
            else {}
        )
        return StepResult(
            self._convert(obs_list), np.asarray(rewards, np.float32), term, trunc, info
        )

    @staticmethod
    def successes(info: dict, n_envs: int):
        if "final_info" in info and isinstance(info["final_info"], dict):
            return np.asarray(info["final_info"]["is_success"], dtype=bool)
        return np.zeros(n_envs, dtype=bool)

    def close(self):
        for env in self.envs:
            env.close()
