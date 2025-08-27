"""A trained policy together with the observation statistics it was trained on."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlmm.ppo.normalize import ObsNormalizer
from rlmm.ppo.policy import ActorCritic


class Agent:
    def __init__(
        self,
        obs_dim: int,
        nvec: Sequence[int],
        hidden: Sequence[int],
        obs_clip: float = 10.0,
        normalize_obs: bool = True,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.nvec = list(nvec)
        self.hidden = list(hidden)
        self.device = torch.device(device)
        self.policy = ActorCritic(obs_dim, self.nvec, self.hidden).to(self.device)
        self.normalize_obs = normalize_obs
        self.obs_norm = ObsNormalizer((obs_dim,), clip=obs_clip)

    def preprocess(self, obs: np.ndarray, update: bool = False) -> torch.Tensor:
        x = self.obs_norm(obs, update=update) if self.normalize_obs else obs.astype(np.float32)
        return torch.as_tensor(x, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Actions for a batch ``(B, obs_dim)`` or a single observation."""
        single = obs.ndim == 1
        x = self.preprocess(obs[None] if single else obs)
        a, _, _ = self.policy.act(x, deterministic=deterministic)
        out = a.cpu().numpy()
        return out[0] if single else out

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "nvec": self.nvec,
                "hidden": self.hidden,
                "normalize_obs": self.normalize_obs,
                "obs_clip": self.obs_norm.clip,
                "obs_rms": self.obs_norm.rms.state_dict(),
                "policy": {k: v.cpu() for k, v in self.policy.state_dict().items()},
                "meta": meta or {},
            },
            path,
        )

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> tuple[Agent, dict[str, Any]]:
        d = torch.load(path, map_location=device, weights_only=False)
        agent = Agent(
            d["obs_dim"],
            d["nvec"],
            d["hidden"],
            obs_clip=d["obs_clip"],
            normalize_obs=d["normalize_obs"],
            device=device,
        )
        agent.obs_norm.rms.load_state_dict(d["obs_rms"])
        agent.policy.load_state_dict(d["policy"])
        agent.policy.eval()
        meta: dict[str, Any] = d["meta"]
        return agent, meta
