from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Abstract base class for a vectorized environment.

    Subclass this and implement the abstract methods to plug your environment
    into a ``holosoma_agent`` algorithm (PPO or FastSAC).

    Required class / instance attributes
    ------------------------------------
    num_envs : int
        Number of parallel environments.
    num_actions : int
        Dimensionality of the action space.
    max_episode_length : int | torch.Tensor
        Maximum steps per episode (scalar or per-env tensor).
    episode_length_buf : torch.Tensor
        Current episode lengths, shape ``(num_envs,)``.
    device : str | torch.device
        Compute device (e.g. ``"cuda:0"``).
    """

    num_envs: int
    num_actions: int
    max_episode_length: int | torch.Tensor
    episode_length_buf: torch.Tensor
    device: str | torch.device

    @abstractmethod
    def get_observations(self) -> dict[str, torch.Tensor]:
        """Return the current observation dict without stepping the environment.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from observation group name → tensor of shape
            ``(num_envs, obs_dim)``.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Apply actions and advance the simulation by one step.

        Parameters
        ----------
        actions : torch.Tensor
            Shape ``(num_envs, num_actions)``.

        Returns
        -------
        obs_dict : dict[str, torch.Tensor]
            Observations after the step.
        rewards : torch.Tensor, shape ``(num_envs,)``
        dones : torch.Tensor, shape ``(num_envs,)``
        extras : dict
            May contain:
            - ``"time_outs"`` (torch.Tensor) – timeout flags
            - ``"episode"`` (dict) – per-episode scalars for logging
            - ``"to_log"`` (dict) – arbitrary tensors to log
        """
        raise NotImplementedError
