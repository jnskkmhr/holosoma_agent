from __future__ import annotations

from abc import abstractmethod

import torch

from holosoma_agent.env.vec_env import VecEnv


class FastSACVecEnv(VecEnv):
    """Abstract environment interface for FastSAC.

    Extends :class:`VecEnv` with the observation-key and action-boundary API
    that ``FastSACAgent`` uses.

    Subclass this, implement all abstract methods, and set:

    The most critical difference from :class:`PPOVecEnv` is
    :meth:`get_action_boundaries`, which tells FastSAC the per-DOF action
    scale limits (replacing holosoma's ``_compute_action_boundaries`` that
    read ``robot_config``).

    Example
    -------
    ::

        class MyEnv(FastSACVecEnv):
            num_envs = 4096
            num_actions = 12
            max_episode_length = 500

            def get_action_boundaries(self):
                # per-DOF upper bounds for tanh scaling
                return torch.ones(self.num_actions, device=self.device) * 3.14
            ...
    """

    @abstractmethod
    def _compute_action_boundaries(self) -> torch.Tensor:
        """Return per-DOF action upper bounds used for tanh scaling.

        Replaces the ``_compute_action_boundaries`` method in the original
        ``FastSACAgent``, which read from ``robot_config``.

        Returns
        -------
        torch.Tensor, shape ``(num_actions,)``
            Positive upper bound for each action dimension.
        """
        raise NotImplementedError
