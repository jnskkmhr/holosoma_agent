from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from holosoma_agent.env.vec_env import VecEnv
from holosoma_agent.configs import AlgoInitConfig


class BaseAlgo(ABC):
    """Abstract base class for RL algorithms.

    Parameters
    ----------
    env : VecEnv
        The training environment (PPOVecEnv or FastSACVecEnv).
    config : AlgoInitConfig
        Algorithm-specific config (PPOConfig or FastSACConfig).
    device : str
        Compute device, e.g. ``"cuda:0"``.
    multi_gpu_cfg : dict | None
        Optional dict with keys ``global_rank``, ``local_rank``, ``world_size``
        for DDP training.
    """

    def __init__(
        self,
        env: VecEnv,
        config: AlgoInitConfig,
        device: str,
        multi_gpu_cfg: dict | None = None,
    ):
        self.env = env
        self.config = config
        self.device = device

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank: int = multi_gpu_cfg["global_rank"]
            self.gpu_local_rank: int = multi_gpu_cfg["local_rank"]
            self.gpu_world_size: int = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_local_rank = 0
            self.gpu_world_size = 1
        self.is_main_process: bool = self.gpu_global_rank == 0

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def setup(self) -> None:
        """Initialise networks, optimisers, and storage."""
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> None:
        """Run the main training loop."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Save a checkpoint to *path*."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a checkpoint from *path*."""
        raise NotImplementedError

    # ── Inference interface ────────────────────────────────────────────────────

    @property
    def actor_onnx_wrapper(self):
        """Return an ONNX-exportable wrapper for the actor."""
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        raise NotImplementedError

    # ── Optional interface ────────────────────────────────────────────────────

    def _collect_env_state(self) -> dict[str, torch.Tensor | float]:
        """Collect environment state for checkpointing via the environment interface."""
        # NOTE: skip implementation for now (see src/holosoma/holosoma/envs/locomotion/locomotion_manager.py)
        return {}

    def _restore_env_state(
        self, env_state: dict[str, torch.Tensor | float] | None
    ) -> None:
        """Restore environment state from checkpoint via the environment interface."""
        # NOTE: skip implementation for now (see src/holosoma/holosoma/envs/locomotion/locomotion_manager.py)
        return

    # ── Multi-GPU helpers ─────────────────────────────────────────────────────

    def _synchronize_model_parameters(self, *models: torch.nn.Module) -> None:
        """Broadcast all parameter tensors from rank 0 to every other rank."""
        raise NotImplementedError

    def _all_reduce_model_grads(self, *models: torch.nn.Module) -> None:
        """Average gradients across all ranks (equivalent to DDP)."""
        raise NotImplementedError

    def has_curricula_enabled(self) -> bool:
        """Check if any curricula are enabled in the environment.

        This helper method checks for the presence of various curriculum flags
        to determine if any curriculum learning is active. This is commonly used
        for multi-GPU synchronization and logging purposes.

        Returns
        -------
        bool
            True if any curriculum is enabled, False otherwise.
        """
        return getattr(self.env, "use_reward_penalty_curriculum", False) or getattr(
            self.env, "use_domain_rand_scale_curriculum", False
        )

    def _synchronize_curriculum_metrics(self):
        """Synchronize curriculum-related metrics across all GPUs."""
        # Check if any curricula are enabled before synchronizing
        if not self.has_curricula_enabled():
            return

        self.env.synchronize_curriculum_state(
            device=self.device, world_size=self.gpu_world_size
        )
