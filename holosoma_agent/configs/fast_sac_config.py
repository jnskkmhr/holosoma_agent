from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .symmetry_config import SymmetryConfig


@dataclass
class FastSACConfig:
    """Configuration for the FastSAC algorithm."""

    # ── Training loop ─────────────────────────────────────────────────────────
    num_learning_iterations: int = 25_000
    """Total environment timesteps."""

    learning_starts: int = 10
    """Timestep at which gradient updates begin."""

    save_interval: int = 1_000
    """Save a checkpoint every this many iterations."""

    logging_interval: int = 100

    # ── Networks ──────────────────────────────────────────────────────────────
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768

    use_layer_norm: bool = True

    num_q_networks: int = 2
    """Number of Q-networks in the ensemble."""

    # ── Distributional critic ─────────────────────────────────────────────────
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0

    # ── CNN encoder ───────────────────────────────────────────────────────────
    use_cnn_encoder: bool = False
    encoder_obs_key: str = "perception_obs"
    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)

    # ── Observation keys ──────────────────────────────────────────────────────
    actor_obs_keys: List[str] = field(default_factory=lambda: ["policy"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic"])

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer_size: int = 1_024
    """Per-environment replay buffer capacity."""

    num_steps: int = 1
    """N-step return horizon."""

    batch_size: int = 8_192

    # ── SAC hyperparameters ───────────────────────────────────────────────────
    gamma: float = 0.97
    tau: float = 0.125
    """Target network soft-update coefficient."""

    policy_frequency: int = 4
    """Actor update frequency (every N critic updates)."""

    num_updates: int = 8
    """Gradient updates per environment step."""

    # ── Entropy temperature (alpha) ───────────────────────────────────────────
    alpha_init: float = 0.001
    use_autotune: bool = True
    target_entropy_ratio: float = 0.0

    # ── Action ────────────────────────────────────────────────────────────────
    use_tanh: bool = True
    log_std_max: float = 0.0
    log_std_min: float = -5.0

    # ── Optimizers ────────────────────────────────────────────────────────────
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    """Gradient clipping norm (0 = disabled)."""

    # ── Misc ──────────────────────────────────────────────────────────────────
    compile: bool = True
    """Use ``torch.compile`` for the update functions."""

    obs_normalization: bool = True

    use_symmetry: bool = False
    symmetry_config: SymmetryConfig | None = None

    amp: bool = True
    """Automatic Mixed Precision."""

    amp_dtype: str = "bf16"
    """AMP dtype: ``"bf16"`` or ``"fp16"``."""
