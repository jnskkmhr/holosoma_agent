from __future__ import annotations

from dataclasses import dataclass, MISSING

from .symmetry_config import SymmetryConfig


@dataclass
class FastSACConfig:
    """Configuration for the FastSAC algorithm."""

    # ── Training loop ─────────────────────────────────────────────────────────
    num_learning_iterations: int = MISSING
    """Total environment timesteps."""
    learning_starts: int = MISSING
    """Timestep at which gradient updates begin."""
    save_interval: int = MISSING
    """Save a checkpoint every this many iterations."""
    logging_interval: int = MISSING

    # ── Networks ──────────────────────────────────────────────────────────────
    actor_hidden_dim: int = MISSING
    critic_hidden_dim: int = MISSING
    use_layer_norm: bool = MISSING
    num_q_networks: int = MISSING
    """Number of Q-networks in the ensemble."""

    # ── Distributional critic ─────────────────────────────────────────────────
    num_atoms: int = MISSING
    v_min: float = MISSING
    v_max: float = MISSING

    # ── CNN encoder ───────────────────────────────────────────────────────────
    use_cnn_encoder: bool = MISSING
    encoder_obs_key: str = MISSING
    encoder_obs_shape: tuple[int, int, int] = MISSING

    # ── Observation keys ──────────────────────────────────────────────────────
    actor_obs_keys: list[str] = MISSING
    critic_obs_keys: list[str] = MISSING

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer_size: int = MISSING
    """Per-environment replay buffer capacity."""
    num_steps: int = MISSING
    """N-step return horizon."""
    batch_size: int = MISSING

    # ── SAC hyperparameters ───────────────────────────────────────────────────
    gamma: float = MISSING
    tau: float = MISSING
    """Target network soft-update coefficient."""
    policy_frequency: int = MISSING
    """Actor update frequency (every N critic updates)."""
    num_updates: int = MISSING
    """Gradient updates per environment step."""

    # ── Entropy temperature (alpha) ───────────────────────────────────────────
    alpha_init: float = MISSING
    use_autotune: bool = MISSING
    target_entropy_ratio: float = MISSING

    # ── Action ────────────────────────────────────────────────────────────────
    use_tanh: bool = MISSING
    log_std_max: float = MISSING
    log_std_min: float = MISSING

    # ── Optimizers ────────────────────────────────────────────────────────────
    critic_learning_rate: float = MISSING
    actor_learning_rate: float = MISSING
    alpha_learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_grad_norm: float = MISSING
    """Gradient clipping norm (0 = disabled)."""

    # ── augmentation ────────────────────────────────────────────────────────
    obs_normalization: bool = MISSING
    use_symmetry: bool = MISSING
    symmetry_config: SymmetryConfig | None = MISSING

    # ── training settings ───────────────────────────────────────────────────
    compile: bool = MISSING
    """Use ``torch.compile`` for the update functions."""
    amp: bool = MISSING
    """Automatic Mixed Precision."""
    amp_dtype: str = MISSING
    """AMP dtype: ``"bf16"`` or ``"fp16"``."""
