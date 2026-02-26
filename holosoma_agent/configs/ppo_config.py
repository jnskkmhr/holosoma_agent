from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, MISSING

from .symmetry_config import SymmetryConfig


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str = MISSING  # not used
    """Target optimizer class (e.g., torch.optim.AdamW)."""
    weight_decay: float = MISSING
    """Weight decay parameter for the optimizer."""


@dataclass
class LayerConfig:
    """Configuration for neural network layer settings."""

    """
    mlp settings
    """

    hidden_dims: list[int] = MISSING
    """List of hidden layer dimensions."""
    activation: str = MISSING
    """Activation function name."""
    dropout_prob: float = MISSING
    """Dropout probability."""
    use_layer_norm: bool = MISSING
    """Whether to use layer normalization."""

    """
    mlp encoder settings
    """

    encoder_activation: str = MISSING
    """Activation function name for encoder layers."""
    encoder_output_dim: int | None = MISSING
    """Output dimension for encoder. Only used for encoder modules."""
    encoder_hidden_dims: list[int] | None = MISSING
    """Hidden dimensions for encoder. Only used for encoder modules."""

    """
    cnn encoder settings
    """

    hidden_channels: tuple[int, ...] | None = MISSING
    """Hidden channel dimensions. Only used for CNN modules."""
    kernel_size: int | tuple[int, ...] = MISSING
    """Kernel size for convolutions. Only used for CNN modules."""
    stride: int | tuple[int, ...] = MISSING
    """Stride for convolutions. Only used for CNN modules."""
    padding: str | int | tuple[str | int, ...] = MISSING
    """Padding mode for convolutions. Only used for CNN modules."""


@dataclass
class ModuleConfig:
    """Configuration for neural network modules."""

    module_type: Literal["MLP", "MLPEncoder", "CNNEncoder"] = MISSING
    """Module type (e.g., MLP)."""

    obs_keys: list[str] = MISSING
    """List of observation keys."""

    layer_config: LayerConfig = MISSING
    """Feature extraction layer settings."""

    min_noise_std: float | None = MISSING
    """Minimum noise standard deviation."""

    min_mean_noise_std: float | None = MISSING
    """Minimum mean noise standard deviation."""


@dataclass
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig = MISSING
    """Actor module configuration."""

    critic: ModuleConfig = MISSING
    """Critic module configuration."""


@dataclass
class PPOConfig:
    """Configuration for the PPO algorithm."""

    # ── Training loop ─────────────────────────────────────────────────────────
    num_learning_iterations: int = MISSING
    """Total number of training iterations (each = one rollout + update)."""
    save_interval: int = MISSING
    """Save a checkpoint every this many iterations."""
    load_optimizer: bool = MISSING
    """Whether to restore optimizer state when loading a checkpoint."""
    init_at_random_ep_len: bool = MISSING
    """Randomise the initial episode-length counter to de-correlate resets."""

    # ── Network architecture ──────────────────────────────────────────────────
    module_dict: PPOModuleDictConfig = MISSING
    init_noise_std: float = MISSING
    """Initial standard deviation of the actor's action distribution."""

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    num_steps_per_env: int = MISSING
    """Rollout length per environment before each update."""
    num_learning_epochs: int = MISSING
    """Number of epochs over the collected rollout per update."""
    num_mini_batches: int = MISSING
    """Number of mini-batches per epoch."""
    clip_param: float = MISSING
    """PPO clipping epsilon."""
    gamma: float = MISSING
    """Discount factor."""
    lam: float = MISSING
    """GAE lambda."""
    value_loss_coef: float = MISSING
    """Weight of the value loss."""
    entropy_coef: float = MISSING
    """Entropy bonus coefficient."""
    max_grad_norm: float = MISSING
    """Gradient clipping max norm."""

    # ── optimizer ─────────────────────────────────────────────────────────
    actor_learning_rate: float = MISSING
    actor_optimizer: OptimizerConfig = MISSING
    critic_learning_rate: float = MISSING
    critic_optimizer: OptimizerConfig = MISSING
    schedule: str = MISSING
    """LR schedule. ``"adaptive"`` adjusts based on KL, ``"fixed"`` keeps it constant."""
    desired_kl: float = MISSING
    """Target KL divergence for adaptive LR."""
    max_actor_learning_rate: float | None = MISSING
    min_actor_learning_rate: float | None = MISSING
    max_critic_learning_rate: float | None = MISSING
    min_critic_learning_rate: float | None = MISSING

    # ── Symmetry augmentation ─────────────────────────────────────────────────
    use_symmetry: bool = MISSING
    """Whether to apply x-z plane symmetry augmentation during training."""
    symmetry_config: SymmetryConfig | None = MISSING
    symmetry_actor_coef: float = MISSING
    symmetry_critic_coef: float = MISSING
