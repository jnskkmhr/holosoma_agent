from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    """Configuration for the PPO algorithm."""

    # ── Network architecture ──────────────────────────────────────────────────
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """Hidden layer dims for the actor MLP."""

    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """Hidden layer dims for the critic MLP."""

    activation: str = "ELU"
    """Activation function (any ``torch.nn`` name, e.g. ``"ELU"``, ``"ReLU"``)."""

    use_layer_norm: bool = False
    """Whether to apply layer normalization inside the MLPs."""

    init_noise_std: float = 0.8
    """Initial standard deviation of the actor's action distribution."""

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    num_steps_per_env: int = 24
    """Rollout length per environment before each update."""

    num_learning_epochs: int = 8
    """Number of epochs over the collected rollout per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping epsilon."""

    gamma: float = 0.99
    """Discount factor."""

    lam: float = 0.95
    """GAE lambda."""

    value_loss_coef: float = 1.0
    """Weight of the value loss."""

    entropy_coef: float = 0.01
    """Entropy bonus coefficient."""

    max_grad_norm: float = 1.0
    """Gradient clipping max norm."""

    # ── Learning rate ─────────────────────────────────────────────────────────
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-5

    schedule: str = "adaptive"
    """LR schedule. ``"adaptive"`` adjusts based on KL, ``"fixed"`` keeps it constant."""

    desired_kl: float = 0.01
    """Target KL divergence for adaptive LR."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    # ── Symmetry augmentation ─────────────────────────────────────────────────
    use_symmetry: bool = False
    """Whether to apply x-z plane symmetry augmentation during training."""

    symmetry_actor_coef: float = 1.0
    symmetry_critic_coef: float = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    num_learning_iterations: int = 1_000_000
    """Total number of training iterations (each = one rollout + update)."""

    save_interval: int = 100
    """Save a checkpoint every this many iterations."""

    load_optimizer: bool = True
    """Whether to restore optimizer state when loading a checkpoint."""

    init_at_random_ep_len: bool = True
    """Randomise the initial episode-length counter to de-correlate resets."""
