from __future__ import annotations
from typing import TypedDict
import torch
from torch import Tensor


class Minibatch(TypedDict):
    """A minibatch of data for training a PPO agent."""

    actor_obs: torch.Tensor
    """The observation of the actor.

    Shape: (mini_batch_size, actor_obs_dim), dtype: torch.float32
    """

    critic_obs: torch.Tensor
    """The observation of the critic.

    Shape: (mini_batch_size, critic_obs_dim), dtype: torch.float32
    """

    actor_obs_encoder: torch.Tensor
    """The encoded observation of the actor.

    Shape: (mini_batch_size, actor_obs_encoder_dim), dtype: torch.float32
    """

    critic_obs_encoder: torch.Tensor
    """The encoded observation of the critic.

    Shape: (mini_batch_size, critic_obs_encoder_dim), dtype: torch.float32
    """

    actions: torch.Tensor
    """The actions taken by the agent.

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    rewards: torch.Tensor
    """The rewards received from the environment.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    dones: torch.Tensor
    """Whether each episode is done after taking the action.

    Shape: (mini_batch_size, 1), dtype: torch.bool
    """

    values: torch.Tensor
    """The value estimates from the critic.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    returns: torch.Tensor
    """The computed (unnormalized) returns for each step.

    The returns are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    advantages: torch.Tensor
    """The computed (normalized) advantages for each step.

    The advantages are computed following Generalized Advantage Estimation (GAE).

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    actions_log_prob: torch.Tensor
    """The log probabilities of the actions.

    Shape: (mini_batch_size, 1), dtype: torch.float32
    """

    action_mean: torch.Tensor
    """The mean of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """

    action_sigma: torch.Tensor
    """The standard deviation of the action distribution (assuming Gaussian distribution).

    Shape: (mini_batch_size, num_act), dtype: torch.float32
    """


class RolloutStorage:
    """Simple buffer for storing rollout data during PPO training.

    Stores transitions in pre-allocated tensors and provides a mini-batch
    generator.  Register every key you want to track before calling
    :meth:`add`.

    Example
    -------
    ::

        storage = RolloutStorage(num_envs=4096, num_transitions_per_env=24)
        storage.register("actor_obs", shape=(48,))
        storage.register("actions",   shape=(12,))
        storage.register("rewards",   shape=())
        storage.register("dones",     shape=())

        for t in range(num_steps):
            storage.add(actor_obs=..., actions=..., rewards=..., dones=...)

        for batch in storage.mini_batch_generator(num_mini_batches=4, num_epochs=8):
            ...  # batch is a dict[str, Tensor]

        storage.clear()
    """

    def __init__(
        self, num_envs: int, num_transitions_per_env: int, device: str = "cpu"
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0
        self._buffers: dict[str, Tensor] = {}

    def register(
        self,
        key: str,
        shape: tuple[int, ...] | list[int] = (),
        dtype: torch.dtype = torch.float,
    ):
        """Register a new data key.

        Parameters
        ----------
        key : str
        shape : tuple[int, ...]
            Per-sample shape (excluding num_transitions and num_envs dims).
        dtype : torch.dtype
        """
        if key in self._buffers:
            raise ValueError(f"Key '{key}' already registered")
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, *shape),
            dtype=dtype,
            device=self.device,
        )
        self._buffers[key] = buffer

    def add(self, **data: Tensor):
        """Add one transition step to all registered buffers.

        Unregistered keys in *data* are silently ignored.
        """
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError(
                f"Buffer overflow: step {self.step} >= {self.num_transitions_per_env}"
            )
        for key, value in data.items():
            if key not in self._buffers:
                continue
            if value.requires_grad:
                raise ValueError(
                    f"Cannot store tensor with requires_grad=True for key '{key}'"
                )
            self._buffers[key][self.step].copy_(value)
        self.step += 1

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._buffers:
            raise KeyError(f"Key '{key}' not registered")
        return self._buffers[key]

    def __setitem__(self, key: str, value: Tensor):
        if key not in self._buffers:
            raise KeyError(f"Key '{key}' not registered")
        if value.requires_grad:
            raise ValueError("Cannot store tensor with requires_grad=True")
        self._buffers[key].copy_(value)

    def clear(self):
        """Reset the step counter (does not re-allocate buffers)."""
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Yield randomised mini-batches over the stored rollout.

        Yields
        ------
        dict[str, Tensor]
            Each value has shape ``(mini_batch_size, *per_sample_shape)``.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )
        flattened = {key: buf.flatten(0, 1) for key, buf in self._buffers.items()}
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield {key: flattened[key][batch_indices] for key in self._buffers}
