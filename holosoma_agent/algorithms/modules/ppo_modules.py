from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from holosoma_agent.algorithms.modules.modules import BaseModule, ModuleConfig


class PPOActor(nn.Module):
    """PPO actor with a learnable diagonal Gaussian noise std.

    Parameters
    ----------
    obs_dim_dict : dict[str, int]
        Flat observation dims (history included).
    module_config : ModuleConfig
    num_actions : int
    init_noise_std : float
    history_length : dict[str, int]
    """

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
        history_length: dict[str, int],
    ):
        super().__init__()
        module_config = self._process_module_config(module_config, num_actions)
        self.actor_module = BaseModule(obs_dim_dict, module_config, history_length)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = module_config.min_noise_std
        self.min_mean_noise_std = module_config.min_mean_noise_std
        self.distribution = None
        Normal.set_default_validate_args(False)
        print(f"Actor Module: {self.actor_module.module}")

    def _process_module_config(
        self, cfg: ModuleConfig, num_actions: int
    ) -> ModuleConfig:
        new_output_dim = []
        for v in cfg.output_dim:
            if v == "robot_action_dim":
                new_output_dim.append(num_actions)
            else:
                new_output_dim.append(v)
        from dataclasses import replace

        return replace(cfg, output_dim=new_output_dim)

    @property
    def actor(self):
        return self.actor_module

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs: torch.Tensor):
        mean = self.actor(actor_obs)
        if self.min_noise_std:
            clamped_std = torch.clamp(self.std, min=self.min_noise_std)
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        elif self.min_mean_noise_std:
            cur_mean = self.std.mean()
            if cur_mean < self.min_mean_noise_std:
                clamped_std = self.std * (self.min_mean_noise_std / (cur_mean + 1e-6))
            else:
                clamped_std = self.std
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        else:
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self.update_distribution(policy_state_dict["actor_obs"])
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(policy_state_dict["actor_obs"])

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")


class PPOCritic(nn.Module):
    """PPO critic (value network)."""

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config: ModuleConfig,
        history_length: dict[str, int],
    ):
        super().__init__()
        self.critic_module = BaseModule(obs_dim_dict, module_config, history_length)
        print(f"Critic Module: {self.critic_module.module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(policy_state_dict["critic_obs"])

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass


class PPOActorEncoder(PPOActor):
    """PPO actor with a separate encoder (MLPEncoder / CNNEncoder type)."""

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
        history_length: dict[str, int],
    ):
        super().__init__(
            obs_dim_dict, module_config, num_actions, init_noise_std, history_length
        )
        self.module_input_name = module_config.layer_config.module_input_name
        self.encoder_input_name = module_config.layer_config.encoder_input_name

    def _get_input(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != self.actor_module.input_dim:
            raise ValueError(
                f"Actor Obs must be {self.actor_module.input_dim}, got {actor_obs.shape[-1]}"
            )
        enc_obs = actor_obs[
            ..., self.actor_module.input_indices_dict[self.encoder_input_name]
        ]
        actor_enc = (
            self.actor_module.encoder(enc_obs)
            if self.actor_module.encoder is not None
            else enc_obs
        )
        state_obs = torch.cat(
            [
                actor_obs[..., self.actor_module.input_indices_dict[k]]
                for k in self.module_input_name
            ],
            dim=-1,
        )
        return torch.cat((actor_enc, state_obs), dim=-1)

    def act(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return super().act(
            {"actor_obs": self._get_input(policy_state_dict["actor_obs"])}
        )

    def act_inference(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return super().act_inference(
            {"actor_obs": self._get_input(policy_state_dict["actor_obs"])}
        )


class PPOCriticEncoder(PPOCritic):
    """PPO critic with a separate encoder."""

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config: ModuleConfig,
        history_length: dict[str, int],
    ):
        super().__init__(obs_dim_dict, module_config, history_length)
        self.module_input_name = module_config.layer_config.module_input_name
        self.encoder_input_name = module_config.layer_config.encoder_input_name

    def _get_input(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.shape[-1] != self.critic_module.input_dim:
            raise ValueError(
                f"Critic Obs must be {self.critic_module.input_dim}, got {critic_obs.shape[-1]}"
            )
        enc_obs = critic_obs[
            ..., self.critic_module.input_indices_dict[self.encoder_input_name]
        ]
        crit_enc = (
            self.critic_module.encoder(enc_obs)
            if self.critic_module.encoder is not None
            else enc_obs
        )
        state_obs = torch.cat(
            [
                critic_obs[..., self.critic_module.input_indices_dict[k]]
                for k in self.module_input_name
            ],
            dim=-1,
        )
        return torch.cat((crit_enc, state_obs), dim=-1)

    def evaluate(self, policy_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return super().evaluate(
            {"critic_obs": self._get_input(policy_state_dict["critic_obs"])}
        )


def setup_ppo_actor_module(
    obs_dim_dict: dict[str, int],
    module_config: ModuleConfig,
    num_actions: int,
    init_noise_std: float,
    device: str,
    history_length: dict[str, int],
) -> PPOActor:
    """Factory: create the correct PPOActor variant from *module_config.type*."""
    t = module_config.type
    if t in ("MLPEncoder", "CNNEncoder"):
        return PPOActorEncoder(
            obs_dim_dict, module_config, num_actions, init_noise_std, history_length
        ).to(device)
    if t == "MLP":
        return PPOActor(
            obs_dim_dict, module_config, num_actions, init_noise_std, history_length
        ).to(device)
    raise ValueError(f"Invalid actor type: {t}")


def setup_ppo_critic_module(
    obs_dim_dict: dict[str, int],
    module_config: ModuleConfig,
    device: str,
    history_length: dict[str, int],
) -> PPOCritic:
    """Factory: create the correct PPOCritic variant from *module_config.type*."""
    t = module_config.type
    if t in ("MLPEncoder", "CNNEncoder"):
        return PPOCriticEncoder(obs_dim_dict, module_config, history_length).to(device)
    if t == "MLP":
        return PPOCritic(obs_dim_dict, module_config, history_length).to(device)
    raise ValueError(f"Invalid critic type: {t}")
