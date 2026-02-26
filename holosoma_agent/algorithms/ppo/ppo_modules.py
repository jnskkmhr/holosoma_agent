from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal

from holosoma_agent.configs.ppo_config import ModuleConfig
from holosoma_agent.algorithms.ppo.modules import BaseModule


class PPOActor(nn.Module):
    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config_dict: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
    ):
        super().__init__()

        self.actor_module = BaseModule(obs_dim_dict, num_actions, module_config_dict)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_noise_std = module_config_dict.min_noise_std
        self.min_mean_noise_std = module_config_dict.min_mean_noise_std
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
        print(f"Actor Module: {self.actor_module.module}")

    @property
    def actor(self):
        return self.actor_module

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

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
            current_mean = self.std.mean()
            if current_mean < self.min_mean_noise_std:
                scale_up = self.min_mean_noise_std / (current_mean + 1e-6)
                clamped_std = self.std * scale_up
            else:
                clamped_std = self.std
            self.distribution = Normal(mean, mean * 0.0 + clamped_std)
        else:
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_state_dict: dict[str, torch.Tensor]):
        self.update_distribution(policy_state_dict["policy"])
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_state_dict: dict[str, torch.Tensor]):
        return self.actor(policy_state_dict["policy"])

    def to_cpu(self):
        self.actor = deepcopy(self.actor).to("cpu")
        self.std.to("cpu")


class PPOCritic(nn.Module):
    def __init__(self, obs_dim_dict: dict[str, int], module_config_dict: ModuleConfig):
        super().__init__()
        self.critic_module = BaseModule(obs_dim_dict, 1, module_config_dict)
        print(f"Critic Module: {self.critic_module.module}")

    @property
    def critic(self):
        return self.critic_module

    def reset(self, dones=None):
        pass

    def evaluate(self, critic_state_dict: dict[str, torch.Tensor]):
        critic_obs = critic_state_dict["critic"]
        return self.critic(critic_obs)

    def get_hidden_states(self):
        return None

    def set_hidden_states(self, hidden_states):
        pass


class PPOActorEncoder(PPOActor):
    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config_dict: ModuleConfig,
        num_actions: int,
        init_noise_std: float,
    ):
        super().__init__(obs_dim_dict, module_config_dict, num_actions, init_noise_std)

    def process_encoder(
        self, obs: torch.Tensor, encoder_obs: torch.Tensor
    ) -> torch.Tensor:
        encoder_obs = (
            self.actor_module.encoder(encoder_obs)
            if self.actor_module.encoder is not None
            else encoder_obs
        )
        return torch.cat([encoder_obs, obs], dim=-1)

    def act(self, policy_state_dict: dict[str, torch.Tensor]):
        actor_obs = policy_state_dict["policy"]
        encoder_obs = policy_state_dict["policy_encoder"]
        input_actor = self.process_encoder(actor_obs, encoder_obs)
        return super().act({"policy": input_actor})

    def act_inference(self, policy_state_dict: dict[str, torch.Tensor]):
        actor_obs = policy_state_dict["policy"]
        encoder_obs = policy_state_dict["policy_encoder"]
        input_actor = self.process_encoder(actor_obs, encoder_obs)
        return super().act_inference({"policy": input_actor})


class PPOCriticEncoder(PPOCritic):
    def __init__(self, obs_dim_dict: dict[str, int], module_config_dict: ModuleConfig):
        super().__init__(obs_dim_dict, module_config_dict)

    def process_encoder(
        self, obs: torch.Tensor, encoder_obs: torch.Tensor
    ) -> torch.Tensor:
        encoder_obs = (
            self.critic_module.encoder(encoder_obs)
            if self.critic_module.encoder is not None
            else encoder_obs
        )
        return torch.cat([encoder_obs, obs], dim=-1)

    def evaluate(self, critic_state_dict: dict[str, torch.Tensor]):
        critic_obs = critic_state_dict["critic"]
        encoder_obs = critic_state_dict["critic_encoder"]
        input_critic = self.process_encoder(critic_obs, encoder_obs)
        return super().evaluate({"critic": input_critic})


def setup_ppo_actor_module(
    obs_dim_dict: dict[str, int],
    module_config: ModuleConfig,
    num_actions: int,
    init_noise_std: float,
    device: torch.device | str,
):
    module_type = module_config.module_type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
        ).to(device)
    if module_type == "MLP":
        return PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
        ).to(device)

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict: dict[str, int],
    module_config: ModuleConfig,
    device: torch.device | str,
):
    module_type = module_config.module_type
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOCriticEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
        ).to(device)
    if module_type == "MLP":
        return PPOCritic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
        ).to(device)
    raise ValueError(f"Invalid critic type: {module_type}")
