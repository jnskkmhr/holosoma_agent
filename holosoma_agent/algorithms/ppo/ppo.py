from __future__ import annotations

import os
from typing import TypedDict

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from loguru import logger

from holosoma_agent.algorithms.base_algo import BaseAlgo
from holosoma_agent.algorithms.ppo.ppo_modules import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)

from holosoma_agent.utils.symmetry_utils import SymmetryUtils
from holosoma_agent.utils.logger import Logger
from holosoma_agent.algorithms.ppo.data_utils import RolloutStorage
from holosoma_agent.configs.ppo_config import PPOConfig
from holosoma_agent.env.vec_env import VecEnv


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


class PPO(BaseAlgo):
    config: PPOConfig

    def __init__(
        self,
        env: VecEnv,
        config: PPOConfig,
        log_dir,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = log_dir
        self.logger = Logger(
            log_dir=str(self.log_dir),
            cfg=self.config.to_dict(),
            env_cfg=self.env.cfg,
            num_envs=env.num_envs,
            is_distributed=self.is_multi_gpu,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=0,
            device=self.device,
        )

        self._init_config()

        self.current_learning_iteration = 0

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()

        # Observation manager system - history is defined per-module in module_dict
        assert self.env.observation_manager is not None
        self.algo_history_length_dict = {
            "actor_obs": self.env.observation_manager.cfg.groups[
                "actor_obs"
            ].history_length,
            "critic_obs": self.env.observation_manager.cfg.groups[
                "critic_obs"
            ].history_length,
        }

        self.num_act = self.env.robot_config.actions_dim

        self.actor_learning_rate = self.config.actor_learning_rate
        self.max_actor_learning_rate = self.config.max_actor_learning_rate or max(
            self.actor_learning_rate, 1e-2
        )
        self.min_actor_learning_rate = self.config.min_actor_learning_rate or min(
            self.actor_learning_rate, 1e-5
        )
        self.critic_learning_rate = self.config.critic_learning_rate
        self.max_critic_learning_rate = self.config.max_critic_learning_rate or max(
            self.critic_learning_rate, 1e-2
        )
        self.min_critic_learning_rate = self.config.min_critic_learning_rate or min(
            self.critic_learning_rate, 1e-5
        )

        # Observation related Config
        self.use_symmetry = self.config.use_symmetry
        self._init_obs_keys()

    def _init_obs_keys(self):
        self.actor_obs_keys = self.config.module_dict.actor.input_dim
        self.critic_obs_keys = self.config.module_dict.critic.input_dim

    def setup(self):
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up Storage")
        self._setup_storage()

        # Log curriculum synchronization status for multi-GPU training
        if self.is_multi_gpu:
            if self.has_curricula_enabled():
                logger.info(
                    f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs"
                )

    def _setup_models_and_optimizer(self):
        self.actor = setup_ppo_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=self.config.module_dict.critic,
            device=self.device,
            history_length=self.algo_history_length_dict,
        )

        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

        # Synchronize model weights across GPUs after initialization
        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()),
            lr=self.actor_learning_rate,
            weight_decay=self.config.actor_optimizer.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic.parameters()),
            lr=self.critic_learning_rate,
            weight_decay=self.config.critic_optimizer.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        """Compute total observation dimension for given observation keys."""
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), (
                f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            )
            # Note: algo_obs_dim_dict from observation_manager.get_obs_dims() already includes history
            obs_dim += key_dim
        return obs_dim

    def _get_zero_input(self):
        """
        Create a dummy (all-zero) input for the actor.

        During training, we cannot use the logic in `self.get_example_obs()`, since it resets environments mid-rollout.
        """
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        return torch.zeros(1, actor_obs_dim, device=self.device)

    def _setup_storage(self):
        self.storage = RolloutStorage(
            self.env.num_envs, self.config.num_steps_per_env, device=self.device
        )
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        print(f"Registering key: actor_obs with shape: {actor_obs_dim}")
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)

        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        print(f"Registering key: critic_obs with shape: {critic_obs_dim}")
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)

        # Register others based on Minibatch structure
        minibatch_keys = [
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ]
        for key, shape, dtype in minibatch_keys:
            self.storage.register(key, shape=shape, dtype=dtype)

    def _eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def _train_mode(self):
        self.actor.train()
        self.critic.train()

    def learn(self):
        self._train_mode()

        obs_dict = self.env.reset_all()

        # Initialize environments with different episode length buffers
        # Must happen AFTER reset_all() to avoid being overwritten by reset
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it

            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                self.export(
                    onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx")
                )

        if self.is_main_process:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"
                )
            )
            self.export(
                onnx_file_path=os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"
                )
            )

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                # Environment step
                actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs = torch.cat(
                    [obs_dict[k] for k in self.critic_obs_keys], dim=1
                )

                actions = self.actor.act({"actor_obs": actor_obs})
                values = self.critic.evaluate({"critic_obs": critic_obs}).detach()

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # Compute bootstrap value for timeouts
                final_rewards = torch.zeros_like(rewards)
                if infos["time_outs"].any():
                    final_critic_obs = torch.cat(
                        [infos["final_observations"][k] for k in self.critic_obs_keys],
                        dim=1,
                    )
                    final_values = self.critic.evaluate(
                        {"critic_obs": final_critic_obs}
                    ).detach()
                    final_rewards += self.config.gamma * torch.squeeze(
                        final_values * infos["time_outs"].unsqueeze(1).to(self.device),
                        1,
                    )

                # Add transition to storage
                self.storage.add(
                    actor_obs=actor_obs,
                    critic_obs=critic_obs,
                    actions=actions,
                    values=values,
                    actions_log_prob=self.actor.get_actions_log_prob(actions)
                    .detach()
                    .unsqueeze(1),
                    action_mean=self.actor.action_mean.detach(),
                    action_sigma=self.actor.action_std.detach(),
                    rewards=(rewards + final_rewards).view(-1, 1),
                    dones=dones.view(-1, 1),
                )

                # Reset actor and critic for completed envs
                self.actor.reset(dones)
                self.critic.reset(dones)

                if self.log_dir is not None:
                    # Update episode stats using logging helper
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

            # Return / Advantage computation
            last_critic_obs = torch.cat(
                [obs_dict[k] for k in self.critic_obs_keys], dim=1
            )
            last_values = (
                self.critic.evaluate({"critic_obs": last_critic_obs})
                .detach()
                .to(self.device)
            )
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"].to(self.device),
                self.storage["dones"].to(self.device),
                self.storage["rewards"].to(self.device),
            )

            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        return obs_dict

    def _compute_returns_and_advantages(self, last_values, values, dones, rewards):
        advantage = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * self.config.gamma * next_values
                - values[step]
            )
            advantage = (
                delta
                + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            )
            returns[step] = advantage + values[step]
        advantages = returns - values

        if self.is_multi_gpu:
            advantages = self._normalize_advantages_multi_gpu(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _training_step(self) -> dict[str, float]:
        generator = self.storage.mini_batch_generator(
            self.config.num_mini_batches, self.config.num_learning_epochs
        )

        minibatch: Minibatch
        loss_dict = {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}
        for minibatch in generator:
            loss_dict = self._update_algo_step(minibatch, loss_dict)

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for key in loss_dict:
            loss_dict[key] /= num_updates
        self.storage.clear()
        return loss_dict

    def _update_algo_step(self, minibatch: Minibatch, loss_dict: dict[str, float]):
        ppo_loss_dict = self._compute_ppo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        ppo_loss = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
        ppo_loss.backward()

        if self.is_multi_gpu:
            self._reduce_parameters()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        loss_dict["Value"] += ppo_loss_dict.pop("value_loss").item()
        loss_dict["Surrogate"] += ppo_loss_dict.pop("surrogate_loss").item()
        loss_dict["Entropy"] += ppo_loss_dict.pop("entropy_loss").item()
        loss_dict["KL"] += ppo_loss_dict.pop("kl_mean").item()
        for key, loss in ppo_loss_dict.items():
            if key not in loss_dict:
                loss_dict[key] = 0.0
            loss_value = loss.item() if torch.is_tensor(loss) else loss
            loss_dict[key] += loss_value
        return loss_dict

    def _compute_ppo_loss(self, minibatch: Minibatch):
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]

        # Symmetry augmentation
        original_batch_size = actions_batch.shape[0]
        if self.use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["actor_obs"],
                env=self.env,
                obs_list=self.actor_obs_keys,
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["critic_obs"],
                env=self.env,
                obs_list=self.critic_obs_keys,
            )
            actions_batch = self.symmetry_utils.augment_actions(
                actions=actions_batch,
            )
            num_aug = int(actor_obs.shape[0] / original_batch_size)
            old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)
        else:
            actor_obs = minibatch["actor_obs"]
            critic_obs = minibatch["critic_obs"]

        self.actor.act({"actor_obs": actor_obs})
        value_batch = self.critic.evaluate({"critic_obs": critic_obs})
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        mu_batch = self.actor.action_mean[:original_batch_size]
        sigma_batch = self.actor.action_std[:original_batch_size]
        entropy_batch = self.actor.entropy[:original_batch_size]

        if self.config.desired_kl is not None and self.config.schedule == "adaptive":
            # Compute the KL divergence between the old and new action distributions
            kl_mean = self._compute_kl_div(
                old_mu_batch, old_sigma_batch, mu_batch, sigma_batch
            )
            self._update_learning_rate(kl_mean)

        # Surrogate loss
        ratio = torch.exp(
            actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        )
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        if self.use_symmetry and (
            self.config.symmetry_actor_coef > 0.0
            or self.config.symmetry_critic_coef > 0.0
        ):
            mean_actions_batch = self.actor.act_inference(
                {"actor_obs": actor_obs.detach().clone()}
            )
            mean_actions_for_original_batch, mean_actions_for_symmetry_batch = (
                mean_actions_batch[:original_batch_size],
                mean_actions_batch[original_batch_size:],
            )
            mean_symmetry_actions_batch = self.symmetry_utils.augment_actions(
                actions=mean_actions_for_original_batch,
            )[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(
                mean_actions_for_symmetry_batch,
                mean_symmetry_actions_batch,
            )

            # Symmetry critic loss
            symmetry_critic_loss = F.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )
        else:
            symmetry_actor_loss = torch.tensor(0.0, device=self.device)
            symmetry_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = entropy_batch.mean()
        actor_loss = (
            surrogate_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )

        critic_loss = (
            self.config.value_loss_coef * value_loss
            + self.config.symmetry_critic_coef * symmetry_critic_loss
        )

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl_mean,
        }

    def _compute_kl_div(
        self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch
    ) -> torch.Tensor:
        with torch.inference_mode():
            # Compute the KL divergence between the old and new action distributions
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            kl = kl_divergence(old_dist, new_dist).sum(-1)
            kl_mean = torch.mean(kl)

            # Reduce the KL divergence across all GPUs
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
        return kl_mean

    def _update_learning_rate(self, kl_mean: torch.Tensor):
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rate = max(
                self.min_actor_learning_rate, self.actor_learning_rate / 1.5
            )
            self.critic_learning_rate = max(
                self.min_critic_learning_rate, self.critic_learning_rate / 1.5
            )
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_learning_rate = min(
                self.max_actor_learning_rate, self.actor_learning_rate * 1.5
            )
            self.critic_learning_rate = min(
                self.max_critic_learning_rate, self.critic_learning_rate * 1.5
            )

        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_learning_rate

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(loaded_dict["actor_model_state_dict"])
            self.critic.load_state_dict(loaded_dict["critic_model_state_dict"])
            if self.config.load_optimizer:
                self.actor_optimizer.load_state_dict(
                    loaded_dict["actor_optimizer_state_dict"]
                )
                self.critic_optimizer.load_state_dict(
                    loaded_dict["critic_optimizer_state_dict"]
                )
                self.actor_learning_rate = loaded_dict["actor_optimizer_state_dict"][
                    "param_groups"
                ][0]["lr"]
                self.critic_learning_rate = loaded_dict["critic_optimizer_state_dict"][
                    "param_groups"
                ][0]["lr"]
                logger.info("Optimizer loaded from checkpoint")
            self.current_learning_iteration = loaded_dict["iter"]
            self._restore_env_state(loaded_dict.get("env_state"))
            return loaded_dict.get("infos")
        return None

    def save(self, path, infos=None):
        checkpoint_dict = {
            "actor_model_state_dict": self.actor.state_dict(),
            "critic_model_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        checkpoint_dict.update(
            self._checkpoint_metadata(iteration=self.current_learning_iteration)
        )
        env_state = self._collect_env_state()
        if env_state:
            checkpoint_dict["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint_dict, path)
