from __future__ import annotations

import os
import pathlib
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from holosoma_agent.algorithms.base_algo import BaseAlgo
from holosoma_agent.algorithms.modules.augmentation_utils import SymmetryUtils
from holosoma_agent.algorithms.modules.data_utils import RolloutStorage
from holosoma_agent.algorithms.modules.logging_utils import LoggingHelper
from holosoma_agent.algorithms.modules.modules import LayerConfig, ModuleConfig
from holosoma_agent.algorithms.modules.ppo_modules import (
    PPOActor,
    PPOCritic,
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma_agent.configs.ppo_config import PPOConfig
from holosoma_agent.env.vec_env import VecEnv


class Minibatch(TypedDict):
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    old_actions_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def _default_actor_module_config(config: PPOConfig, num_actions: int) -> ModuleConfig:
    return ModuleConfig(
        type="MLP",
        input_dim=["actor_obs"],
        output_dim=["robot_action_dim"],  # resolved to num_actions in PPOActor
        layer_config=LayerConfig(
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
        ),
    )


def _default_critic_module_config(config: PPOConfig) -> ModuleConfig:
    return ModuleConfig(
        type="MLP",
        input_dim=["critic_obs"],
        output_dim=[1],
        layer_config=LayerConfig(
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
        ),
    )


class PPO(BaseAlgo):
    """Proximal Policy Optimization.

    Parameters
    ----------
    env : VecEnv
    config : PPOConfig
    device : str
    log_dir : str | pathlib.Path
    symmetry_utils : SymmetryUtils | None
        Optional symmetry augmentation helper.
    actor_module_config : ModuleConfig | None
        Override default actor architecture.
    critic_module_config : ModuleConfig | None
        Override default critic architecture.
    multi_gpu_cfg : dict | None
    """

    def __init__(
        self,
        env: VecEnv,
        config: PPOConfig,
        device: str,
        log_dir: str | pathlib.Path = "./logs",
        symmetry_utils: SymmetryUtils | None = None,
        actor_module_config: ModuleConfig | None = None,
        critic_module_config: ModuleConfig | None = None,
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = pathlib.Path(log_dir)
        self.symmetry_utils = symmetry_utils
        self._actor_module_config = actor_module_config
        self._critic_module_config = critic_module_config

        self.current_learning_iteration: int = 0
        self.actor: PPOActor | None = None
        self.critic: PPOCritic | None = None
        self.storage: RolloutStorage | None = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        logger.info("Setting up PPO")
        self._setup_models_and_optimizer()
        self._setup_storage()
        self._setup_logging()
        if self.is_multi_gpu:
            self._synchronize_model_parameters(self.actor, self.critic)
        logger.info("PPO setup complete")

    def _setup_models_and_optimizer(self):
        env = self.env
        config = self.config
        obs_dim_dict = env.get_obs_dims()
        history_length = {
            k: 1 for k in obs_dim_dict
        }  # default history 1 unless env says otherwise
        if hasattr(env, "history_lengths"):
            history_length.update(env.history_lengths)

        actor_mc = self._actor_module_config or _default_actor_module_config(
            config, env.num_actions
        )
        critic_mc = self._critic_module_config or _default_critic_module_config(config)

        self.actor = setup_ppo_actor_module(
            obs_dim_dict=obs_dim_dict,
            module_config=actor_mc,
            num_actions=env.num_actions,
            init_noise_std=config.init_noise_std,
            device=self.device,
            history_length=history_length,
        )
        self.critic = setup_ppo_critic_module(
            obs_dim_dict=obs_dim_dict,
            module_config=critic_mc,
            device=self.device,
            history_length=history_length,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate
        )

    def _setup_storage(self):
        env = self.env
        config = self.config
        self.storage = RolloutStorage(
            env.num_envs, config.num_steps_per_env, device=self.device
        )
        obs_dim_dict = env.get_obs_dims()
        actor_obs_dim = sum(obs_dim_dict[k] for k in env.actor_obs_keys)
        critic_obs_dim = sum(obs_dim_dict[k] for k in env.critic_obs_keys)

        self.storage.register("actor_obs", shape=(actor_obs_dim,))
        self.storage.register("critic_obs", shape=(critic_obs_dim,))
        self.storage.register("actions", shape=(env.num_actions,))
        self.storage.register("rewards", shape=())
        self.storage.register("dones", shape=())
        self.storage.register("values", shape=())
        self.storage.register("returns", shape=())
        self.storage.register("advantages", shape=())
        self.storage.register("old_actions_log_prob", shape=())

    def _setup_logging(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(self.log_dir))
        self.logging_helper = LoggingHelper(
            writer=writer,
            log_dir=self.log_dir,
            num_envs=self.env.num_envs * self.gpu_world_size,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            device=self.device,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

    # ── Main training loop ────────────────────────────────────────────────────

    def learn(self) -> None:
        env = self.env
        config = self.config

        # Randomise initial episode lengths to de-correlate resets
        if config.init_at_random_ep_len:
            env.episode_length_buf = torch.randint_like(
                env.episode_length_buf, high=int(env.max_episode_length)
            )

        obs_dict = env.get_observations()
        actor_obs = env.get_actor_obs(obs_dict)
        critic_obs = env.get_critic_obs(obs_dict)

        self.actor.train()
        self.critic.train()

        for it in range(
            self.current_learning_iteration, config.num_learning_iterations
        ):
            # ─ Collect rollout ───────────────────────────────────────────────
            with self.logging_helper.record_collection_time():
                with torch.inference_mode():
                    for _ in range(config.num_steps_per_env):
                        state_dict = {"actor_obs": actor_obs, "critic_obs": critic_obs}
                        actions = self.actor.act(state_dict)
                        values = self.critic.evaluate(state_dict).squeeze()
                        log_probs = self.actor.get_actions_log_prob(actions)

                        obs_dict_next, rewards, dones, infos = env.step(actions)
                        next_actor_obs = env.get_actor_obs(obs_dict_next)
                        next_critic_obs = env.get_critic_obs(obs_dict_next)

                        self.logging_helper.update_episode_stats(rewards, dones, infos)
                        time_outs = infos.get("time_outs", torch.zeros_like(dones))
                        dones_for_storage = dones * ~time_outs

                        self.storage.add(
                            actor_obs=actor_obs,
                            critic_obs=critic_obs,
                            actions=actions,
                            rewards=rewards,
                            dones=dones_for_storage,
                            values=values,
                            old_actions_log_prob=log_probs,
                        )
                        actor_obs, critic_obs = next_actor_obs, next_critic_obs

                    # Bootstrap value for GAE
                    final_values = self.critic.evaluate(
                        {"actor_obs": actor_obs, "critic_obs": critic_obs}
                    ).squeeze()
                    self._compute_returns(final_values, time_outs)

            # ─ Update ────────────────────────────────────────────────────────
            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            # ─ Adaptive LR (schedule driven by KL inside _update_algo_step) ─

            # ─ Logging & checkpoint ──────────────────────────────────────────
            if it % 1 == 0:
                self.logging_helper.post_epoch_logging(
                    it=it,
                    loss_dict=loss_dict,
                    extra_log_dicts={"Policy": {"std": self.actor.std.mean().item()}},
                )
            if it % config.save_interval == 0 and self.is_main_process:
                self.save(str(self.log_dir / f"model_{it:08d}.pt"))

        self.current_learning_iteration += config.num_learning_iterations

    def _compute_returns(
        self, last_values: torch.Tensor, time_outs: torch.Tensor
    ) -> None:
        config = self.config
        values = self.storage["values"]
        rewards = self.storage["rewards"]
        dones = self.storage["dones"]

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0.0
        for step in reversed(range(config.num_steps_per_env)):
            next_non_terminal = 1.0 - dones[step]
            if step == config.num_steps_per_env - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            delta = (
                rewards[step]
                + config.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_gae_lam = (
                delta + config.gamma * config.lam * next_non_terminal * last_gae_lam
            )
            advantages[step] = last_gae_lam
        returns = advantages + values
        self.storage["advantages"] = advantages
        self.storage["returns"] = returns

    def _training_step(self) -> dict[str, float]:
        config = self.config
        gen = self.storage.mini_batch_generator(
            config.num_mini_batches, config.num_learning_epochs
        )
        loss_dict: dict[str, float] = {
            "Value": 0.0,
            "Surrogate": 0.0,
            "Entropy": 0.0,
            "KL": 0.0,
            "SymmetryActor": 0.0,
            "SymmetryCritic": 0.0,
        }
        for batch in gen:
            loss_dict = self._update_algo_step(batch, loss_dict)
        num_updates = config.num_learning_epochs * config.num_mini_batches
        for k in loss_dict:
            loss_dict[k] /= num_updates
        self.storage.clear()
        return loss_dict

    def _update_algo_step(
        self, batch: dict[str, torch.Tensor], loss_dict: dict[str, float]
    ) -> dict[str, float]:
        config = self.config
        use_symmetry = (
            config.use_symmetry
            and self.symmetry_utils is not None
            and config.symmetry_actor_coef + config.symmetry_critic_coef > 0.0
        )

        original_batch_size = batch["actor_obs"].shape[0]

        # ── Symmetry data augmentation ────────────────────────────────────────
        if use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=batch["actor_obs"], obs_list=self.env.actor_obs_keys
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=batch["critic_obs"], obs_list=self.env.critic_obs_keys
            )
            actions_batch = self.symmetry_utils.augment_actions(
                actions=batch["actions"]
            )
            num_aug = actor_obs.shape[0] // original_batch_size
            old_log_prob = (
                batch["old_actions_log_prob"].unsqueeze(-1).repeat(num_aug, 1)
            )
            advantages = batch["advantages"].unsqueeze(-1).repeat(num_aug, 1)
            returns = batch["returns"].unsqueeze(-1).repeat(num_aug, 1)
        else:
            actor_obs = batch["actor_obs"]
            critic_obs = batch["critic_obs"]
            actions_batch = batch["actions"]
            old_log_prob = batch["old_actions_log_prob"].unsqueeze(-1)
            advantages = batch["advantages"].unsqueeze(-1)
            returns = batch["returns"].unsqueeze(-1)

        # ── Forward passes ────────────────────────────────────────────────────
        self.actor.update_distribution(actor_obs)
        actions_log_prob = self.actor.get_actions_log_prob(actions_batch)
        entropy = self.actor.entropy[:original_batch_size]  # only original

        value = self.critic.evaluate(
            {"actor_obs": actor_obs, "critic_obs": critic_obs}
        ).squeeze()

        # ── Advantage normalisation ───────────────────────────────────────────
        advantages = advantages.squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Surrogate loss ────────────────────────────────────────────────────
        ratio = torch.exp(actions_log_prob - old_log_prob.squeeze(-1))
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - config.clip_param, 1 + config.clip_param)
            * advantages
        )
        surrogate_loss = torch.max(-surr1, -surr2).mean()

        # ── Value loss (clipped) ──────────────────────────────────────────────
        returns = returns.squeeze(-1)
        target_values = batch["values"].repeat(num_aug if use_symmetry else 1)
        value_clipped = target_values + (value - target_values).clamp(
            -config.clip_param, config.clip_param
        )
        value_losses = (value - returns).pow(2)
        value_losses_clipped = (value_clipped - returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        # ── Symmetry losses ───────────────────────────────────────────────────
        if use_symmetry:
            # Actor symmetry: compare mean actions from original vs mirror half
            mean_actions = self.actor.act_inference(
                {"actor_obs": actor_obs.detach().clone()}
            )
            mean_orig = mean_actions[:original_batch_size]
            mean_mirror = mean_actions[original_batch_size:]
            expected_mirror = self.symmetry_utils.augment_actions(mean_orig)[
                original_batch_size:
            ]
            sym_actor_loss = F.mse_loss(mean_mirror, expected_mirror)
            # Critic symmetry: original and mirrored values should be equal
            sym_critic_loss = F.mse_loss(
                value[:original_batch_size],
                value[original_batch_size:],
            )
        else:
            sym_actor_loss = torch.tensor(0.0, device=self.device)
            sym_critic_loss = torch.tensor(0.0, device=self.device)

        entropy_loss = entropy.mean()

        # ── Adaptive LR ───────────────────────────────────────────────────────
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        if config.schedule == "adaptive" and config.desired_kl is not None:
            if approx_kl > config.desired_kl * 2.0:
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = max(1e-9, pg["lr"] / 1.5)
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = max(1e-9, pg["lr"] / 1.5)
            elif approx_kl < config.desired_kl / 2.0:
                for pg in self.actor_optimizer.param_groups:
                    pg["lr"] = min(1e-2, pg["lr"] * 1.5)
                for pg in self.critic_optimizer.param_groups:
                    pg["lr"] = min(1e-2, pg["lr"] * 1.5)

        # ── Actor update ──────────────────────────────────────────────────────
        actor_loss = (
            surrogate_loss
            - config.entropy_coef * entropy_loss
            + config.symmetry_actor_coef * sym_actor_loss
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        if config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.actor)
        self.actor_optimizer.step()

        # ── Critic update ─────────────────────────────────────────────────────
        critic_loss = (
            config.value_loss_coef * value_loss
            + config.symmetry_critic_coef * sym_critic_loss
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
        if self.is_multi_gpu:
            self._all_reduce_model_grads(self.critic)
        self.critic_optimizer.step()

        loss_dict["Value"] += value_loss.item()
        loss_dict["Surrogate"] += surrogate_loss.item()
        loss_dict["Entropy"] += entropy_loss.item()
        loss_dict["KL"] += approx_kl
        loss_dict["SymmetryActor"] += sym_actor_loss.item()
        loss_dict["SymmetryCritic"] += sym_critic_loss.item()
        return loss_dict

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "iteration": self.current_learning_iteration,
            },
            path,
        )
        logger.info(f"Saved PPO checkpoint to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if self.config.load_optimizer:
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )
        self.current_learning_iteration = checkpoint.get("iteration", 0)
        logger.info(
            f"Loaded PPO checkpoint from {path} (iter {self.current_learning_iteration})"
        )
