from __future__ import annotations

import copy
import time
import itertools
import math
import os
import pathlib
from contextlib import contextmanager
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tensordict import TensorDict
from torch.amp import GradScaler, autocast

from holosoma_agent.algorithms.base_algo import BaseAlgo
from holosoma_agent.algorithms.fast_sac.networks import (
    Actor,
    ActorEncoder,
    CNNActor,
    Critic,
    CriticEncoder,
    CNNCritic,
)
from holosoma_agent.algorithms.fast_sac.fast_sac_utils import (
    EmpiricalNormalization,
    SimpleReplayBuffer,
    save_params,
)
from holosoma_agent.utils.symmetry_utils import SymmetryUtils
from holosoma_agent.utils.logger import Logger
from holosoma_agent.configs.fast_sac_config import FastSACConfig
from holosoma_agent.env.fast_sac_env import FastSACVecEnv


class FastSACAgent(BaseAlgo):
    """FastSAC off-policy RL agent.

    Parameters
    ----------
    env : FastSACVecEnv
    config : FastSACConfig
    device : str
    log_dir : str | pathlib.Path
        Directory for TensorBoard logs and checkpoints.
    symmetry_utils : SymmetryUtils | None
        Optional symmetry augmentation helper.  Pass ``None`` to disable (even if
        ``config.use_symmetry is True``).
    multi_gpu_cfg : dict | None
    """

    def __init__(
        self,
        env: FastSACVecEnv,
        config: FastSACConfig,
        device: str | torch.device = "cpu",
        log_dir: str | pathlib.Path = "./logs",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(env, config, device, multi_gpu_cfg)
        self.log_dir = log_dir
        self.global_step: int = 0

        # Will be populated by setup()
        self.actor: Actor | CNNActor | None = None
        self.qnet: Critic | CNNCritic | None = None
        self.qnet_target: Critic | CNNCritic | None = None

    """
    network and optimizer setup
    """

    def setup(self) -> None:
        logger.info("Setting up FastSACAgent")
        env = self.env

        # I think this is smarter than defining everything in config like holosoma
        obs_space = env.observation_space  # gymnasium.spaces.dict.Dict
        actor_obs_space_shape = [
            obs_space[k].shape[-1] for k in self.config.actor_obs_keys
        ]
        actor_obs_dim = sum(actor_obs_space_shape)

        critic_obs_space_shape = [
            obs_space[k].shape[-1] for k in self.config.critic_obs_keys
        ]
        critic_obs_dim = sum(critic_obs_space_shape)

        action_scale = env._compute_action_boundaries()

        # Observation normalizers
        if self.config.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(actor_obs_dim, self.device).to(
                self.device
            )
            self.critic_obs_normalizer = EmpiricalNormalization(
                critic_obs_dim, self.device
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        # Select actor/critic class
        if self.config.module_type == "MLP":
            actor_cls, critic_cls = Actor, Critic
        elif self.config.module_type == "MLPEncoder":
            actor_cls, critic_cls = ActorEncoder, CriticEncoder
        elif self.config.module_type == "CNNEncoder":
            actor_cls, critic_cls = CNNActor, CNNCritic
        else:
            logger.warning(f"Unknown module type: {self.config.module_type}, using MLP")
            actor_cls, critic_cls = Actor, Critic

        self.actor = actor_cls(
            obs_dim=actor_obs_dim,
            action_dim=env.num_actions,
            hidden_dim=self.config.actor_hidden_dim,
            use_tanh=self.config.use_tanh,
            use_layer_norm=self.config.use_layer_norm,
            log_std_max=self.config.log_std_max,
            log_std_min=self.config.log_std_min,
            device=self.device,
            action_scale=action_scale,
        ).to(self.device)

        self.qnet = critic_cls(
            obs_dim=critic_obs_dim,
            action_dim=env.num_actions,
            hidden_dim=self.config.critic_hidden_dim,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            use_layer_norm=self.config.use_layer_norm,
            num_q_networks=self.config.num_q_networks,
            device=self.device,
        ).to(self.device)

        print(self.actor)
        print(self.qnet)

        # Temperature
        self.log_alpha = torch.tensor(
            [math.log(self.config.alpha_init)], requires_grad=True, device=self.device
        )
        self.policy = self.actor.explore

        self.qnet_target = critic_cls(
            obs_dim=critic_obs_dim,
            action_dim=env.num_actions,
            hidden_dim=self.config.critic_hidden_dim,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            use_layer_norm=self.config.use_layer_norm,
            num_q_networks=self.config.num_q_networks,
            device=self.device,
        ).to(self.device)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()),
            lr=self.config.actor_learning_rate,
            weight_decay=self.config.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.q_optimizer = torch.optim.AdamW(
            list(self.qnet.parameters()),
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )

        self.target_entropy = -env.num_actions * self.config.target_entropy_ratio
        self.alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha],
            lr=self.config.alpha_learning_rate,
            fused=True,
            betas=(0.9, 0.95),
        )

        # AMP scaler
        if self.config.amp:
            dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
            self.amp_dtype = dtype_map.get(self.config.amp_dtype, torch.bfloat16)
            self.scaler = GradScaler() if self.amp_dtype == torch.float16 else None
        else:
            self.amp_dtype = None
            self.scaler = None
        self.scaler = GradScaler(enabled=self.config.amp)

        logger.info("FastSAC network setup complete")

    def setup_learning(self):
        env = self.env

        # I think this is smarter than defining everything in config like holosoma
        obs_space = env.observation_space  # gymnasium.spaces.dict.Dict
        actor_obs_space_shape = [
            obs_space[k].shape[-1] for k in self.config.actor_obs_keys
        ]
        actor_obs_dim = sum(actor_obs_space_shape)

        critic_obs_space_shape = [
            obs_space[k].shape[-1] for k in self.config.critic_obs_keys
        ]
        critic_obs_dim = sum(critic_obs_space_shape)

        # Replay buffer
        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=self.config.buffer_size,
            n_obs=actor_obs_dim,
            n_act=env.num_actions,
            n_critic_obs=critic_obs_dim,
            n_steps=self.config.num_steps,
            gamma=self.config.gamma,
            device=self.device,
        )

        # Logging
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

        if self.config.use_symmetry:
            self.symmetry_utils = SymmetryUtils(
                self.env, self.config.symmetry_config, self.device
            )

        if self.is_multi_gpu:
            self._synchronize_model_parameters(self.actor, self.qnet)

    """
    multi-gpu training utils
    """

    @contextmanager
    def amp_context(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        """Synchronize actor, qnet, and log_alpha parameters across all GPUs."""
        # Broadcast actor weights from rank 0 to all other ranks
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast qnet weights from rank 0 to all other ranks
        for param in self.qnet.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast log_alpha parameter from rank 0 to all other ranks
        torch.distributed.broadcast(self.log_alpha.data, src=0)

        # Load qnet_target weights from synced qnet
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Batches and all-reduces gradients across GPUs to reduce NCCL call count.

        This flattens all existing parameter gradients into a single contiguous
        tensor, performs one all_reduce, averages by world size, and then
        scatters the reduced values back into the original gradient tensors.
        """
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    """
    update functions
    """

    def _update_critic(
        self, data: TensorDict
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        args = self.config

        scaler = self.scaler
        actor = self.actor
        qnet = self.qnet
        qnet_target = self.qnet_target
        q_optimizer = self.q_optimizer
        alpha_optimizer = self.alpha_optimizer

        with self.amp_context():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = (
                    actor.get_actions_and_log_probs(next_observations)
                )
                discount = args.gamma ** data["next"]["effective_n_steps"]

                target_distributions = qnet_target.projection(
                    next_critic_observations,
                    next_state_actions,
                    rewards
                    - discount
                    * bootstrap
                    * self.log_alpha.exp()
                    * next_state_log_probs,
                    bootstrap,
                    discount,
                )
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(qnet)

        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self.amp_context():
                alpha_loss = (
                    -self.log_alpha.exp()
                    * (next_state_log_probs.detach() + self.target_entropy)
                ).mean()

            scaler.scale(alpha_loss).backward()

            if self.is_multi_gpu:
                if self.log_alpha.grad is not None:
                    torch.distributed.all_reduce(
                        self.log_alpha.grad.data, op=torch.distributed.ReduceOp.SUM
                    )
                    self.log_alpha.grad.data.copy_(
                        self.log_alpha.grad.data / self.gpu_world_size
                    )

            scaler.unscale_(alpha_optimizer)

            scaler.step(alpha_optimizer)
            scaler.update()

        return (
            rewards.mean(),
            critic_grad_norm.detach(),
            qf_loss.detach(),
            target_value_max.detach(),
            target_value_min.detach(),
            alpha_loss.detach(),
        )

    def _update_policy(
        self, data: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = self.actor
        qnet = self.qnet
        actor_optimizer = self.actor_optimizer
        scaler = self.scaler
        args = self.config

        with self.amp_context():
            critic_observations = data["critic_observations"]

            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            # For logging, this is a bit wasteful though, but could be useful
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                # Compute policy entropy (negative log probability)
                policy_entropy = -log_probs.mean()

            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()

        if self.is_multi_gpu:
            self._all_reduce_model_grads(actor)

        scaler.unscale_(actor_optimizer)

        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            policy_entropy.detach(),
            action_std.detach(),
        )

    def _sample_and_prepare_batches(
        self, batch_size: int, num_updates: int, normalize_obs, normalize_critic_obs
    ) -> list[TensorDict]:
        """
        Sample a large batch once and split it into smaller batches for each update.
        This reduces sampling overhead by `num_updates` and normalization overhead by `num_updates`.
        """
        # Sample a large batch (batch_size * num_updates)
        large_batch_size = batch_size * num_updates
        large_data = self.rb.sample(large_batch_size)
        samples_per_update = batch_size * self.env.num_envs

        if self.config.use_symmetry:
            samples_per_update *= 2

            augmented_large_data: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
                "next": {}
            }

            augmented_large_data["observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["observations"],
                    obs_list=self.config.actor_obs_keys,
                )
            )
            augmented_large_data["actions"] = self.symmetry_utils.augment_actions(
                actions=large_data["actions"]
            )
            assert isinstance(augmented_large_data["next"], dict)
            augmented_large_data["next"]["observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["next"]["observations"],
                    obs_list=self.config.actor_obs_keys,
                )
            )
            augmented_large_data["critic_observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["critic_observations"],
                    obs_list=self.config.critic_obs_keys,
                )
            )
            augmented_large_data["next"]["critic_observations"] = (
                self.symmetry_utils.augment_observations(
                    obs=large_data["next"]["critic_observations"],
                    obs_list=self.config.critic_obs_keys,
                )
            )

            # Calculate augmentation factor and repeat non-augmented data
            observations_tensor = augmented_large_data["observations"]
            assert isinstance(observations_tensor, torch.Tensor), (
                "observations should be a Tensor after data augmentation"
            )
            num_aug = int(
                observations_tensor.shape[0] / large_data["next"]["rewards"].shape[0]
            )
            augmented_large_data["next"]["rewards"] = large_data["next"][
                "rewards"
            ].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["dones"] = large_data["next"]["dones"].repeat(
                num_aug
            )  # type: ignore[index]
            augmented_large_data["next"]["truncations"] = large_data["next"][
                "truncations"
            ].repeat(num_aug)  # type: ignore[index]
            augmented_large_data["next"]["effective_n_steps"] = large_data["next"][
                "effective_n_steps"
            ].repeat(num_aug)  # type: ignore[index]

            # Override large_data
            large_data = augmented_large_data

        # Normalize all data once
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(
            large_data["next"]["observations"]
        )
        large_data["critic_observations"] = normalize_critic_obs(
            large_data["critic_observations"]
        )
        large_data["next"]["critic_observations"] = normalize_critic_obs(
            large_data["next"]["critic_observations"]
        )

        # Split into smaller batches
        prepared_batches = []

        for i in range(num_updates):
            start_idx = i * samples_per_update
            end_idx = (i + 1) * samples_per_update

            # Create a slice of the large batch
            batch_data = TensorDict(
                {
                    "observations": large_data["observations"][start_idx:end_idx],
                    "actions": large_data["actions"][start_idx:end_idx],
                    "next": {
                        "rewards": large_data["next"]["rewards"][start_idx:end_idx],
                        "dones": large_data["next"]["dones"][start_idx:end_idx],
                        "truncations": large_data["next"]["truncations"][
                            start_idx:end_idx
                        ],
                        "observations": large_data["next"]["observations"][
                            start_idx:end_idx
                        ],
                        "effective_n_steps": large_data["next"]["effective_n_steps"][
                            start_idx:end_idx
                        ],
                    },
                    "critic_observations": large_data["critic_observations"][
                        start_idx:end_idx
                    ],
                },
                batch_size=samples_per_update,
            )
            batch_data["next"]["critic_observations"] = large_data["next"][
                "critic_observations"
            ][start_idx:end_idx]

            prepared_batches.append(batch_data)

        return prepared_batches

    def learn(self) -> None:
        args = self.config
        device = self.device

        self.setup_learning()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        if args.compile:
            update_critic = torch.compile(self._update_critic)
            update_policy = torch.compile(self._update_policy)
            policy = torch.compile(self.policy)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            update_critic = self._update_critic
            update_policy = self._update_policy
            policy = self.policy
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward
        qnet = self.qnet
        qnet_target = self.qnet_target
        env = self.env
        rb = self.rb

        obs_dict, _ = env.reset()
        obs = torch.cat([obs_dict[key] for key in args.actor_obs_keys], dim=-1)
        critic_obs = torch.cat([obs_dict[key] for key in args.critic_obs_keys], dim=-1)

        dones = None
        # Initialize metrics that might not be updated every step
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        # pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)

        # while self.global_step <= args.num_learning_iterations:
        for it in range(
            self.global_step, self.global_step + args.num_learning_iterations
        ):
            self.global_step = it
            # Synchronize curriculum metrics across GPUs before rollout
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            # Rollout and add it to replay buffer
            start_collect_time = time.time()
            with torch.no_grad(), self.amp_context():
                norm_obs = normalize_obs(obs, update=False)
                actions = policy(obs=norm_obs, dones=dones)

            next_obs, rewards, dones, infos = env.step(actions.float())
            truncations = infos["time_outs"]

            next_actor_obs = infos["observations"]["actor"]
            next_critic_obs = infos["observations"]["critic"]

            # Compute 'true' next_obs and next_critic_obs for saving
            true_next_obs = torch.where(
                truncations[:, None] > 0,
                infos["observations"]["final"]["actor_obs"],
                next_actor_obs,
            )
            true_next_critic_obs = torch.where(
                truncations[:, None] > 0,
                infos["observations"]["final"]["critic_obs"],
                next_critic_obs,
            )
            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(
                        actions, device=device, dtype=torch.float
                    ),
                    "next": {
                        "observations": true_next_obs,
                        "rewards": torch.as_tensor(
                            rewards, device=device, dtype=torch.float
                        ),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(env.num_envs,),
                device=device,
            )
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs

            obs = next_actor_obs
            critic_obs = next_critic_obs

            rb.extend(transition)

            collect_time = time.time() - start_collect_time

            # log rollout
            self.logger.process_env_step(rewards, dones, infos)

            # Start learning after collecting enough samples
            # NOTE: args.batch_size is the global batch size
            batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
            if self.global_step > args.learning_starts:
                # Use batched sampling: sample once, normalize once, split into updates
                prepared_batches = self._sample_and_prepare_batches(
                    batch_size,
                    args.num_updates,
                    normalize_obs,
                    normalize_critic_obs,
                )
                for i, data in enumerate(prepared_batches):
                    # Data is already normalized, just run the updates
                    (
                        buffer_rewards,
                        critic_grad_norm,
                        qf_loss,
                        qf_max,
                        qf_min,
                        alpha_loss,
                    ) = update_critic(data)
                    if args.num_updates > 1:
                        if i % args.policy_frequency == 1:
                            (
                                actor_grad_norm,
                                actor_loss,
                                policy_entropy,
                                action_std,
                            ) = update_policy(data)
                    elif self.global_step % args.policy_frequency == 0:
                        actor_grad_norm, actor_loss, policy_entropy, action_std = (
                            update_policy(data)
                        )

                    # update target networks with polyak averaging
                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                    # grab loss metrics
                    loss_dict = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "alpha_loss": alpha_loss,
                        "alpha_value": self.log_alpha.exp().detach().mean(),
                        "policy_entropy": policy_entropy,
                    }

                    learning_rate_dict = {
                        "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
                        "critic_lr": self.q_optimizer.param_groups[0]["lr"],
                        "alpha_lr": self.alpha_optimizer.param_groups[0]["lr"],
                    }

                learn_time = time.time() - start_collect_time

                if self.global_step % args.logging_interval == 0:
                    self.logger.log(
                        it=self.global_step,
                        start_it=0,  # TODO: support resuming
                        total_it=args.num_learning_iterations,
                        collect_time=collect_time,
                        learn_time=learn_time,
                        loss_dict=loss_dict,
                        learning_rate_dict=learning_rate_dict,
                        action_std=action_std,
                    )

                if (args.save_interval > 0) and (
                    self.global_step % args.save_interval == 0
                ):
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        self.save(
                            os.path.join(
                                self.log_dir,
                                "models",
                                f"model_{self.global_step}.pt",
                            )
                        )

        if self.is_main_process:
            self.save(
                os.path.join(self.log_dir, "models", f"model_{self.global_step}.pt")
            )

    def save(self, path: str) -> None:  # type: ignore[override]
        save_params(
            self.global_step,
            self.actor,
            self.qnet,
            self.qnet_target,
            self.log_alpha,
            self.obs_normalizer,
            self.critic_obs_normalizer,
            self.actor_optimizer,
            self.q_optimizer,
            self.alpha_optimizer,
            self.scaler,
            self.config,
            path,
        )
        self.logger.save_model(path, self.global_step)

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            ckpt_path, map_location=self.device, weights_only=False
        )

        # Handle DDP-wrapped models
        actor_state_dict = torch_checkpoint["actor_state_dict"]
        qnet_state_dict = torch_checkpoint["qnet_state_dict"]

        self.actor.load_state_dict(actor_state_dict)
        self.qnet.load_state_dict(qnet_state_dict)

        self.obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        self.critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        self.qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        self.log_alpha.data.copy_(torch_checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(
            torch_checkpoint["actor_optimizer_state_dict"]
        )
        self.q_optimizer.load_state_dict(torch_checkpoint["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(
            torch_checkpoint["alpha_optimizer_state_dict"]
        )
        self.scaler.load_state_dict(torch_checkpoint["grad_scaler_state_dict"])
        self.global_step = torch_checkpoint["global_step"]

    def get_inference_policy(
        self, device: str | None = None
    ) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        # Use the underlying module for inference
        policy = self.actor.to(device)
        obs_normalizer = self.obs_normalizer.to(device)
        policy.eval()
        obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            if self.obs_normalizer:
                normalized_obs = obs_normalizer(obs["policy"], update=False)
            else:
                normalized_obs = obs["policy"]
            # Actions are already scaled by the actor
            action, _, _ = policy(normalized_obs)
            return action

        return policy_fn

    @property
    def actor_onnx_wrapper(self) -> nn.Module:
        # Use the underlying module for JIT/ONNX export
        actor = copy.deepcopy(self.actor).to("cpu")
        obs_normalizer = copy.deepcopy(self.obs_normalizer).to("cpu")
        actor.action_scale = actor.action_scale.to("cpu")

        class ActorWrapper(nn.Module):
            def __init__(self, actor, obs_normalizer):
                super().__init__()
                self.actor = actor
                self.obs_normalizer = obs_normalizer

            def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
                if self.obs_normalizer is not None:
                    normalized_obs = self.obs_normalizer(actor_obs, update=False)
                else:
                    normalized_obs = actor_obs
                # Actions are already scaled by the actor
                action, _, _ = self.actor(normalized_obs)
                return action

        return ActorWrapper(actor, obs_normalizer if self.obs_normalizer else None)
