"""FastSAC neural network modules.

Direct port of holosoma's fast_sac.py — pure PyTorch, no holosoma dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Actor ─────────────────────────────────────────────────────────────────────


class Actor(nn.Module):
    """Gaussian actor for FastSAC (MLP backbone)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

        self.use_tanh = use_tanh
        self.use_layer_norm = use_layer_norm

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if action_scale is not None:
            self.action_scale = action_scale.to(self.device)
        else:
            self.action_scale = torch.ones(action_dim, device=self.device)

        self.setup_network()

    def setup_network(self) -> None:
        # Build MLP dynamically from self.hidden_dim list.
        # Each element is one hidden layer; consecutive pairs define a Linear layer.
        dims = [self.obs_dim] + self.hidden_dim  # e.g. [obs, 256, 128, 64]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim, device=self.device))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim, device=self.device))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

        last_dim = self.hidden_dim[-1]
        self.fc_mean = nn.Linear(last_dim, self.action_dim, device=self.device)
        self.fc_log_std = nn.Linear(last_dim, self.action_dim, device=self.device)
        nn.init.constant_(self.fc_mean.weight, 0.0)
        nn.init.constant_(self.fc_mean.bias, 0.0)
        nn.init.constant_(self.fc_log_std.weight, 0.0)
        nn.init.constant_(self.fc_log_std.bias, 0.0)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_log_probs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            # Apply tanh to get bounded actions in [-1, 1]
            tanh_action = torch.tanh(raw_action)
            # Scale and bias to get final actions
            action = tanh_action * self.action_scale

            # Compute log probability with proper Jacobian correction
            log_prob = dist.log_prob(raw_action)
            # Jacobian correction for tanh transformation
            log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
            # Jacobian correction for scaling transformation
            log_prob -= torch.log(self.action_scale + 1e-6)
            # TODO: check diff from rl-games SAC
        else:
            # Non-tanh case
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        log_prob = log_prob.sum(1)
        return action, log_prob

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        dones: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                tanh_mean = torch.tanh(mean)
                return tanh_mean * self.action_scale
            return mean

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale
        else:
            action = raw_action

        return action


# TODO: Implement MLP encoder later
class ActorEncoder(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim,
            log_std_max,
            log_std_min,
            use_tanh,
            use_layer_norm,
            device,
            action_scale,
        )


# TODO: Implement CNN encoder later
class CNNActor(Actor):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim,
            log_std_max,
            log_std_min,
            use_tanh,
            use_layer_norm,
            device,
            action_scale,
        )


# class CNNActor(Actor):
#     """FastSAC actor with a CNN encoder for image observations."""

#     def __init__(
#         self,
#         obs_indices: dict[str, slice | list[int]],
#         obs_keys: list[str],
#         obs_dim: int,
#         action_dim: int,
#         hidden_dim: int = 512,
#         use_layer_norm: bool = True,
#         log_std_max: float = 0.0,
#         log_std_min: float = -5.0,
#         action_boundaries: torch.Tensor | None = None,
#         encoder_obs_key: str = "perception_obs",
#         encoder_obs_shape: tuple[int, int, int] = (1, 13, 9),
#         encoder_hidden_channels: tuple[int, ...] = (32, 64),
#         encoder_kernel_sizes: tuple[int, ...] = (3, 3),
#         encoder_strides: tuple[int, ...] = (1, 1),
#     ):
#         super().__init__(
#             obs_dim,
#             action_dim,
#             hidden_dim,
#             log_std_max,
#             log_std_min,
#             use_layer_norm,
#             action_boundaries,
#         )
#         self.encoder_obs_key = encoder_obs_key
#         in_ch, in_h, in_w = encoder_obs_shape
#         paddings = tuple(k // 2 for k in encoder_kernel_sizes)
#         self.cnn_encoder = nn.Sequential(
#             *[
#                 layer
#                 for i, (out_ch, k, s, p) in enumerate(
#                     zip(
#                         encoder_hidden_channels,
#                         encoder_kernel_sizes,
#                         encoder_strides,
#                         paddings,
#                     )
#                 )
#                 for layer in [
#                     nn.Conv2d(
#                         in_ch if i == 0 else encoder_hidden_channels[i - 1],
#                         out_ch,
#                         k,
#                         s,
#                         p,
#                     ),
#                     nn.ReLU(),
#                 ]
#             ]
#         )
#         cnn_out_dim = calculate_cnn_output_dim(
#             in_ch,
#             in_h,
#             in_w,
#             encoder_kernel_sizes,
#             encoder_strides,
#             paddings,
#             encoder_hidden_channels,
#         )
#         self.cnn_linear = nn.Linear(cnn_out_dim, hidden_dim)
#         self.encoder_obs_shape = encoder_obs_shape

#     def _encode(self, all_obs: torch.Tensor) -> torch.Tensor:
#         enc_obs = all_obs[..., self.obs_indices[self.encoder_obs_key]]
#         b = enc_obs.shape[0]
#         enc_obs = enc_obs.view(b, *self.encoder_obs_shape)
#         enc_feat = self.cnn_encoder(enc_obs).view(b, -1)
#         return self.cnn_linear(enc_feat)

#     def forward(self, all_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         obs = self._get_obs(all_obs)
#         enc = self._encode(all_obs)
#         x = obs + enc # TODO: use cat
#         x = self.net(x)
#         mean = self.fc_mean(x)
#         log_std = torch.tanh(self.fc_log_std(x))
#         log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
#             log_std + 1
#         )
#         return mean, log_std


# ── Critic (Distributional) ───────────────────────────────────────────────────


class DistributionalQNetwork(nn.Module):
    """C51-style distributional Q-network for a single ensemble member."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build MLP dynamically from hidden_dim list.
        # Input is (obs + action), hidden layers from list, output is num_atoms.
        dims = [obs_dim + action_dim] + list(hidden_dim)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim, device=self.device))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim, device=self.device))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-1], num_atoms, device=self.device))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = (
            rewards.unsqueeze(1)
            + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        )
        target_z = target_z.clamp(self.v_min, self.v_max)
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=device
            )
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(
            0, lower_indices, (next_dist * (upper.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, upper_indices, (next_dist * (b - lower.float())).view(-1)
        )
        return proj_dist


class Critic(nn.Module):
    """Ensemble of distributional Q-networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_layer_norm = use_layer_norm
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = num_q_networks
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_network()
        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=self.device)
        )

    def setup_network(self) -> None:
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    num_atoms=self.num_atoms,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    # NOTE: holosoma code defines proces_obs method that handles multi-sensor inputs
    # and keep forward method not override, which makes no sense
    # I would directly modify forward method for ease of implementation
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        outputs = [qnet(obs, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.q_support, dim=-1)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        projections = [
            qnet.projection(
                obs,
                actions,
                rewards,
                bootstrap,
                discount,
                self.q_support,
                self.q_support.device,
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)


# TODO: Implement MLP encoder later
class CriticEncoder(Critic):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim,
            num_atoms,
            v_min,
            v_max,
            use_layer_norm,
            num_q_networks,
            device,
        )


# TODO: Implement CNN later
class CNNCritic(Critic):
    """Ensemble of distributional Q-networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        device: torch.device | None = None,
    ):
        super().__init__(
            obs_dim,
            action_dim,
            hidden_dim,
            num_atoms,
            v_min,
            v_max,
            use_layer_norm,
            num_q_networks,
            device,
        )


# class CNNCritic(Critic):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def setup_qnetworks(self) -> None:
#         """Setup CNN encoder and Q-networks with correct input dimensions."""
#         if self.encoder_obs_shape is None:
#             raise ValueError("encoder_obs_shape must be provided for CNNCritic")

#         # Create the CNN encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(
#                 self.encoder_obs_shape[0],
#                 16,
#                 kernel_size=4,
#                 stride=2,
#                 padding=1,
#                 device=self.device,
#             ),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, device=self.device),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Calculate CNN output dimension using mathematical calculation
#         cnn_output_dim = calculate_cnn_output_dim(self.encoder_obs_shape)

#         # Calculate total input dimension: CNN features + state observations
#         state_obs_dim = sum(
#             self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys
#         )
#         total_obs_dim = cnn_output_dim + state_obs_dim

#         # Setup Q-networks with the correct observation dimension
#         self._setup_qnetworks_with_obs_dim(total_obs_dim)

#     def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
#         if self.encoder_obs_key is None or self.encoder_obs_shape is None:
#             raise ValueError(
#                 "encoder_obs_key and encoder_obs_shape must be provided for CNNCritic"
#             )

#         encoder_obs = torch.cat(
#             [
#                 obs[
#                     ...,
#                     self.obs_indices[self.encoder_obs_key]["start"] : self.obs_indices[
#                         self.encoder_obs_key
#                     ]["end"],
#                 ]
#             ],
#             -1,
#         )
#         encoder_obs = encoder_obs.view(encoder_obs.shape[0], *self.encoder_obs_shape)
#         encoder_x = self.encoder(encoder_obs)

#         # Handle state observations. This could include encoder obs if the user wants
#         state_x = torch.cat(
#             [
#                 obs[
#                     ...,
#                     self.obs_indices[obs_key]["start"] : self.obs_indices[obs_key][
#                         "end"
#                     ],
#                 ]
#                 for obs_key in self.obs_keys
#             ],
#             -1,
#         )

#         # Concatenate CNN features with state observations
#         return torch.cat([encoder_x, state_x], -1)


# def calculate_cnn_output_dim(input_shape: tuple[int, int, int]) -> int:
#     """
#     Calculate CNN output dimension for the fixed CNN architecture.

#     The CNN has the following architecture:
#     1. Conv2d(channels, 16, kernel_size=4, stride=2, padding=1)
#     2. Conv2d(16, 16, kernel_size=4, stride=2, padding=1)
#     3. Flatten()

#     Args:
#         input_shape: (channels, height, width)

#     Returns:
#         Output dimension after flattening
#     """
#     channels, height, width = input_shape

#     # First conv layer: Conv2d(channels, 16, kernel_size=4, stride=2, padding=1)
#     h1 = (height + 2 * 1 - 4) // 2 + 1
#     w1 = (width + 2 * 1 - 4) // 2 + 1

#     # Second conv layer: Conv2d(16, 16, kernel_size=4, stride=2, padding=1)
#     h2 = (h1 + 2 * 1 - 4) // 2 + 1
#     w2 = (w1 + 2 * 1 - 4) // 2 + 1

#     # Flatten: 16 channels * h2 * w2
#     return 16 * h2 * w2
