from __future__ import annotations

from typing import Dict, List, Sequence

import torch


class SymmetryUtils:
    """X-Z plane symmetry utilities for humanoid robots.

    Unlike the holosoma version, this class does **not** read ``env.robot_config``
    or ``env.observation_manager``. All information is passed explicitly at
    construction time, making it fully decoupled from the environment.

    Parameters
    ----------
    dof_names : list[str]
        Ordered list of DOF names in the robot (defines the action vector layout).
    symmetry_joint_names : dict[str, str]
        Mapping ``{left_joint_name: right_joint_name}`` for left-right joint swapping.
    flip_sign_joint_names : list[str]
        DOF names whose sign must be flipped when mirroring.
    obs_dims : dict[str, int]
        Flat dimension of each observation group (history already included).
    history_lengths : dict[str, int]
        History factor per observation group.
    sub_obs_keys : dict[str, list[str]]
        Ordered sub-observation keys per group (must have a ``mirror_obs_<key>`` method
        defined on this class for each).
    obs_dims_single_frame : dict[str, int]
        Single-frame (no history) flat dimension per group.
    sub_obs_indices_single_frame : dict[str, dict[str, torch.Tensor]]
        Per-group, per-sub-key index tensors into the single-frame obs vector.
    device : str
    """

    def __init__(
        self,
        dof_names: List[str],
        symmetry_joint_names: Dict[str, str],
        flip_sign_joint_names: List[str],
        obs_dims: Dict[str, int],
        history_lengths: Dict[str, int],
        sub_obs_keys: Dict[str, List[str]],
        obs_dims_single_frame: Dict[str, int],
        sub_obs_indices_single_frame: Dict[str, Dict[str, torch.Tensor]],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.observation_dims = obs_dims
        self.history_lengths = history_lengths
        self.sub_observation_keys = sub_obs_keys
        self.observation_dims_single_frame = obs_dims_single_frame
        self.sub_observation_indices_single_frame = sub_obs_indices_single_frame

        # Build joint index map
        name_to_idx = {name: i for i, name in enumerate(dof_names)}
        joint_index_mapping = {}
        for j1, j2 in symmetry_joint_names.items():
            if j1 in name_to_idx and j2 in name_to_idx:
                joint_index_mapping[name_to_idx[j1]] = name_to_idx[j2]
        self.joint_index_map = torch.tensor(
            [joint_index_mapping.get(i, i) for i in range(len(dof_names))],
            device=device,
            dtype=torch.long,
        )

        # Build sign flip mask
        flip_indices = {
            name_to_idx[n] for n in flip_sign_joint_names if n in name_to_idx
        }
        self.sign_flip_mask = torch.tensor(
            [-1.0 if i in flip_indices else 1.0 for i in range(len(dof_names))],
            device=device,
            dtype=torch.float,
        )

    # ── High-level augmentation helpers ──────────────────────────────────────

    def augment_observations(
        self, obs: torch.Tensor, obs_list: Sequence[str]
    ) -> torch.Tensor:
        """Double the batch by concatenating obs with its mirrored version."""
        return torch.cat((obs, self.mirror_xz_plane(obs, obs_list)), dim=0)

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Double the batch by concatenating actions with mirrored actions."""
        return torch.cat((actions, self.mirror_action_xz_plane(actions)), dim=0)

    def mirror_xz_plane(
        self, observation: torch.Tensor, obs_list: Sequence[str]
    ) -> torch.Tensor:
        """Apply x-z plane symmetry to an observation tensor."""
        mirrored = observation.clone()
        B, _ = mirrored.shape
        idx = 0
        for obs_key in obs_list:
            length = self.observation_dims[obs_key]
            seg = mirrored[..., idx : idx + length]
            seg = seg.reshape(
                B,
                self.history_lengths[obs_key],
                self.observation_dims_single_frame[obs_key],
            )
            for sub_key in self.sub_observation_keys[obs_key]:
                sub_idx = self.sub_observation_indices_single_frame[obs_key][sub_key]
                mirror_fn = getattr(self, f"mirror_obs_{sub_key}")
                seg[..., sub_idx] = mirror_fn(seg[..., sub_idx])
            mirrored[..., idx : idx + length] = seg.reshape(B, length)
            idx += length
        return mirrored

    def mirror_action_xz_plane(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., self.joint_index_map] * self.sign_flip_mask

    # ── Per-component mirror functions ────────────────────────────────────────

    def mirror_obs_base_lin_vel(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 1] = -x[..., 1]
        return x

    def mirror_obs_base_ang_vel(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 0] = -x[..., 0]
        x[..., 2] = -x[..., 2]
        return x

    def mirror_obs_base_orientation(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 1] = -x[..., 1]
        return x

    def mirror_obs_projected_gravity(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 1] = -x[..., 1]
        return x

    def mirror_obs_command_lin_vel(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 1] = -x[..., 1]
        return x

    def mirror_obs_command_ang_vel(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 0] = -x[..., 0]
        return x

    def mirror_obs_command_stand(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def mirror_obs_command_waist_dofs(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 0] = -x[..., 0]
        x[..., 1] = -x[..., 1]
        return x

    def mirror_obs_command_base_height(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def mirror_obs_sin_phase(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 0] = -x[..., 0]
        return x

    def mirror_obs_cos_phase(self, x: torch.Tensor) -> torch.Tensor:
        x[..., 0] = -x[..., 0]
        return x

    def mirror_obs_dof_pos(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_dof_vel(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_actions(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_ee_apply_force(self, x: torch.Tensor) -> torch.Tensor:
        left = x[..., :3].clone()
        left[..., 1] = -left[..., 1]
        right = x[..., 3:].clone()
        right[..., 1] = -right[..., 1]
        return torch.cat([right, left], dim=-1)
