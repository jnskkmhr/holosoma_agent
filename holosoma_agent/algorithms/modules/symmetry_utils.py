from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from holosoma_agent.configs.symmetry_config import SymmetryConfig


class SymmetryUtils:
    """X-Z plane symmetry utilities for humanoid robots.

    Unlike the holosoma version, this class does **not** read ``env.robot_config``
    or ``env.observation_manager``. All information is passed explicitly at
    construction time, making it fully decoupled from the environment.
    """

    def __init__(
        self,
        cfg: SymmetryConfig,
        device: torch.device | str = None,
    ) -> None:

        self.device = device
        self.cfg = cfg

        # Initialize attributes that will be set during initialization
        self.observation_dims: Dict[str, int] = {}
        self.observation_dims_single_frame: Dict[
            str, int
        ] = {}  # Dimension without history
        self.history_lengths: Dict[str, int] = {}
        self.sub_observation_keys: Dict[str, List[str]] = {}
        self.sub_observation_indices: Dict[str, Dict[str, torch.Tensor]] = {}
        self.sub_observation_indices_single_frame: Dict[
            str, Dict[str, torch.Tensor]
        ] = {}  # Indices within single frame
        self.sub_observation_dims: Dict[str, int] = {}
        self.joint_index_map: torch.Tensor = torch.empty(0)
        self.sign_flip_mask: torch.Tensor = torch.empty(0)

        self._init_observation_config()
        self._init_joint_config()

    """
    initialization
    """

    def _init_observation_config(self) -> None:
        pass

    def _init_joint_config(self) -> None:
        pass

    """
    augmentation code
    """

    def augment_observations(
        self, obs: torch.Tensor, obs_list: Sequence[str]
    ) -> torch.Tensor:
        """Double the batch by concatenating obs with its mirrored version."""
        mirrored_obs = self.mirror_xz_plane(obs, obs_list)
        return torch.cat((obs, mirrored_obs), dim=0)

    def augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Double the batch by concatenating actions with mirrored actions."""
        mirrored_actions = self.mirror_action_xz_plane(actions)
        return torch.cat((actions, mirrored_actions), dim=0)

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

    def mirror_obs_base_lin_vel(self, base_lin_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the base linear velocity in robot's base frame.

        Parameters
        ----------
        base_lin_vel : torch.Tensor
            Base linear velocity with layout [v_x, v_y, v_z] in base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored velocity with y-component negated: [v_x, -v_y, v_z].
        """
        base_lin_vel[..., 1] = -base_lin_vel[..., 1]  # Flip y component
        return base_lin_vel

    def mirror_obs_base_ang_vel(self, base_ang_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the base angular velocity in robot's base frame.

        Parameters
        ----------
        base_ang_vel : torch.Tensor
            Base angular velocity with layout [ω_x, ω_y, ω_z] in base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored angular velocity with x and z components negated: [-ω_x, ω_y, -ω_z].
        """
        base_ang_vel[..., 0] = -base_ang_vel[..., 0]  # Flip x component
        base_ang_vel[..., 2] = -base_ang_vel[..., 2]  # Flip z component
        return base_ang_vel

    def mirror_obs_base_orientation(
        self, base_orientation: torch.Tensor
    ) -> torch.Tensor:
        """Mirrors the base orientation representation.

        Parameters
        ----------
        base_orientation : torch.Tensor
            Base orientation with layout [θ_x, θ_y, θ_z] (Euler angles or similar representation).

        Returns
        -------
        torch.Tensor
            Mirrored orientation with y-component negated: [θ_x, -θ_y, θ_z].
        """
        base_orientation[..., 1] = -base_orientation[..., 1]  # Flip y component
        return base_orientation

    def mirror_obs_projected_gravity(
        self, projected_gravity: torch.Tensor
    ) -> torch.Tensor:
        """Mirrors the projected gravity vector in robot's base frame.

        Parameters
        ----------
        projected_gravity : torch.Tensor
            Gravity vector projected into base frame with layout [g_x, g_y, g_z].

        Returns
        -------
        torch.Tensor
            Mirrored gravity vector with y-component negated: [g_x, -g_y, g_z].
        """
        projected_gravity[..., 1] = -projected_gravity[..., 1]  # Flip y component
        return projected_gravity

    def mirror_obs_command_lin_vel(self, command_lin_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded linear velocity.

        Parameters
        ----------
        command_lin_vel : torch.Tensor
            Commanded linear velocity with layout [v_x_cmd, v_y_cmd, v_z_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored velocity command with y-component negated: [v_x_cmd, -v_y_cmd, v_z_cmd].
        """
        command_lin_vel[..., 1] = -command_lin_vel[..., 1]  # Flip y component
        return command_lin_vel

    def mirror_obs_command_ang_vel(self, command_ang_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the commanded angular velocity (yaw only).

        Parameters
        ----------
        command_ang_vel : torch.Tensor
            Commanded yaw angular velocity (1D scalar) with layout [ω_yaw_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored yaw angular velocity command with sign negated: [-ω_yaw_cmd].
        """
        command_ang_vel[..., 0] = -command_ang_vel[..., 0]
        return command_ang_vel

    def mirror_obs_command_stand(self, command_stand: torch.Tensor) -> torch.Tensor:
        """Mirrors the stand command (no transformation needed).

        Parameters
        ----------
        command_stand : torch.Tensor
            Stand command signal (scalar or binary).
        Returns
        -------
        torch.Tensor
            Unchanged stand command.
        """
        return command_stand

    def mirror_obs_command_waist_dofs(
        self, command_waist_dofs: torch.Tensor
    ) -> torch.Tensor:
        """Mirrors the commanded waist joint positions.

        Parameters
        ----------
        command_waist_dofs : torch.Tensor
            Waist joint commands with layout [yaw_cmd, roll_cmd, pitch_cmd].

        Returns
        -------
        torch.Tensor
            Mirrored waist commands with yaw and roll negated: [-yaw_cmd, -roll_cmd, pitch_cmd].
        """
        # flip yaw
        command_waist_dofs[..., 0] = -command_waist_dofs[..., 0]
        # flip roll
        command_waist_dofs[..., 1] = -command_waist_dofs[..., 1]
        return command_waist_dofs

    def mirror_obs_command_base_height(
        self, command_base_height: torch.Tensor
    ) -> torch.Tensor:
        """Mirrors the commanded base height (no transformation needed).

        Parameters
        ----------
        command_base_height : torch.Tensor
            Commanded base height (scalar).
            Inputs: [height_cmd].

        Returns
        -------
        torch.Tensor
            Unchanged height command.
            Outputs: [height_cmd].
        """
        return command_base_height

    def mirror_obs_sin_phase(self, sin_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the sine phase for gait timing.

        Parameters
        ----------
        sin_phase : torch.Tensor
            Sine of gait phase with layout [sin(φ_left), sin(φ_right), ...].

        Returns
        -------
        torch.Tensor
            Mirrored phase with first component negated: [-sin(φ_left), sin(φ_right), ...].
        """
        sin_phase[..., 0] = -sin_phase[..., 0]
        return sin_phase

    def mirror_obs_cos_phase(self, cos_phase: torch.Tensor) -> torch.Tensor:
        """Mirrors the cosine phase for gait timing.

        Parameters
        ----------
        cos_phase : torch.Tensor
            Cosine of gait phase with layout [cos(φ_left), cos(φ_right), ...].

        Returns
        -------
        torch.Tensor
            Mirrored phase with first component negated: [-cos(φ_left), cos(φ_right), ...].
        """
        cos_phase[..., 0] = -cos_phase[..., 0]
        return cos_phase

    def mirror_obs_dof_pos(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """Mirrors the joint positions using joint mapping and sign flipping.

        Parameters
        ----------
        dof_pos : torch.Tensor
            Joint positions with layout following the robot's DOF order.
            Typically [hip_joints..., knee_joints..., ankle_joints..., waist_joints..., arm_joints...].
            Inputs: [q0, q1, q2, ..., qN] (original joint order).

        Returns
        -------
        torch.Tensor
            Mirrored joint positions with left-right mapping applied and signs flipped for appropriate joints.
            Outputs: [q_mapped0 * sign0, q_mapped1 * sign1, ..., q_mappedN * signN] (mirrored and sign-flipped).
        """
        return dof_pos[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_dof_vel(self, dof_vel: torch.Tensor) -> torch.Tensor:
        """Mirrors the joint velocities (same mapping as joint positions).

        Parameters
        ----------
        dof_vel : torch.Tensor
            Joint velocities with same layout as dof_pos.
            Inputs: [qd0, qd1, qd2, ..., qdN] (original joint velocities).

        Returns
        -------
        torch.Tensor
            Mirrored joint velocities with same transformation as joint positions.
            Outputs: [qd_mapped0 * sign0, qd_mapped1 * sign1, ..., qd_mappedN * signN] (mirrored and sign-flipped).
        """
        return dof_vel[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Mirrors the previous actions (same mapping as joint positions).

        Parameters
        ----------
        actions : torch.Tensor
            Previous action values with same layout as dof_pos.
            Inputs: [a0, a1, a2, ..., aN] (original actions).

        Returns
        -------
        torch.Tensor
            Mirrored actions with same transformation as joint positions.
            Outputs: [a_mapped0 * sign0, a_mapped1 * sign1, ..., a_mappedN * signN] (mirrored and sign-flipped).
        """
        return actions[..., self.joint_index_map] * self.sign_flip_mask

    def mirror_obs_ee_apply_force(self, ee_apply_force: torch.Tensor) -> torch.Tensor:
        """Mirrors the end-effector applied forces in base frame.

        Parameters
        ----------
        ee_apply_force : torch.Tensor
            Applied forces with layout [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz].
            Forces are already transformed to base frame coordinates.

        Returns
        -------
        torch.Tensor
            Mirrored forces with left-right swapped and y-components negated:
            [right_fx, -right_fy, right_fz, left_fx, -left_fy, left_fz].
        """
        # Note: this force is already transformed to the base frame
        left_ee_apply_force = ee_apply_force[..., :3].clone()
        left_ee_apply_force[..., 1] = -left_ee_apply_force[..., 1]
        right_ee_apply_force = ee_apply_force[..., 3:].clone()
        right_ee_apply_force[..., 1] = -right_ee_apply_force[..., 1]
        return torch.cat([right_ee_apply_force, left_ee_apply_force], dim=-1)
