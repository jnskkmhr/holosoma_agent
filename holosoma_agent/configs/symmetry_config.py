from __future__ import annotations

from dataclasses import dataclass, MISSING


@dataclass
class SymmetryConfig:
    """Symmetry configuration for X-Z plane mirroring."""

    joint_names: list[str] = MISSING
    symmetry_joint_names: dict[str, str] = MISSING
    sign_flip_joints: list[str] = MISSING
