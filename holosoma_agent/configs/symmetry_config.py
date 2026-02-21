from __future__ import annotations

from dataclasses import dataclass, field, MISSING
from typing import List


@dataclass
class SymmetryConfig:
    """Symmetry configuration for X-Z plane mirroring."""

    sub_observation_dims: dict[str, dict[str, int]] = MISSING
    joint_names: list[str] = MISSING
    sign_flip_joints: list[str] = MISSING
