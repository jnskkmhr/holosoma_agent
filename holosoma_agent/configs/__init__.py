from typing import Union

from holosoma_agent.configs.fast_sac_config import FastSACConfig
from holosoma_agent.configs.ppo_config import PPOConfig

AlgoInitConfig = Union[PPOConfig, FastSACConfig]

__all__ = ["AlgoInitConfig", "PPOConfig", "FastSACConfig"]
