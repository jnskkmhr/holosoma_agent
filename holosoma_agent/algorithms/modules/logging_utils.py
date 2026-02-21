from __future__ import annotations

import os
import pathlib
import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator

import torch
import wandb
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from torch.utils.tensorboard import SummaryWriter

from holosoma_agent.algorithms.modules.average_meters import TensorAverageMeterDict

console = Console()


class LoggingHelper:
    """Training logging helper — TensorBoard + Weights & Biases + rich console.

    Parameters
    ----------
    writer : SummaryWriter
    log_dir : str | pathlib.Path
    num_envs : int
    num_steps_per_env : int
    num_learning_iterations : int
    device : str
    prefix : str
        Prefix prepended to every TensorBoard key.
    title : str
        Title of the rich console panel.
    is_main_process : bool
    num_gpus : int
    """

    def __init__(
        self,
        writer: SummaryWriter,
        log_dir: str | pathlib.Path,
        num_envs: int,
        num_steps_per_env: int,
        num_learning_iterations: int,
        device: str = "cpu",
        prefix: str = "",
        title: str = "Training Log",
        is_main_process: bool = True,
        num_gpus: int = 1,
    ):
        self.writer = writer
        self.log_dir = str(log_dir)
        self.device = device
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.collection_time: float = 0.0
        self.learn_time: float = 0.0
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.num_learning_iterations = num_learning_iterations
        self.prefix = prefix
        self.title = title
        self.is_main_process = is_main_process
        self.num_gpus = num_gpus

        self.ep_infos: list[dict[str, Any]] = []
        self.raw_ep_infos: list[dict[str, Any]] = []
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.cur_episode_length = torch.zeros(
            num_envs, dtype=torch.float, device=device
        )
        self.episode_env_tensors = TensorAverageMeterDict()

    @contextmanager
    def record_collection_time(self) -> Generator[None, None, None]:
        start = time.perf_counter()
        yield
        self.collection_time += time.perf_counter() - start

    @contextmanager
    def record_learn_time(self) -> Generator[None, None, None]:
        start = time.perf_counter()
        yield
        self.learn_time += time.perf_counter() - start

    def update_episode_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict[str, Any]
    ) -> None:
        if not self.is_main_process:
            return
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        if "raw_episode" in infos:
            self.raw_ep_infos.append(infos["raw_episode"])
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        new_ids = (dones > 0).nonzero(as_tuple=False)
        if len(new_ids) > 0:
            self.rewbuffer.extend(
                self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.lenbuffer.extend(
                self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.cur_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0
        if "to_log" in infos:
            self.episode_env_tensors.add(infos["to_log"])

    def post_epoch_logging(
        self,
        it: int,
        loss_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        self.tot_timesteps += self.num_steps_per_env * self.num_envs * self.num_gpus
        self.tot_time += self.collection_time + self.learn_time
        iteration_time = self.collection_time + self.learn_time
        ep_string, ep_scalars = self._log_episode_info()
        env_log_dict = {
            f"Env/{k}": v for k, v in self.episode_env_tensors.mean_and_clear().items()
        }
        fps = int(
            self.num_steps_per_env
            * self.num_envs
            * self.num_gpus
            / (self.collection_time + self.learn_time + 1e-8)
        )
        self._logging_to_writer(
            it, loss_dict, extra_log_dicts, env_log_dict, fps, ep_scalars
        )
        log_string = self._create_console_output(
            it,
            loss_dict,
            env_log_dict,
            extra_log_dicts,
            ep_string,
            width,
            pad,
            iteration_time,
            fps,
        )
        with Live(
            Panel(log_string, title=self.title), refresh_per_second=4, console=console
        ):
            pass
        self.ep_infos.clear()
        self.raw_ep_infos.clear()
        self.learn_time = 0.0
        self.collection_time = 0.0

    def _log_episode_info(self) -> tuple[str, dict[str, float]]:
        if not self.is_main_process:
            return "", {}
        ep_string = ""
        scalars: dict[str, float] = {}
        for infos, prefix in [
            (self.ep_infos, "Episode"),
            (self.raw_ep_infos, "RawEpisode"),
        ]:
            if not infos:
                continue
            for key in infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep in infos:
                    v = ep[key]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor([v])
                    if len(v.shape) == 0:
                        v = v.unsqueeze(0)
                    infotensor = torch.cat((infotensor, v.to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor).item()
                scalars[f"{prefix}/{key}"] = value
                ep_string += (
                    f"""{f"Mean {prefix.lower()} {key}:":>{35}} {value:.4f}\n"""
                )
        return ep_string, scalars

    def _logging_to_writer(
        self,
        it: int,
        loss_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        env_log_dict: dict[str, float],
        fps: int,
        ep_scalars: dict[str, float],
    ) -> None:
        if not self.is_main_process:
            return
        scalars: dict[str, float] = {}
        for k, v in loss_dict.items():
            scalars[f"Loss/{k}"] = v
        scalars.update(env_log_dict)
        scalars.update(ep_scalars)
        for section, d in extra_log_dicts.items():
            for k, v in d.items():
                scalars[f"{section}/{k}"] = v
        scalars["Perf/total_fps"] = fps
        scalars["Perf/collection_time"] = self.collection_time
        scalars["Perf/learning_time"] = self.learn_time
        if self.rewbuffer:
            scalars["Train/mean_reward"] = statistics.mean(self.rewbuffer)
        if self.lenbuffer:
            scalars["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
        scalars["Train/num_samples"] = self.tot_timesteps
        scalars = {f"{self.prefix}{k}": v for k, v in scalars.items()}
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, global_step=it)
        if wandb.run is not None:
            wandb.log(dict(scalars, global_step=it), step=it)

    def _create_console_output(
        self,
        it,
        loss_dict,
        env_log_dict,
        extra_log_dicts,
        ep_string,
        width,
        pad,
        iteration_time,
        fps,
    ) -> str:
        if not self.is_main_process:
            return ""
        header = (
            f" \033[1m Learning iteration {it}/{self.num_learning_iterations} \033[0m "
        )
        s = f"{header.center(width, ' ')}\n\n"
        s += (
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s """
            f"""(Collection: {self.collection_time:.3f}s, Learning: {self.learn_time:.3f}s)\n"""
        )
        if self.rewbuffer:
            s += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
        if self.lenbuffer:
            s += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
        for k, v in loss_dict.items():
            s += f"{f'{k}:':>{pad}} {v:.4f}\n"
        for k, v in env_log_dict.items():
            s += f"{f'{k}:':>{pad}} {v:.4f}\n"
        for section, d in extra_log_dicts.items():
            for k, v in d.items():
                s += f"{f'{section}/{k}:':>{pad}} {v:.4f}\n"
        s += ep_string
        eta = self.tot_time / (it + 1) * (self.num_learning_iterations - it)
        s += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {eta:.1f}s\n"""
        )
        s += f"Logging Directory: {self.log_dir}"
        return s

    def save_checkpoint_artifact(self, state_dict: dict[str, Any], path: str) -> None:
        if not path.startswith(self.log_dir):
            raise ValueError(
                f"Path {path} is not in the logging directory {self.log_dir}"
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        torch.save(state_dict, path)
        self.save_to_wandb(path)

    def save_to_wandb(self, file_path: str) -> None:
        if wandb.run is None:
            return
        wandb.save(file_path, base_path=self.log_dir)
