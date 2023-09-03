from dataclasses import dataclass
from typing import Optional, Callable, Any
import os
from pathlib import Path

import submitit

from .executor import Executor, ExecutorConfig


@dataclass
class SlurmConfig(ExecutorConfig):
    log_dir: Optional[Path] = None
    gpus_per_node: Optional[int] = None
    mem: Optional[int] = None
    partition: Optional[str] = None
    comment: Optional[str] = None
    gpu_type: Optional[str] = None
    time_in_min: Optional[int] = None
    nodes: Optional[int] = None
    cpus_per_task: Optional[int] = None
    slurm_array_parallelism: Optional[int] = None
    account: str = "def-sofian"
    _default_log_dir: str = "logs"

    def __post_init__(self):
        scratch_dir = os.environ.get("SCRATCH")
        if self.log_dir is None:
            if scratch_dir is not None:
                self.log_dir = Path(scratch_dir) / self._default_log_dir
            else:
                self.log_dir = Path(".") / self._default_log_dir


def create_slurm_executor(config: SlurmConfig = SlurmConfig()):
    assert config.log_dir is not None
    executor = submitit.AutoExecutor(folder=config.log_dir)
    executor.update_parameters(
        slurm_partition=config.partition,
        slurm_comment=config.comment,
        slurm_constraint=config.gpu_type,
        slurm_time=config.time_in_min,
        timeout_min=config.time_in_min,
        nodes=config.nodes,
        cpus_per_task=config.cpus_per_task,
        tasks_per_node=config.gpus_per_node,
        gpus_per_node=config.gpus_per_node,
        mem_gb=config.mem,
        slurm_array_parallelism=config.slurm_array_parallelism,
        slurm_account=config.account,
    )

    return executor


class SlurmExecutor(Executor):
    __name__ = "slurm"

    def __init__(self, config: SlurmConfig = SlurmConfig()):
        self.executor = submitit.AutoExecutor(folder=config.log_dir)

        self.executor.update_parameters(
            slurm_partition=config.partition,
            slurm_comment=config.comment,
            slurm_constraint=config.gpu_type,
            slurm_time=config.time_in_min,
            timeout_min=config.time_in_min,
            nodes=config.nodes,
            cpus_per_task=config.cpus_per_task,
            tasks_per_node=config.gpus_per_node,
            gpus_per_node=config.gpus_per_node,
            mem_gb=config.mem,
            slurm_array_parallelism=config.slurm_array_parallelism,
            slurm_account=config.account,
        )

    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        job = self.executor.submit(func, *args, **kwargs)
        print("Launched job: %s" % (str(job.job_id)))
