from .executor import ExecutorConfig, Executor
from dataclasses import dataclass
from typing import Optional, Callable, Any
from pathlib import Path

try:
    import submitit
except ImportError:
    print(
        "Couldn't import submitit. Install it if you plan on running this code on the cluster."
    )


@dataclass
class SlurmConfig(ExecutorConfig):
    log_folder: Path = Path("./logs")
    gpus_per_node: Optional[int] = None
    partition: Optional[str] = None
    comment: Optional[str] = None
    gpu_type: Optional[str] = None
    time_in_min: Optional[int] = None
    nodes: Optional[int] = None
    cpus_per_task: Optional[int] = None
    slurm_array_parallelism: Optional[int] = None
    mem_gb: Optional[int] = 16
    account: Optional[str] = "def-sofian"

    def __post_init__(self):
        self.log_folder.mkdir(parents=True, exist_ok=True)


class SlurmExecutor(Executor):
    __name__ = "slurm"

    def __init__(self, config: SlurmConfig = SlurmConfig()):
        self.executor = submitit.AutoExecutor(folder=config.log_folder)

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
            mem_gb=config.mem_gb,
            slurm_array_parallelism=config.slurm_array_parallelism,
            slurm_account=config.account,
        )

    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        job = self.executor.submit(func, *args, **kwargs)
        print("Launched job: %s" % (str(job.job_id)))
