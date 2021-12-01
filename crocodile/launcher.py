from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class SlurmConfig:
    log_folder: str = MISSING
    gpus_per_node: int = 1
    mem_by_gpu: int = 16
    partition: str = ""
    comment: str = ""
    gpu_type: str = ""
    time_in_min: int = 5
    nodes: int = 1
    cpus_per_task: int = 1
    slurm_array_parallelism: int = 1


class Launcher:
    @dataclass
    class Params:
        slurm: Optional[Path] = None
        gpus_per_node: Optional[int] = None

    def __init__(self, args: Params = Params()):
        self.executor = self.create_executor(args)

    @classmethod
    def create_executor(cls, args: Params):
        if args.slurm is None:
            return None
        return cls.create_slurm_executor(args)

    @staticmethod
    def load_config(args: Params):
        schema = OmegaConf.structured(SlurmConfig)
        conf = OmegaConf.load(args.slurm)
        conf = OmegaConf.merge(schema, conf)
        if args.gpus_per_node is not None:
            conf.gpus_per_node = args.gpus_per_node
        return conf

    @classmethod
    def create_slurm_executor(cls, args: Params):
        import submitit

        config = cls.load_config(args)

        executor = submitit.AutoExecutor(folder=config.log_folder)
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
            mem_gb=config.mem_by_gpu * config.gpus_per_node,
            slurm_array_parallelism=config.slurm_array_parallelism,
        )

        return executor

    def run(self, args):
        raise NotImplementedError

    def launch(self, args):
        if self.executor is None:
            self.run(args)
        else:
            job = self.executor.submit(self.run, args)
            print("Launched job: %s" % (str(job.job_id)))