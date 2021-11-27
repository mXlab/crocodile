from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass 
class SlurmConfig:
    log_folder: Path = MISSING
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

    def __init__(self, args: Params = Params()):
        self.executor = None
        if self.slurm is not None:
            config = self.load_config(args.slurm)
            self.executor = self.create_executor(config)

    @staticmethod
    def load_config(filename: Path):
        schema = OmegaConf.structured(SlurmConfig)
        conf = OmegaConf.load(filename)
        return OmegaConf.merge(schema, conf)

    @staticmethod
    def create_executor(config) -> submitit.AutoExecutor:
        import submitit

        executor = submitit.AutoExecutor(folder=config.log_folder)
        executor.update_parameters(
            slurm_partition=config.partition,
            slurm_comment=config.comment,
            slurm_constraint=config.gpu_type,
            slurm_time=config.time_in_min,
            timeout_min=configtime_in_min,
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
            print("Launched job: %s"%(str(job.job_id)))






