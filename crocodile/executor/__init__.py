from .executor import Executor, LocalExecutor
from .slurm import SlurmExecutor, SlurmConfig, default_config
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from omegaconf import OmegaConf


def load_slurm_config(path: Optional[Path] = None) -> SlurmConfig:
    schema = OmegaConf.structured(default_config)
    if path is not None and path.is_file():
        conf = OmegaConf.load(path)
        schema = OmegaConf.merge(schema, conf)
    return schema


@dataclass
class ExecutorConfig:
    config_file: Optional[Path] = None
    slurm_options: SlurmConfig = SlurmConfig()

    def __post_init__(self):
        config = load_slurm_config(self.config_file)
        self.slurm_options.merge(config)


def load_executor(config: ExecutorConfig = ExecutorConfig()) -> Executor:
    if config.config_file is None:
        return LocalExecutor()
    else:
        return SlurmExecutor(config)
