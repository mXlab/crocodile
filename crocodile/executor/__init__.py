from .executor import BaseExecutorConfig, LocalExecutor, LocalExecutorConfig
from .slurm import SlurmExecutor, SlurmConfig


def load_executor(config: BaseExecutorConfig):
    match config:
        case LocalExecutorConfig():
            return LocalExecutor()
        case SlurmConfig():
            return SlurmExecutor(config)
