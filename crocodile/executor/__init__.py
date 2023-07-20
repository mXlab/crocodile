from .executor import ExecutorConfig, LocalExecutor, LocalExecutorConfig
from .slurm import SlurmExecutor, SlurmConfig


executor_subgroups = {"local": LocalExecutorConfig(), "slurm": SlurmConfig()}


def load_executor(config: ExecutorConfig):
    match config:
        case LocalExecutorConfig():
            return LocalExecutor()
        case SlurmConfig():
            return SlurmExecutor(config)
        case _:
            raise ValueError("Please choose a valid executor.")
