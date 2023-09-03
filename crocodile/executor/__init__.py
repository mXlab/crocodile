from .executor import Executor, LocalExecutor, LocalExecutorConfig, ExecutorConfig
from .slurm import SlurmExecutor, SlurmConfig

executor_subgroups = {"local": LocalExecutorConfig, "slurm": SlurmConfig}


def load_executor(config: ExecutorConfig) -> Executor:
    match config:
        case LocalExecutorConfig():
            return LocalExecutor()
        case SlurmConfig():
            return SlurmExecutor(config)
        case _:
            raise ValueError("Executor not supported.")
