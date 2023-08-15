from dataclasses import dataclass
from simple_parsing import parse, subgroups

from crocodile.generator import TrainParams, load_generator
from crocodile.executor import (
    load_executor,
    ExecutorConfig,
    LocalExecutorConfig,
    SlurmConfig,
)


@dataclass
class TrainConfig:
    executor: ExecutorConfig = subgroups(
        {
            "local": LocalExecutorConfig(),
            "slurm": SlurmConfig(_default_log_dir="crocodile/logs"),
        },
        default="local",
    )
    trainer: TrainParams = TrainParams()


if __name__ == "__main__":
    config = parse(TrainConfig)

    executor = load_executor(config.executor)
    generator = load_generator(config.trainer.generator)
    executor(generator.train, config.trainer)
