from dataclasses import dataclass
from simple_parsing import parse, subgroups

from crocodile.trainer import TrainConfig
from crocodile.loader import trainer_subgroups, load_trainer
from crocodile.executor import (
    load_executor,
    ExecutorConfig,
    executor_subgroups,
)


@dataclass
class Config:
    executor: ExecutorConfig = subgroups(
        executor_subgroups,
        default="local",
    )
    trainer: TrainConfig = subgroups(trainer_subgroups, default="fastgan")


if __name__ == "__main__":
    config = parse(Config)

    executor = load_executor(config.executor)
    trainer = load_trainer(config.trainer)
    executor(trainer.train)
