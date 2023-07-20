from dataclasses import dataclass
from simple_parsing import parse, subgroups

from crocodile.trainer import Trainer, TrainerConfig
from crocodile.executor import load_executor, executor_subgroups, ExecutorConfig


@dataclass
class Config:
    executor: ExecutorConfig = subgroups(executor_subgroups, default="local")
    trainer: TrainerConfig = TrainerConfig()


if __name__ == "__main__":
    config = parse(Config)
    print("Initializing executor...")
    executor = load_executor(config.executor)
    print("Initializing trainer...")
    trainer = Trainer(config.trainer)
    print(f"Launching job on {executor.__name__}")
    executor(trainer.train)  # TODO: fix type error
