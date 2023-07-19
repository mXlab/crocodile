from dataclasses import dataclass
from typing import Union
from simple_parsing import ArgumentParser

from crocodile.trainer import Trainer, TrainerConfig


@dataclass
class Prepare:
    """Download all necessary assets for training."""

    config: TrainerConfig = TrainerConfig()

    def execute(self):
        trainer = Trainer(self.config)


@dataclass
class Train:
    """Run the training."""

    config: TrainerConfig = TrainerConfig()

    def execute(self):
        trainer = Trainer(self.config)
        trainer.train()


@dataclass
class Program:
    """A CLI for training a generator on the Laurence dataset."""

    command: Union[Train, Prepare]

    def execute(self):
        return self.command.execute()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Program, dest="prog")
    args = parser.parse_args()
    prog: Program = args.prog

    prog.execute()
