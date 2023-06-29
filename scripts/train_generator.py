from dataclasses import dataclass
from typing import Union
from crocodile.generator import TrainParams, load_generator
from crocodile.executor import load_executor, ExecutorConfig
from simple_parsing import ArgumentParser

@dataclass
class Prepare:

    params: TrainParams

    def execute(self):
        """Execute the program."""
        LaurenceDataset.download(params.dataset)

@dataclass
class Train:
    executor: ExecutorConfig
    params: TrainParams

    def execute(self):
        """Execute the program."""
        executor = load_executor(self.executor)
        generator = load_generator(self.params.generator)
        generator.train(self.params)


@dataclass
class Program:
    """Some top-level command"""

    command: Union[Train, Prepare]

    def execute(self):
        """Execute the program."""
        return self.command.execute()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Program, dest="prog")
    args = parser.parse_args()
    prog: Program = args.prog
    prog.execute()