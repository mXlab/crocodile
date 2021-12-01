from crocodile.launcher import Launcher
from crocodile.dataset import LaurenceDataset
from . import ModelType
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from pathlib import Path


class Train(Launcher):
    @dataclass
    class Params(Launcher.Params):
        output_dir: Path = Path("./results")
        model_type: ModelType = ModelType.STYLEFORMER
        batch_size: int = 64

    def run(self, args):
        model = ModelType.load(args.train.model_type)
        model.train(args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Train.Params, dest="train")
    parser.add_arguments(LaurenceDataset.Params, dest="dataset")
    args = parser.parse_args()

    launcher = Train(args.train)
    launcher.launch(args)
