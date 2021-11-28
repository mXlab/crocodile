from pygan.launcher import Launcher
from dataclasses import dataclass
from pygan.models import ModelType
from pathlib import Path
import os
import subprocess
from simple_parsing import ArgumentParser


class Generate(Launcher):
    @dataclass
    class Params(Launcher.Params):
        model_path: Path
        num_frames: int = 100
        model_type: ModelType = ModelType.STYLEFORMER
        output_dir: Path = Path("./results/styleformer/test1")

    def __init__(self, args):
        super().__init__(args)

    def run(self, args):
        if args.model_path is None or not args.model_path.is_file():
            raise("Please specify a valid path for the model to load.")

        if args.train.model_type == ModelType.STYLEFORMER:
            os.chdir('pygan/models/Styleformer')
            command = "python generate.py --outdir=%s --network %s --num_frames %i" % (
                args.output_dir.resolve(), args.model_path.resolve(), args.num_frames)
            print("Running: %s" % command)
            subprocess.run(command.split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Generate.Params, dest="train")
    args = parser.parse_args()

    launcher = Generate(args.train)
    launcher.launch(args.train)
