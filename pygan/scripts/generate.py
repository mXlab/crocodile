from pygan.launcher import Launcher
from pygan.models import ModelType
from pygan.dataset import LaurenceDataset
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
from simple_parsing import ArgumentParser


class Generate(Launcher):
    @dataclass
    class Params(Launcher.Params):
        model_path: Path = None
        num_frames: int = 100
        model_type: ModelType = ModelType.STYLEFORMER
        output_dir: Path = None
        output_size: int = None
        name: str = "test1"

        def __post_init__(self):
            if self.output_dir is None:
                self.output_dir = Path("./results/%s/%s" % (
                                       self.model_type.value, self.name))

    def __init__(self, args):
        super().__init__(args)

    def run(self, args):
        dataset = LaurenceDataset(args.dataset)

        if args.output_size is None:
            args.ouput_size = dataset.resolution

        model_path = args.model_path.resolve()
        output_dir = args.output_dir.resolve()
        if model_path is None or not model_path.is_file():
            raise("Please specify a valid path for the model to load.")

        if args.model_type == ModelType.STYLEFORMER:
            os.chdir('pygan/models/Styleformer')
            command = "python generate.py --outdir=%s --network %s --num_frames %i" % (
                output_dir, model_path, args.num_frames)
            print("Running: %s" % command)
            subprocess.run(command.split())

        elif args.model_type == ModelType.FASTGAN:
            os.chdir('pygan/models/FastGAN')
            command = "python eval.py %s --ckpt=%s --im_size %i --out_size %i --n_samples %i" % (
                output_dir, model_path, dataset.resolution, args.output_size, args.num_frames)
            print("Running: %s" % command)
            subprocess.run(command.split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Generate.Params, dest="train")
    args = parser.parse_args()

    launcher = Generate(args.train)
    launcher.launch(args.train)
