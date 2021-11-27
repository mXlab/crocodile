from pygan.launcher import Launcher
from pygan.dataset import LaurenceDataset
from pygan.models import ModelType
import torch
from dataclasses import dataclass
import subprocess
import os
from simple_parsing import ArgumentParser


class Train(Launcher):
    @dataclass
    class Params(Launcher.Params):
        output_dir: str = "./results"
        model_type: ModelType = ModelType.STYLEFORMER

    def __init__(self):
        super().__init__()

    def run(self, args):

        dataset = LaurenceDataset(args.dataset)

        if args.train.model_type == ModelType.STYLEFORMER:
            gpus = torch.cuda.device_count()
            print("Running Styleformer, nb of gpus: %i" % gpus)
            os.chdir('pygan/models/Styleformer')
            command = "python train.py --outdir=%s --data=%s --gpus=%i" % (
                args.train.output_dir, dataset.get_path(), gpus)
            subprocess.run(command.split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Train.Params, dest="train")
    parser.add_arguments(LaurenceDataset.Params, dest="dataset")
    args = parser.parse_args()

    launcher = Train()
    launcher.run(args)
