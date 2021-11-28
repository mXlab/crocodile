from pygan.launcher import Launcher
from pygan.dataset import LaurenceDataset
from pygan.models import ModelType
import torch
from dataclasses import dataclass
import subprocess
import os
from simple_parsing import ArgumentParser
from pathlib import Path


class Train(Launcher):
    @dataclass
    class Params(Launcher.Params):
        output_dir: Path = Path("./results")
        model_type: ModelType = ModelType.STYLEFORMER
        batch_size: int = 128

    def __init__(self, args):
        super().__init__(args)

    def run(self, args):

        print("Loading dataset...")
        dataset = LaurenceDataset(args.dataset)

        if args.train.model_type == ModelType.STYLEFORMER:
            data_path = dataset.get_path()
            gpus = torch.cuda.device_count()
            os.chdir('pygan/models/Styleformer')
            command = "python train.py --outdir=%s --data=%s --gpus=%i --num_layers=1,2,1,1 --g_dict=1024,256,64,64 --linformer=1" % (
                args.train.output_dir.resolve(), data_path, gpus)
            print("Running: %s" % command)
            subprocess.run(command.split())
        elif args.train.model_type == ModelType.FASTGAN:
            data_path = dataset.get_path()
            os.chdir('pygan/models/FastGAN')
            command = "python train.py --path=%s --batch_size %i --im_size %i" % (
                data_path, args.train.batch_size, dataset.resolution)
            print("Running: %s" % command)
            subprocess.run(command.split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Train.Params, dest="train")
    parser.add_arguments(LaurenceDataset.Params, dest="dataset")
    args = parser.parse_args()

    launcher = Train(args.train)
    launcher.launch(args)
