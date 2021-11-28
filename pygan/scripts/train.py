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
                args.train.output_dir, data_path, gpus)
            print("Running: %s" % command)
            subprocess.run(command.split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Train.Params, dest="train")
    parser.add_arguments(LaurenceDataset.Params, dest="dataset")
    args = parser.parse_args()

    launcher = Train(args)
    launcher.launch(args)
