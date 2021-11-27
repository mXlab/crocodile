from pygan.launcher import Launcher
from pygan.dataset import LaurenceDataset
from pygan.models import ModelType
import torch
from dataclasses import dataclass
import subprocess
import os


class Train(Launcher):
    @dataclass
    class Params(Launcher.Params):
        output_dir: str = "./results"
        model_type: ModelType = ModelType.STYLEFORMER

    def __init__(self):
        super().__init__()

    def run(self, args):

        dataset = LaurenceDataset(args.dataset)

        if args.model_type == ModelType.STYLEFORMER:
            gpus = torch.cuda.device_count()
            os.chdir('pygan/models/Styleformer')
            command = "python train.py --outdir=%s --data=%s --gpus=%i" % (
                args.output_dir, dataset.get_path(), gpus)
            subprocess.run(command.split())
