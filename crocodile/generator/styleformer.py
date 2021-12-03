import torch
from crocodile.dataset import LaurenceDataset
import os
import subprocess
import Styleformer.dnnlib as dnnlib
import Styleformer.legacy as legacy
import numpy as np
from .base import Model


class Styleformer(Model):
    @staticmethod
    def train(args):
        print("Loading dataset...")
        dataset = LaurenceDataset(args.dataset)

        data_path = dataset.get_path()

        output_dir = args.train.output_dir.resolve()
        gpus = torch.cuda.device_count()
        command = "python -m Styleformer.train --outdir=%s --data=%s --gpus=%i --num_layers=1,2,1,1 --g_dict=1024,256,64,64 --linformer=1" % (
            output_dir, data_path, gpus)
        print("Running: %s" % command)
        subprocess.run(command.split())

    @staticmethod
    def load_model(path: str, device=None):
        if device is None:
            device = torch.device('cuda')
        with dnnlib.util.open_url(path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
        return Styleformer(model, model.z_dim, device)

    def sample_z(self, n_samples=1):
        return torch.from_numpy(np.random.randn(n_samples, self.model.z_dim)).to(self.device)

    def __call__(self, z):
        if self.model is None:
            raise("No model loaded. Use `load_model` first.")

        label = torch.zeros([len(z), self.model.c_dim], device=self.device)
        img = self.model(z, label)
        return img
