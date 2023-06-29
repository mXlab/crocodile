import torch
from torch import nn
from crocodile.dataset import LaurenceDataset
import subprocess
import Styleformer.dnnlib as dnnlib
import Styleformer.legacy as legacy
import numpy as np
from .generator import TrainParams, Generator
from typing import Optional


class StyleformerModel(nn.Module):
    z_dim: int
    c_dim: int


class Styleformer(Generator):
    model: StyleformerModel

    @classmethod
    def prepare(cls, params: TrainParams = TrainParams()):
        print("Loading dataset...")
        LaurenceDataset.download(params.dataset)

    @classmethod
    def train(cls, params: TrainParams = TrainParams()):
        dataset = LaurenceDataset(params.dataset)
        data_path = dataset.get_path()
        cls.set_dir(params)
        gpus = torch.cuda.device_count()
        command = (
            "python -m Styleformer.train --outdir=%s --data=%s --gpus=%i --num_layers=1,2,1,1 --g_dict=1024,256,64,64 --linformer=1"
            % (params.log_dir, data_path, gpus)
        )
        print("Running: %s" % command)
        subprocess.run(command.split())

    @staticmethod
    def load(
        params: TrainParams = TrainParams(),
        epoch: Optional[int] = None,
        device=None,
    ) -> Generator:
        if epoch is None:
            path = sorted(params.log_dir.glob("network-snapshot-*.pkl"))[-1]
        else:
            path = params.log_dir / f"network-snapshot-{epoch:06d}.pkl"

        if device is None:
            device = torch.device("cuda")
        with dnnlib.util.open_url(path.as_posix()) as f:
            model = legacy.load_network_pkl(f)["G_ema"]
        return Styleformer(model, model.z_dim, device=device)

    def sample_z(self, n_samples: int = 1) -> torch.Tensor:
        return torch.from_numpy(np.random.randn(n_samples, self.model.z_dim)).to(
            self.device
        )

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("No model loaded. Use `load_model` first.")

        label = torch.zeros([len(z), self.model.c_dim], device=self.device)
        img = self.model(z, label)
        return img
