from .generator import Generator, TrainParams
from crocodile.dataset import LaurenceDataset
import subprocess
import torch
import FastGAN.models as fastgan
from typing import Optional


class FastGAN(Generator):
    @classmethod
    def train(cls, params: TrainParams = TrainParams()):
        print("Loading dataset...")
        dataset = LaurenceDataset(params.dataset)

        data_path = dataset.get_path()
        cls.set_dir(params)
        command = "python FastGAN.train.py --outdir %s --path=%s --batch_size %i --im_size %i" % (
            params.log_dir, data_path, params.batch_size, dataset.resolution)
        print("Running: %s" % command)
        subprocess.run(command.split())

    @staticmethod
    def load(self, params: TrainParams = TrainParams(), epoch: Optional[int] = None, device=None) -> Generator:
        if device is None:
            device = torch.device('cuda')

        if epoch is None:
            path = sorted(params.log_dir.glob("models/*.pth"))[-1]
        else:
            path = params.log_dir / "models/%.6d.pth" % epoch

        checkpoint = torch.load(path, map_location=lambda a, b: a)
        args = checkpoint["args"]

        net_ig = fastgan.Generator(
            ngf=args.ngf, nz=args.nz, im_size=args.im_size)
        net_ig.to(device)

        checkpoint['g'] = {
            k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])

        net_ig.eval()
        net_ig.to(device)

        return FastGAN(net_ig, net_ig.nz, device)

    def sample_z(self, n_samples: int = 1) -> torch.Tensor:
        return torch.randn(n_samples, self.model.nz).to(self.device)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)[0]