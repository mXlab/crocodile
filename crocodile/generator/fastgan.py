from .base import Model
import os
from crocodile.dataset import LaurenceDataset
import subprocess
import torch
from FastGAN.models import Generator


class FastGAN(Model):
    @staticmethod
    def train(args):
        print("Loading dataset...")
        dataset = LaurenceDataset(args.dataset)

        data_path = dataset.get_path()
        os.chdir('pygan/models/FastGAN')
        command = "python train.py --path=%s --batch_size %i --im_size %i" % (
            data_path, args.train.batch_size, dataset.resolution)
        print("Running: %s" % command)
        subprocess.run(command.split())

    @staticmethod
    def load_model(path: str, device=None):
        if device is None:
            device = torch.device('cuda')

        checkpoint = torch.load(path, map_location=lambda a,b: a)
        args = checkpoint["args"]

        net_ig = Generator(ngf=args.ngf, nz=args.nz, im_size=args.im_size)
        net_ig.to(device)

        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])

        net_ig.eval()
        net_ig.to(device)

        return FastGAN(net_ig, net_ig.nz, device)

    def sample_z(self, n_samples=1):
        return torch.randn(n_samples, self.model.nz).to(self.device)

    def __call__(self, z):
        return self.model(z)[0]
