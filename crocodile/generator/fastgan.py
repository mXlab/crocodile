from .generator import Generator, TrainParams
from crocodile.dataset import LaurenceDataset
import subprocess
import torch
import FastGAN.models as models
from typing import Optional, Dict, Any


class FastGANModelArgs:
    ngf: int
    nz: int
    im_size: int


class FastGAN(Generator):
    @classmethod
    def prepare(cls, params: TrainParams = TrainParams()):
        print("Loading dataset...")
        LaurenceDataset.download(params.dataset)

    @classmethod
    def train(cls, params: TrainParams = TrainParams()):
        dataset = LaurenceDataset(params.dataset)
        data_path = dataset.get_path()
        cls.set_dir(params)
        command = (
            "python -m FastGAN.train --outdir %s --path=%s --batch_size %i --im_size %i"
            % (params.log_dir, data_path, params.batch_size, dataset.resolution)
        )
        print("Running: %s" % command)
        subprocess.run(command.split())

    @staticmethod
    def load(
        params: TrainParams = TrainParams(), epoch: Optional[int] = None, device=None
    ) -> Generator:
        if device is None:
            device = torch.device("cuda")

        if epoch is None:
            path = sorted(params.log_dir.glob("models/*.pth"))[-1]
        else:
            path = params.log_dir / f"models/{epoch:.6d}.pth"

        checkpoint = torch.load(path, map_location=lambda a, b: a)
        args: FastGANModelArgs = checkpoint["args"]

        net_ig = models.Generator(ngf=args.ngf, nz=args.nz, im_size=args.im_size)
        net_ig.to(device)

        # operation.load_params(net_ig, checkpoint['g_ema'])
        generator: Dict[str, Any] = checkpoint["g"]
        checkpoint["g"] = {k.replace("module.", ""): v for k, v in generator.items()}
        net_ig.load_state_dict(checkpoint["g"])

        net_ig.eval()
        net_ig.to(device)

        return FastGAN(net_ig, args.nz, args.im_size, device)

    def sample_z(self, n_samples: int = 1) -> torch.Tensor:
        if self.latent_dim is None:
            raise ValueError(
                "Cannot sample from a generator with an unknown latent dimension."
            )
        return torch.randn(n_samples, self.latent_dim).to(self.device)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Cannot call a generator without a model.")
        return self.model(z)[0]
