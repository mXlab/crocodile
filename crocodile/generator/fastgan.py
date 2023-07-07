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
        raise NotImplementedError()
        print("Loading dataset...")
        LaurenceDataset.download(params.dataset)
        data_path = params.dataset.get_dataset_path()
        print("Running: %s" % command)
        subprocess.run(command.split())

    @classmethod
    def train(cls, params: TrainParams = TrainParams()):
        raise NotImplementedError()
        dataset = LaurenceDataset(params.dataset)
        data_path = params.dataset.get_dataset_path()
        db_path = params.get_db_path()
        command = f"python -m FastGAN.train --outdir {params.log_dir} --path {data_path} --batch_size {params.batch_size} --im_size {dataset.resolution} --db_path {db_path}"
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
