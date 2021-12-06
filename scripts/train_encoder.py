from crocodile.executor import load_executor, ExecutorConfig, ExecutorCallable
from crocodile.encoder import load_encoder, EncoderParams
from crocodile.generator import load_from_path
from dataclasses import dataclass
from crocodile.dataset import LaurenceDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import torch.optim as optim
import torch
from simple_parsing import ArgumentParser
from typing import Optional
from tqdm import tqdm
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import register_decoding_fn
from crocodile.utils.logger import Logger
import os


@dataclass
class Params(Serializable):
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    encoder: EncoderParams = EncoderParams()
    lr: float = 1e-2
    num_epochs: int = 100
    log_dir: Path = Path("./results/encoder")
    name: str = "test_1"
    slurm_job_id: Optional[str] = os.environ.get('SLURM_JOB_ID')
    num_test_samples: int = 10

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


register_decoding_fn(Path, Path)


class EncoderTraining(ExecutorCallable):
    def __call__(self, args: Params, resume=False):
        device = torch.device('cuda')

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        trans = transforms.Compose(transform_list)

        generator = load_from_path(
            args.generator_path, args.epoch, device=device)
        args.dataset.resolution = generator.resolution

        dataset = LaurenceDataset(
            args.dataset, transform=trans, target_transform=transforms.ToTensor())

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True)

        encoder = load_encoder(args.encoder).build(
            dataset.seq_length*dataset.seq_dim, generator.latent_dim)
        encoder.to(device)

        optimizer = optim.Adam(encoder.parameters(), lr=args.lr)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img, label_ref, = iter(dataloader).next()
        img = img[:args.num_test_samples]
        label_ref = label_ref[:args.num_test_samples]
        logger.save_image("groundtruth", img)

        for epoch in range(args.num_epochs):
            loss_mean = 0
            for img, label, _ in tqdm(dataloader):
                optimizer.zero_grad()

                img = img.to(device)
                label = label.float().to(device)

                z = encoder(label)
                img_recons = generator(z)

                loss = ((img - img_recons)**2).view(len(img), -1).sum(-1).mean()
                loss.backward()

                optimizer.step()

                loss_mean += loss.detach().item()

            logger.add({"epoch": epoch, "loss_mean": loss_mean})

            with torch.no_grad():
                label = label_ref.float().to(device)
                z = encoder(label)
                img_recons = generator(z)

            logger.save_image(f"recons_{epoch:04d}", img_recons)
            logger.save_model("{epoch:04d}", encoder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(EncoderTraining(), args.params)
