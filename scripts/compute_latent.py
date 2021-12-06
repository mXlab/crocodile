from crocodile.executor import load_executor, ExecutorConfig, ExecutorCallable
from crocodile.generator import load_from_path
from crocodile.dataset import LaurenceDataset, LatentDataset
from crocodile.utils.optim import OptimizerArgs, load_optimizer
from crocodile.utils.loss import LossParams, load_loss
from crocodile.utils.logger import Logger
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from pathlib import Path
import torch
from simple_parsing import ArgumentParser
from typing import Optional
from tqdm import tqdm
from simple_parsing.helpers import Serializable
from simple_parsing.helpers.serialization import register_decoding_fn
import os


@dataclass
class Params(Serializable):
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    optimizer: OptimizerArgs = OptimizerArgs()
    loss: LossParams = LossParams()
    num_epochs: int = 100
    log_dir: Path = Path("./results/latent")
    name: str = "test_1"
    num_test_samples: int = 10
    slurm_job_id: Optional[str] = os.environ.get('SLURM_JOB_ID')
    debug: bool = False

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


register_decoding_fn(Path, Path)


class ComputeLatent(ExecutorCallable):
    def __call__(self, args: Params, resume=False):
        torch.manual_seed(1234)
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
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        latent_dataset = LatentDataset(len(dataset), dim=generator.latent_dim)

        optimizer = load_optimizer(latent_dataset.parameters(), args.optimizer)

        loss_fn = load_loss(args.loss.loss, args.loss)
        loss_eval = load_loss()

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img_ref, _, index_ref = iter(dataloader).next()
        img_ref = img_ref[:args.num_test_samples]
        index_ref = index_ref[:args.num_test_samples]
        logger.save_image("groundtruth", img_ref)

        for epoch in range(args.num_epochs):
            loss_mean = 0
            for img, _, index in tqdm(dataloader):
                optimizer.zero_grad()

                if args.debug:
                    img = img_ref
                    index = index_ref

                img = img.to(device)
                z = latent_dataset[index]
                z = z.to(device)
                z.requires_grad_()

                img_recons = generator(z)

                loss = loss_fn(img, img_recons)
                loss.backward()

                optimizer.step()

                loss_mean += loss.detach().item()

                if args.debug:
                    break

            with torch.no_grad():
                z = latent_dataset[index_ref].to(device)
                img_recons = generator(z)
                loss = loss_eval(img_ref, img_recons).detach().item()

            logger.save_image(f"recons_{epoch:04d}", img_recons)
            logger.save("latent", latent_dataset)
            logger.add(
                {"epoch": epoch, "train_loss": loss_mean, "eval_loss": loss})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(ComputeLatent(), args.params)
