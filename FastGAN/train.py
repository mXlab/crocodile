"""FastGAN train module."""
from dataclasses import dataclass, asdict
import random
import os
from pathlib import Path
from typing import Iterator, Literal, Optional
from simple_parsing import parse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import mlflow


from crocodile.optimizer import AdamConfig, load_optimizer
from crocodile.trainer import TrainConfig, Trainer
from crocodile.utils import flatten_dict
from crocodile.dataset import LaurenceDataset
from torchmetrics.image.fid import FrechetInceptionDistance


from . import lpips
from .models import (
    weights_init,
    Discriminator,
    FastGANGenerator as Generator,
    GeneratorConfig,
    DiscriminatorConfig,
)
from .operation import copy_G_params, load_params
from .operation import ImageFolder, InfiniteSamplerWrapper
from .diffaug import DiffAugment

# torch.backends.cudnn.benchmark = True


def crop_image_by_part(image: torch.Tensor, part: int):
    """Crop image by part"""
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


@dataclass
class FastGANTrainConfig(TrainConfig):
    """FastGAN train config class."""

    total_iterations: int = 50000
    checkpoint: Optional[str] = None
    batch_size: int = 8
    eval_batch_size: int = 16
    im_size: int = 1024
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    optimizer_G: AdamConfig = AdamConfig(lr=0.0002, betas=(0.5, 0.999))
    optimizer_D: AdamConfig = AdamConfig(lr=0.0002, betas=(0.5, 0.999))
    dataloader_workers: int = 8
    img_save_interval: int = 5000
    model_save_interval: int = 5000
    eval_save_interval: int = 5000
    num_valid_samples: int = 2000
    num_test_samples: int = 8
    ema_beta: float = 0.001
    policy: str = "color,translation"
    remote_server_uri: str = "https://crocodile-gqhfy6c73a-uc.a.run.app"
    experiment_name: str = "fastgan"
    log_dir: Optional[Path] = None


class FastGANTrainer(Trainer):
    """FastGAN trainer class."""

    device: Literal["cuda", "cpu"]
    dataloader: Iterator
    fid_loader: DataLoader

    def __init__(self, config: FastGANTrainConfig):
        self.config = config

        dataset = LaurenceDataset(config.dataset)
        data_root = dataset.get_path()

        transform_list = [
            transforms.Resize((int(config.im_size), int(config.im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        trans = transforms.Compose(transform_list)

        self.dataset = ImageFolder(root=data_root, transform=trans)

        # from model_s import Generator, Discriminator
        self.netG = Generator(im_size=config.im_size, config=config.generator)
        self.netG.apply(weights_init)

        self.netD = Discriminator(im_size=config.im_size, config=config.discriminator)
        self.netD.apply(weights_init)

        self.optimizerG = load_optimizer(self.netG.parameters(), config.optimizer_G)
        self.optimizerD = load_optimizer(self.netD.parameters(), config.optimizer_D)

        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=False)

        self.fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)

    def init_fid(self):
        with torch.no_grad():
            for real_image in tqdm(self.fid_loader):
                real_image = real_image.to(self.device)
                self.fid.update(self.netG.unormalize(real_image), real=True)

    def compute_fid(self, netG: Generator):
        """Compute FID score."""
        self.fid.reset()
        with torch.no_grad():
            for _ in range(
                self.config.num_valid_samples // self.config.eval_batch_size
            ):
                noise = netG.noise(self.config.eval_batch_size).to(self.device)
                fake_images = netG.generate(noise)
                self.fid.update(fake_images, real=False)

        return self.fid.compute()

    def train_d(self, data, label: str = "real"):
        """Train function of discriminator"""
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = self.netD(data, label, part=part)
            err = (
                F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
                + self.percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum()
                + self.percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum()
                + self.percept(
                    rec_part,
                    F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]),
                ).sum()
            )
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = self.netD(data, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    def train(self):
        print("Loading device...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device}")

        self.dataloader = iter(
            DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                sampler=InfiniteSamplerWrapper(self.dataset),
                num_workers=self.config.dataloader_workers,
                pin_memory=True,
            )
        )

        self.fid_loader = DataLoader(
            self.dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_workers,
        )

        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=torch.cuda.is_available()
        )

        self.netG.to(self.device)
        self.netD.to(self.device)

        avg_param_G = copy_G_params(self.netG)
        fixed_noise = self.netG.noise(self.config.num_test_samples).to(self.device)

        print("Initializing fid...")
        self.fid.to(self.device)
        self.init_fid()

        current_iteration = 0
        if self.config.checkpoint is not None:
            ckpt = torch.load(self.config.checkpoint)
            self.netG.load_state_dict(ckpt["g"])
            self.netD.load_state_dict(ckpt["d"])
            avg_param_G = ckpt["g_ema"]
            self.optimizerG.load_state_dict(ckpt["opt_g"])
            self.optimizerD.load_state_dict(ckpt["opt_d"])
            current_iteration = int(self.config.checkpoint.split("_")[-1].split(".")[0])
            del ckpt

        mlflow.set_tracking_uri(self.config.remote_server_uri)
        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run() as run:
            log_dir = self.config.log_dir
            if log_dir is None:
                scratch_dir = os.environ.get("SCRATCH")
                if scratch_dir is not None:
                    log_dir = Path(scratch_dir) / "crocodile" / "logs"
                else:
                    log_dir = Path("./logs")

            exp_dir = log_dir / str(run.info.run_id)
            model_dir = exp_dir / "models"
            model_ema_dir = exp_dir / "models-ema"
            image_dir = exp_dir / "images"
            image_ema_dir = exp_dir / "images-ema"

            model_dir.mkdir(parents=True, exist_ok=True)
            model_ema_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_ema_dir.mkdir(parents=True, exist_ok=True)

            print(f"Saving experiment to {exp_dir}...")
            params = flatten_dict(asdict(self.config))
            mlflow.log_params(params)
            self.config.save(exp_dir / "config.yaml")
            fid = float("nan")
            fid_ema = float("nan")
            for iteration in tqdm(
                range(current_iteration, self.config.total_iterations + 1)
            ):
                real_image = next(self.dataloader)
                real_image = real_image.to(self.device)
                current_batch_size = real_image.size(0)
                noise = self.netG.noise(current_batch_size).to(self.device)

                fake_images = self.netG.multires(noise)

                real_image = DiffAugment(real_image, policy=self.config.policy)
                fake_images = [
                    DiffAugment(fake, policy=self.config.policy) for fake in fake_images
                ]

                ## 2. train Discriminator
                self.netD.zero_grad()

                err_dr, rec_img_all, rec_img_small, rec_img_part = self.train_d(
                    real_image, label="real"
                )
                self.train_d([fi.detach() for fi in fake_images], label="fake")
                self.optimizerD.step()

                ## 3. train Generator
                self.netG.zero_grad()
                pred_g = self.netD(fake_images, "fake")
                err_g = -pred_g.mean()

                err_g.backward()
                self.optimizerG.step()

                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(1 - self.config.ema_beta).add_(
                        self.config.ema_beta * p.data
                    )

                if (iteration + 1) % 100 == 0:
                    print(
                        f"GAN: loss d: {err_dr:.5f}, loss g: {-err_g.item():.5f}, fid: {fid:.2f}, fid_ema: {fid_ema:.2f}"
                    )

                if (iteration + 1) % self.config.eval_save_interval == 0:
                    fid = self.compute_fid(self.netG)
                    mlflow.log_metric("fid", fid.item(), step=iteration)

                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    fid_ema = self.compute_fid(self.netG)
                    mlflow.log_metric("fid-ema", fid_ema.item(), step=iteration)
                    load_params(self.netG, backup_para)

                if (iteration + 1) % self.config.img_save_interval == 0:
                    with torch.no_grad():
                        img = to_pil_image(
                            vutils.make_grid(
                                self.netG.generate(fixed_noise),
                                nrow=4,
                            )
                        )
                        mlflow.log_image(img, f"images/iteration={iteration}.jpg")
                        img.save(image_dir / f"{iteration}.jpg")

                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    with torch.no_grad():
                        img = to_pil_image(
                            vutils.make_grid(
                                self.netG.generate(fixed_noise),
                                nrow=4,
                            )
                        )
                        mlflow.log_image(img, f"images-ema/iteration={iteration}.jpg")
                        img.save(image_ema_dir / f"{iteration}.jpg")
                    load_params(self.netG, backup_para)

                if (
                    iteration % self.config.model_save_interval == 0
                    or iteration == self.config.total_iterations
                ):
                    torch.save(
                        {"g": self.netG.state_dict(), "config": self.config},
                        model_dir / f"{iteration}.pth",
                    )
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    torch.save(
                        {"g": self.netG.state_dict(), "config": self.config},
                        model_ema_dir / f"{iteration}.pth",
                    )
                    load_params(self.netG, backup_para)

    @staticmethod
    def load_generator(config: FastGANTrainConfig, params):
        generator = Generator(im_size=config.im_size, config=config.generator)
        generator.load_state_dict(params)
        return generator


if __name__ == "__main__":
    train_config = parse(FastGANTrainConfig)
    trainer = FastGANTrainer(train_config)
    trainer.train()
