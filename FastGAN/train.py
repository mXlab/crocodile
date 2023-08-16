"""FastGAN train module."""
from dataclasses import dataclass, asdict
import random
from typing import Optional
from simple_parsing import parse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import mlflow
from mlflow.models import infer_signature

from crocodile.optimizer import AdamConfig, load_optimizer
from crocodile.trainer.base import TrainConfig, Trainer
from crocodile.utils import flatten_dict
from crocodile.dataset import LaurenceDataset


from FastGAN import lpips
from FastGAN.models import (
    weights_init,
    Discriminator,
    FastGANGenerator as Generator,
    GeneratorConfig,
    DiscriminatorConfig,
)
from FastGAN.operation import copy_G_params, load_params
from FastGAN.operation import ImageFolder, InfiniteSamplerWrapper
from FastGAN.diffaug import DiffAugment

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


def train_d(net: nn.Module, data: torch.Tensor, label: str = "real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = (
            F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
            + percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum()
            + percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum()
            + percept(
                rec_part,
                F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]),
            ).sum()
        )
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


@dataclass
class FastGANTrainConfig(TrainConfig):
    """FastGAN train config class."""

    total_iterations: int = 50000
    checkpoint: Optional[str] = None
    batch_size: int = 8
    im_size: int = 1024
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    optimizer_G: AdamConfig = AdamConfig(lr=0.0002, betas=(0.5, 0.999))
    optimizer_D: AdamConfig = AdamConfig(lr=0.0002, betas=(0.5, 0.999))
    dataloader_workers: int = 8
    img_save_interval: int = 1000
    model_save_interval: int = 5000
    num_test_samples: int = 8
    ema_beta: float = 0.001
    policy: str = "color,translation"
    remote_server_uri: str = "https://crocodile-gqhfy6c73a-uc.a.run.app"
    experiment_name: str = "fastgan"


class FastGANTrainer(Trainer):
    """FastGAN trainer class."""

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

        dataset = ImageFolder(root=data_root, transform=trans)

        self.dataloader = iter(
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=InfiniteSamplerWrapper(dataset),
                num_workers=config.dataloader_workers,
                pin_memory=True,
            )
        )

        # from model_s import Generator, Discriminator
        self.netG = Generator(im_size=config.im_size, config=config.generator)
        self.netG.apply(weights_init)

        self.netD = Discriminator(im_size=config.im_size, config=config.discriminator)
        self.netD.apply(weights_init)

        self.optimizerG = load_optimizer(self.netG.parameters(), config.optimizer_G)
        self.optimizerD = load_optimizer(self.netD.parameters(), config.optimizer_D)

        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)

    def train(self):
        mlflow.set_tracking_uri(self.config.remote_server_uri)
        mlflow.set_experiment(self.config.experiment_name)
        with mlflow.start_run():
            params = flatten_dict(asdict(self.config))
            mlflow.log_params(params)

            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

            self.netG.to(device)
            self.netD.to(device)

            avg_param_G = copy_G_params(self.netG)
            fixed_noise = self.netG.noise(self.config.num_test_samples).to(device)

            signature = infer_signature(
                fixed_noise.cpu().numpy(), self.netG(fixed_noise).detach().cpu().numpy()
            )

            current_iteration = 0
            if self.config.checkpoint is not None:
                ckpt = torch.load(self.config.checkpoint)
                self.netG.load_state_dict(ckpt["g"])
                self.netD.load_state_dict(ckpt["d"])
                avg_param_G = ckpt["g_ema"]
                self.optimizerG.load_state_dict(ckpt["opt_g"])
                self.optimizerD.load_state_dict(ckpt["opt_d"])
                current_iteration = int(
                    self.config.checkpoint.split("_")[-1].split(".")[0]
                )
                del ckpt

            for iteration in tqdm(
                range(current_iteration, self.config.total_iterations + 1)
            ):
                real_image = next(self.dataloader)
                real_image = real_image.to(device)
                current_batch_size = real_image.size(0)
                noise = self.netG.noise(current_batch_size).to(device)

                fake_images = self.netG(noise)

                real_image = DiffAugment(real_image, policy=self.config.policy)
                fake_images = [
                    DiffAugment(fake, policy=self.config.policy) for fake in fake_images
                ]

                ## 2. train Discriminator
                self.netD.zero_grad()

                err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
                    self.netD, real_image, label="real"
                )
                train_d(self.netD, [fi.detach() for fi in fake_images], label="fake")
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

                if iteration % 100 == 0:
                    print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, -err_g.item()))

                if iteration % self.config.img_save_interval == 0:
                    with torch.no_grad():
                        img = to_pil_image(
                            vutils.make_grid(
                                self.netG(fixed_noise)[0].add(1).mul(0.5),
                                nrow=4,
                            )
                        )
                        mlflow.log_image(img, f"images/iteration={iteration}.jpg")

                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    with torch.no_grad():
                        img = to_pil_image(
                            vutils.make_grid(
                                self.netG(fixed_noise)[0].add(1).mul(0.5),
                                nrow=4,
                            )
                        )
                        mlflow.log_image(img, f"images-ema/iteration={iteration}.jpg")
                    load_params(self.netG, backup_para)

                if (
                    iteration % self.config.model_save_interval == 0
                    or iteration == self.config.total_iterations
                ):
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    mlflow.pytorch.log_model(
                        self.netG,
                        f"models-ema/iteration={iteration}",
                        signature=signature,
                    )
                    load_params(self.netG, backup_para)
                    mlflow.pytorch.log_model(
                        self.netG, f"models/iteration={iteration}", signature=signature
                    )


if __name__ == "__main__":
    config = parse(FastGANTrainConfig)
    trainer = FastGANTrainer(config)
    trainer.train()
