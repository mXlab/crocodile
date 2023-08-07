from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import utils as vutils
import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance
import mlflow

from crocodile.generator import GeneratorConfig

from . import lpips
from .models import (
    weights_init,
    Discriminator as FastGANDiscriminator,
    FastGANGenerator,
)
from .diffaug import DiffAugment


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def normalize(x: torch.Tensor):
    return (x + 1) / 2


@dataclass
class FastGANConfig(GeneratorConfig):
    ngf: int = 64
    ndf: int = 64
    nz: int = 256
    nlr: float = 0.0002
    nbeta1: float = 0.5
    nbeta2: float = 0.999
    ema_momentum: float = 0.001
    policy: str = "color,translation"


class FastGAN(pl.LightningModule):
    def __init__(self, config: FastGANConfig, im_size: int) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.config = config

        self.netG = self.load_generator(config, im_size)
        self.netG.apply(weights_init)

        self.netD = FastGANDiscriminator(ndf=config.ndf, im_size=im_size)
        self.netD.apply(weights_init)

        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=False)

        ema_avg = (
            lambda averaged_model_parameter, model_parameter, num_averaged: (
                1 - self.config.ema_momentum
            )
            * averaged_model_parameter
            + self.config.ema_momentum * model_parameter
        )
        self.emaG = torch.optim.swa_utils.AveragedModel(self.netG, avg_fn=ema_avg)

        self.fid = FrechetInceptionDistance(normalize=True)
        self.fid_ema = FrechetInceptionDistance(normalize=True)

    @property
    def latent_dim(self):
        return self.config.nz

    def setup(self, stage):
        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=torch.cuda.is_available()
        )
        self.fixed_noise = torch.FloatTensor(16, self.latent_dim).normal_(0, 1)

    def configure_optimizers(self):
        optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=self.config.nlr,
            betas=(self.config.nbeta1, self.config.nbeta2),
        )
        optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.config.nlr,
            betas=(self.config.nbeta1, self.config.nbeta2),
        )
        return optimizerG, optimizerD

    def training_step(self, batch, batch_idx: int):
        optimizerG, optimizerD = self.optimizers()  # Todo: Fix error
        real_image = batch["image"].to(self.device)
        current_batch_size = real_image.size(0)
        noise = (
            torch.Tensor(current_batch_size, self.latent_dim)
            .normal_(0, 1)
            .to(self.device)
        )

        fake_images = self.netG(noise)

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
        optimizerD.step()

        ## 3. train Generator
        self.netG.zero_grad()
        pred_g = self.netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        self.emaG.update_parameters(self.netG)

        self.log("err_g", err_g)
        self.log("err_d", err_dr)

    def validation_step(self, batch, batch_idx: int):
        # this is the test loop
        self.train()
        real_image = normalize(batch["image"].to(self.device))
        noise = (
            torch.Tensor(len(real_image), self.latent_dim).normal_(0, 1).to(self.device)
        )

        fake_images = normalize(self.netG(noise)[0])

        self.fid.update(fake_images, real=False)
        self.fid.update(real_image, real=True)

        fake_images = normalize(self.emaG(noise)[0])

        self.fid_ema.update(fake_images, real=False)
        self.fid_ema.update(real_image, real=True)

        self.log("fid", self.fid, on_step=False, on_epoch=True)
        self.log("fid_ema", self.fid_ema, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.train()
        step = self.trainer.global_step
        epoch = self.trainer.current_epoch
        fixed_noise = self.fixed_noise.to(self.device)
        if self.trainer.log_dir is not None:
            img = np.moveaxis(
                vutils.make_grid(
                    self.netG(fixed_noise)[0].add(1).mul(0.5),
                    nrow=4,
                )
                .cpu()
                .numpy(),
                0,
                -1,
            )
            mlflow.log_image(img, f"images/epoch={epoch}-step={step}.jpg")

            img = np.moveaxis(
                vutils.make_grid(self.emaG(fixed_noise)[0].add(1).mul(0.5), nrow=4)
                .cpu()
                .numpy(),
                0,
                -1,
            )
            mlflow.log_image(img, f"images-ema/epoch={epoch}-step={step}.jpg")

    def train_d(self, data, label="real"):
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

    @staticmethod
    def load_generator(config: FastGANConfig, im_size: int):
        netG = FastGANGenerator(ngf=config.ngf, nz=config.nz, im_size=im_size)
        return netG

    def generate(self, noise: torch.Tensor):
        return self.netG.generate(noise)
