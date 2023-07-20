from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from torchmetrics.image.kid import KernelInceptionDistance

from crocodile.generator import GeneratorConfig

from . import lpips
from .models import weights_init, Discriminator, Generator
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


@dataclass
class FastGANConfig(GeneratorConfig):
    ngf: int = 64
    ndf: int = 64
    nz: int = 256
    nlr: float = 0.0002
    nbeta1: float = 0.5
    nbeta2: float = 0.999
    ema_momentum: float = 0.001
    im_size: int = 512
    policy: str = "color,translation"


class FastGAN(pl.LightningModule):
    def __init__(self, config: FastGANConfig) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.config = config

        self.netG = Generator(ngf=config.ngf, nz=config.nz, im_size=config.im_size)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ndf=config.ndf, im_size=config.im_size)
        self.netD.apply(weights_init)

        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=torch.cuda.is_available()
        )

        ema_avg = (
            lambda averaged_model_parameter, model_parameter, num_averaged: (
                1 - self.config.ema_momentum
            )
            * averaged_model_parameter
            + self.config.ema_momentum * model_parameter
        )
        self.emaG = torch.optim.swa_utils.AveragedModel(self.netG, avg_fn=ema_avg)

        self.fid = KernelInceptionDistance(subsets=3, subset_size=100)
        self.fid_ema = KernelInceptionDistance(subsets=3, subset_size=100)

    @property
    def latent_dim(self):
        return self.config.nz

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

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        optimizerG, optimizerD = self.optimizers()  # Todo: Fix error
        real_image = batch.to(self.device)
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

        self.log("err_g", err_g, on_step=True, on_epoch=False)
        self.log("err_d", err_dr, on_step=True, on_epoch=False)

    def valid_step(self, batch: torch.Tensor, batch_idx: int):
        # this is the test loop
        real_image = batch.to(self.device)
        noise = (
            torch.Tensor(len(real_image), self.latent_dim).normal_(0, 1).to(self.device)
        )

        fake_images = self.netG(noise)[0]

        self.fid(fake_images, real=False)
        self.fid(real_image, real=True)

        fake_images = self.emaG(noise)[0]
        self.fid_ema(fake_images, real=False)
        self.fid_ema(real_image, real=True)

    def on_valid_epoch_end(self) -> None:
        mean, std = self.fid.compute()
        self.log("fid_mean", mean)
        self.log("fid_std", std)

        mean, std = self.fid_ema.compute()
        self.log("fid_ema_mean", mean)
        self.log("fid_ema_std", std)

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

    def generate(self, z: torch.Tensor):
        z = z.to(self.device)
        return self.netG(z)[0]
