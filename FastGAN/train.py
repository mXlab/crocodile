from dataclasses import dataclass
import random
from pathlib import Path
from typing import Optional
from simple_parsing import parse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from tqdm import tqdm

from crocodile.optimizer import AdamConfig, load_optimizer


from . import lpips
from .models import (
    weights_init,
    Discriminator,
    Generator,
    GeneratorConfig,
    DiscriminatorConfig,
)
from .operation import copy_G_params, load_params, get_dir
from .operation import ImageFolder, InfiniteSamplerWrapper
from .diffaug import DiffAugment

percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=True)


# torch.backends.cudnn.benchmark = True


def crop_image_by_part(image: torch.Tensor, part: int):
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
class FastGANTrainConfig:
    data_root: Path = Path("../lmdbs/art_landscape_1k")
    total_iterations: int = 50000
    checkpoint: Optional[str] = None
    batch_size: int = 8
    im_size: int = 1024
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    optimizer_G: AdamConfig = AdamConfig(lr=0.0002, betas=(0.5, 0.999))
    optimizer_D: AdamConfig = AdamConfig(lr=0.0002, betas=(0.5, 0.999))
    use_cuda: bool = True
    multi_gpu: bool = True
    dataloader_workers: int = 8
    img_save_interval: int = 1000
    model_save_interval: int = 5000
    num_test_samples: int = 8
    ema_beta: float = 0.001
    policy: str = "color,translation"
    


def train(config: FastGANTrainConfig):
    saved_model_folder, saved_image_folder = get_dir(config)

    device = torch.device("cpu")
    if config.use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
        transforms.Resize((int(config.im_size), int(config.im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    trans = transforms.Compose(transform_list)

    dataset = ImageFolder(root=config.data_root, transform=trans)

    dataloader = iter(
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
    netG = Generator(im_size=config.im_size, config=config.generator)
    netG.apply(weights_init)

    netD = Discriminator(im_size=config.im_size, config=config.discriminator)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = netG.noise(config.num_test_samples).to(device)

    optimizerG = load_optimizer(netG.parameters(), config.optimizer_G)
    optimizerD = load_optimizer(netD.parameters(), config.optimizer_D)

    current_iteration = 0
    if config.checkpoint is not None:
        ckpt = torch.load(config.checkpoint)
        netG.load_state_dict(ckpt["g"])
        netD.load_state_dict(ckpt["d"])
        avg_param_G = ckpt["g_ema"]
        optimizerG.load_state_dict(ckpt["opt_g"])
        optimizerD.load_state_dict(ckpt["opt_d"])
        current_iteration = int(config.checkpoint.split("_")[-1].split(".")[0])
        del ckpt

    if config.multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    for iteration in tqdm(range(current_iteration, config.total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = netG.noise(current_batch_size).to(device)

        fake_images = netG(noise)

        real_image = DiffAugment(real_image, policy=config.policy)
        fake_images = [DiffAugment(fake, policy=config.policy) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(
            netD, real_image, label="real"
        )
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(1-config.ema_beta).add_(config.ema_beta * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, -err_g.item()))

        if iteration % config.img_save_interval == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(
                    netG(fixed_noise)[0].add(1).mul(0.5),
                    saved_image_folder + "/%d.jpg" % iteration,
                    nrow=4,
                )
            load_params(netG, backup_para)

        if (
            iteration % config.model_save_interval == 0
            or iteration == config.total_iterations
        ):
            torch.save(
                {
                    "g": netG.state_dict(),
                    "g_ema": avg_param_G,
                    "d": netD.state_dict(),
                    "config": config,
                },
                saved_model_folder + "/%.6d.pth" % iteration,
            )


if __name__ == "__main__":
    config = parse(FastGANTrainConfig)

    train(config)
