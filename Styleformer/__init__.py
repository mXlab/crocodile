from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import numpy as np
import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl

from crocodile.generator import GeneratorConfig
from crocodile.optimizer import load_optimizer, AdamConfig
from crocodile.utils.conditional_fields import WithConditionalFields, conditional_field


from .torch_utils import misc, training_stats
from .training.networks_Generator import (
    Generator,
    GeneratorConfig as StyleformerGeneratorConfig,
)
from .training.networks_Discriminator import Discriminator, DiscriminatorConfig
from .training.loss import StyleGAN2Loss, StyleGAN2LossConfig
from .training.augment import AugmentPipe, AugmentPipeConfig


@dataclass
class Phase:
    name: str
    module: nn.Module
    opt: optim.Optimizer
    interval: int
    start_event: Optional[Any] = None
    end_event: Optional[Any] = None


def build_phases(
    name: str,
    module: nn.Module,
    optimizer_config: AdamConfig,
    reg_interval: Optional[int],
):
    phases: List[Phase] = []
    if reg_interval is None:
        opt = load_optimizer(module.parameters(), optimizer_config)
        phases.append(Phase(name=name + "both", module=module, opt=opt, interval=1))
    else:  # Lazy regularization.
        mb_ratio = reg_interval / (reg_interval + 1)
        optimizer_config.lr = optimizer_config.lr * mb_ratio
        optimizer_config.betas = tuple(
            beta**mb_ratio for beta in optimizer_config.betas
        )
        opt = load_optimizer(module.parameters(), optimizer_config)
        phases.append(Phase(name=name + "main", module=module, opt=opt, interval=1))
        phases.append(
            Phase(name=name + "reg", module=module, opt=opt, interval=reg_interval)
        )
    return phases


@dataclass
class EMAStrategyConfig:
    ema_rampup: Optional[float] = None
    ema_kimg: float = 2


class EMAStrategy:
    def __init__(self, config: EMAStrategyConfig) -> None:
        self.ema_rampup = config.ema_rampup
        self.ema_kimg = config.ema_kimg

    def __call__(self, p_ema, p, num_averaged):
        ema_nimg = self.ema_kimg * 1000
        if self.ema_rampup is not None:
            ema_nimg = min(ema_nimg, num_averaged * self.ema_rampup)
        ema_beta = 0.5 ** (1 / max(ema_nimg, 1e-8))
        return p.lerp(p_ema, ema_beta)


@dataclass
class StyleformerGANConfig(GeneratorConfig, WithConditionalFields):
    c_dim: int
    img_resolution: int
    img_channels: int
    generator: StyleformerGeneratorConfig = StyleformerGeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
    batch_gpu: int = 4
    G_reg_interval: int = 4
    D_reg_interval: int = 16
    batch_size: int = 4
    num_gpus: int = 1
    ema: EMAStrategyConfig = EMAStrategyConfig()
    generator_optimizer: AdamConfig = AdamConfig()
    discriminator_optimizer: AdamConfig = AdamConfig()
    loss: StyleGAN2LossConfig = StyleGAN2LossConfig()
    augment: Optional[AugmentPipeConfig] = None
    augment_p: float = 0
    ada_target: Optional[float] = None
    ada_interval: int = 4
    ada_kimg: int = 500


class StyleformerGAN(pl.LightningModule):
    def __init__(self, config: StyleformerGANConfig) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.config = config

        self.G = Generator(
            config.c_dim,
            config.img_resolution,
            config.img_channels,
            config=config.generator,
        )
        self.D = Discriminator(
            config.c_dim,
            config.img_resolution,
            config.img_channels,
            config=config.discriminator,
        )

        self.phases: List[Phase] = []
        self.phases += build_phases(
            "G", self.G, config.generator_optimizer, config.G_reg_interval
        )
        self.phases += build_phases(
            "D", self.D, config.discriminator_optimizer, config.D_reg_interval
        )

        self.ema_G = torch.optim.swa_utils.AveragedModel(
            self.G, avg_fn=EMAStrategy(config.ema)
        )

        self.augment_pipe = None
        self.ada_stats = None
        if (config.augment is not None) and (
            config.augment_p > 0 or config.ada_target is not None
        ):
            self.augment_pipe = AugmentPipe(config.augment)
            self.augment_pipe.p.copy_(torch.as_tensor(config.augment_p))
            if config.ada_target is not None:
                self.ada_stats = training_stats.Collector(regex="Loss/signs/real")

    def setup(self):
        for phase in self.phases:
            if self.global_rank == 0:
                phase.start_event = torch.cuda.Event(enable_timing=True)
                phase.end_event = torch.cuda.Event(enable_timing=True)
        self.loss = StyleGAN2Loss(
            self.G.mapping, self.G.synthesis, self.D, self.config.loss
        )  # subclass of training.loss.Loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        phase_real_img, phase_real_c = batch
        phase_real_img = (
            phase_real_img.to(self.device).to(torch.float32) / 127.5 - 1
        ).split(self.config.batch_gpu)
        phase_real_c = phase_real_c.to(self.device).split(self.config.batch_gpu)
        all_gen_z = torch.randn(
            [len(self.phases) * self.config.batch_size, self.G.z_dim],
            device=self.device,
        )
        all_gen_z = [
            phase_gen_z.split(self.config.batch_gpu)
            for phase_gen_z in all_gen_z.split(self.config.batch_size)
        ]
        all_gen_c = [
            training_set.get_label(np.random.randint(len(training_set)))
            for _ in range(len(self.phases) * self.config.batch_size)
        ]
        all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(self.device)
        all_gen_c = [
            phase_gen_c.split(self.config.batch_gpu)
            for phase_gen_c in all_gen_c.split(self.config.batch_size)
        ]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(self.phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(self.device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(
                zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)
            ):
                sync = (
                    round_idx
                    == self.config.batch_size
                    // (self.config.batch_gpu * self.config.num_gpus)
                    - 1
                )
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real_img=real_img,
                    real_c=real_c,
                    gen_z=gen_z,
                    gen_c=gen_c,
                    sync=sync,
                    gain=gain,
                )

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + "_opt"):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(
                            param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                        )
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(self.device))

        # Update G_ema.
        with torch.autograd.profiler.record_function("Gema"):
            self.ema_G.update_parameters(self.G)

        if (self.ada_stats is not None) and (batch_idx % self.config.ada_interval == 0):
            self.ada_stats.update()
            adjust = (
                np.sign(self.ada_stats["Loss/signs/real"] - self.config.ada_target)
                * (self.config.batch_size * self.config.ada_interval)
                / (self.config.ada_kimg * 1000)
            )
            self.augment_pipe.p.copy_(
                (self.augment_pipe.p + adjust).max(misc.constant(0, device=self.device))
            )
