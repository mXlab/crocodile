from dataclasses import dataclass
import random

from simple_parsing import parse

from crocodile.trainer import Trainer, TrainerConfig, FastGANConfig
from crocodile.executor import load_executor, SlurmConfig


@dataclass
class Config:
    num_runs: int = 10
    executor: SlurmConfig = SlurmConfig(gpus_per_node=1, time_in_min=2880)
    trainer: TrainerConfig = TrainerConfig()


def sample_fastgan_config():
    ema_momentum = random.choice([1e-3, 2e-3, 5e-4])
    nbeta1 = random.choice([0.5, 0.9, 0.99])
    nbeta2 = random.choice([0.999, 0.9999, 0.99999])
    nf = random.choice([32, 64, 128])
    nz = random.choice([256, 512, 1024])
    nlr = random.choice([5e-4, 2e-4, 1e-4])
    return FastGANConfig(
        ema_momentum=ema_momentum,
        nbeta1=nbeta1,
        nbeta2=nbeta2,
        ngf=nf,
        ndf=nf,
        nz=nz,
        nlr=nlr,
    )


if __name__ == "__main__":
    config = parse(Config)
    print("Initializing executor...")
    executor = load_executor(config.executor)
    for i in range(config.num_runs):
        config.trainer.generator = sample_fastgan_config()
        trainer = Trainer(config.trainer)
        executor(trainer.train)  # TODO: fix type error
