from dataclasses import dataclass
import os
from typing import Optional
from simple_parsing import Serializable, parse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from uuid import uuid4
from pathlib import Path
from ignite.engine import Engine, Events
from ignite.handlers import EMAHandler, Checkpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import FID

import random

from crocodile.db import load_db

from FastGAN.models import weights_init, Discriminator, Generator
from FastGAN.operation import ImageFolder
from FastGAN.diffaug import DiffAugment

policy = "color,translation"
from FastGAN import lpips

# torch.backends.cudnn.benchmark = True


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
class TrainerParams(Serializable):
    db_path: str
    log_dir: Path = Path("./logs")
    data_root: Path = Path("./data/laurence/512")
    name: str = "fastgan-default"
    use_gpu: bool = True
    ngf: int = 64
    ndf: int = 64
    nz: int = 256
    im_size: int = 512
    nlr: float = 0.0002
    nbeta1: float = 0.5
    nbeta2: float = 0.999
    batch_size: int = 8
    _dataloader_workers: Optional[int] = None
    num_epochs: int = 1000
    logging_interval: int = 100
    saving_interval: int = 5000
    checkpoint_interval: int = 1000
    ema_momentum: float = 0.001
    num_test_samples: int = 10000
    prepare_only: bool = False

    @property
    def dataloader_workers(self):
        if self._dataloader_workers is None:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                return 0
            else:
                return cpu_count
        else:
            return self._dataloader_workers


class Trainer:
    def __init__(self, params: TrainerParams):
        self.params = params

        if params.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.netG = Generator(ngf=params.ngf, nz=params.nz, im_size=params.im_size)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ndf=params.ndf, im_size=params.im_size)
        self.netD.apply(weights_init)

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=params.nlr, betas=(params.nbeta1, params.nbeta2)
        )
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=params.nlr, betas=(params.nbeta1, params.nbeta2)
        )

        self.engine = Engine(self.train_step)
        self.ema_handler = EMAHandler(self.netG, momentum=params.ema_momentum)
        self.ema_handler.attach(
            self.engine, name="ema", event=Events.ITERATION_COMPLETED(every=1)
        )

        self.fid_metric = FID()
        self.fid_metric_ema = FID()

        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=torch.cuda.is_available()
        )

        if params.prepare_only:
            exit(0)

        exp_id = uuid4().hex
        path = Path(params.log_dir) / exp_id
        db = load_db(params.db_path)
        self.experiment = db.create_experiment(params.name, exp_id, path, params)

    def get_engine(self):
        return self.engine

    @property
    def nz(self):
        return self.params.nz

    def train_step(self, engine, batch):
        real_image = batch.to(self.device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, self.nz).normal_(0, 1).to(self.device)

        fake_images = self.netG(noise)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]

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

        return {
            "err_dr": err_dr,
            "err_g": err_g.item(),
        }

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

    def evaluate(self, test_loader):
        self.fid_metric.reset()
        self.fid_metric_ema.reset()
        with torch.no_grad():
            for batch in test_loader:
                real_image = batch.to(self.device)
                noise = (
                    torch.Tensor(len(real_image), self.nz).normal_(0, 1).to(self.device)
                )

                fake_images = self.netG(noise)[0]
                fake_images_ema = self.ema_handler.model(noise)[0]

                self.fid_metric.update((fake_images, real_image))
                self.fid_metric_ema.update((fake_images_ema, real_image))

        return self.fid_metric.compute(), self.fid_metric_ema.compute()

    def save(self, test_loader):
        fid, fid_ema = self.evaluate(test_loader)
        print(
            f"Epoch[{self.engine.state.epoch}], Iter[{self.engine.state.iteration}], fid: {fid}, fid_ema: {fid_ema}"
        )
        self.experiment.save_model(
            name="default",
            model_type="fastgan",
            model=self.netG,
            iteration=self.engine.state.iteration,
            fid=fid,
        )
        self.experiment.save_model(
            name="ema",
            model_type="fastgan",
            model=self.ema_handler.model,
            iteration=self.engine.state.iteration,
            fid=fid_ema,
        )

    def get_checkpointer(self):
        checkpointer = Checkpoint(
            to_save={
                "G": self.netG,
                "G_ema": self.ema_handler.model,
                "G_opt": self.optimizerG,
                "D_opt": self.optimizerD,
                "engine": self.engine,
            },
            save_handler=self.experiment.get_root_dir() / "checkpoint",
            n_saved=1,
        )
        return checkpointer

    def start(self, train_loader, num_epochs: int):
        self.experiment.start()
        self.engine.run(train_loader, max_epochs=num_epochs)

    def end(self):
        self.experiment.end()


def run(config: TrainerParams):
    transform_list = [
        transforms.Resize((int(config.im_size), int(config.im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    trans = transforms.Compose(transform_list)
    dataset = ImageFolder(root=config.data_root, transform=trans)
    testset, _ = random_split(
        dataset, [config.num_test_samples, len(dataset) - config.num_test_samples]
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_workers,
    )
    test_loader = DataLoader(
        testset, batch_size=config.batch_size, num_workers=config.dataloader_workers
    )

    trainer = Trainer(config)
    engine = trainer.get_engine()

    @engine.on(Events.ITERATION_COMPLETED(every=config.logging_interval))
    def log(engine: Engine):
        print(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}], Loss D: {engine.state.output['err_dr']:.5f}, Loss G: {engine.state.output['err_g']:.5f}"  # TODO: fix type error
        )

    checkpointer = trainer.get_checkpointer()
    engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.checkpoint_interval), checkpointer
    )

    @engine.on(Events.ITERATION_COMPLETED(every=config.saving_interval))
    def save(engine: Engine):
        trainer.save(test_loader)

    ProgressBar().attach(engine)

    print("Start training...")
    trainer.start(train_loader, config.num_epochs)
    trainer.end()


if __name__ == "__main__":
    args = parse(TrainerParams)
    run(args)
