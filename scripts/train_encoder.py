import os
from collections import defaultdict
from dataclasses import dataclass


from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from simple_parsing import parse, subgroups
from tqdm import tqdm
from crocodile.params import TrainEncoderParams as Params
from crocodile.executor import load_executor, ExecutorConfig, executor_subgroups
from crocodile.encoder import Encoder
from crocodile.loader import load_from_path
from crocodile.dataset import LaurenceDataset, LatentDataset
from crocodile.optimizer import load_optimizer
from crocodile.utils.loss import EuclideanLoss, PerceptualLoss
from crocodile.utils.logger import Logger


@dataclass
class Config:
    train: Params
    executor: ExecutorConfig = subgroups(executor_subgroups, default="local")


class TrainEncoder:
    def __call__(self, args: Params, resume=False):
        args.slurm_job_id = os.environ.get("SLURM_JOB_ID")
        device = torch.device("cuda")
        torch.manual_seed(1234)

        generator = load_from_path(args.generator_path).to(device)

        transform_list = [
            transforms.Resize((int(generator.im_size), int(generator.im_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]

        trans = transforms.Compose(transform_list)
        dataset = LaurenceDataset(args.dataset, transform=trans)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        latent_dataset = None
        if args.latent_path is not None:
            latent_dataset = LatentDataset.load(args.latent_path)

        encoder = Encoder(
            dataset.seq_dim, dataset.seq_length, generator.num_latent, args.encoder
        )
        encoder.to(device)

        optimizer = load_optimizer(encoder.parameters(), args.optimizer)

        percep_loss = None
        if args.loss.percep_coeff is not None:
            percep_loss = PerceptualLoss(args.loss.perceptual_options)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img, biodata_ref, index = next(iter(dataloader))
        img = img[: args.num_test_samples]
        biodata_ref = biodata_ref[: args.num_test_samples].float()
        logger.save_image("groundtruth", img)

        if latent_dataset is not None:
            index = index[: args.num_test_samples]
            z = latent_dataset[index]
            z = z.to(device)
            img = generator(z)
            logger.save_image("groundtruth_latent", img)

        regularization_coeff = args.latent_regularization
        best_loss = None
        for epoch in range(args.num_epochs):
            if args.decreasing_regularization:
                regularization_coeff = args.latent_regularization * (
                    1 - epoch / (args.num_epochs - 1)
                )

            metrics = defaultdict(float)
            for img, biodata, idx in tqdm(dataloader, disable=args.debug):
                optimizer.zero_grad()

                img = img.to(device)
                biodata = biodata.float().to(device)

                z = encoder(biodata)
                img_recons = generator(z)

                loss = EuclideanLoss()(img, img_recons, reduce="sum").mean()
                metrics["loss_recons"] += loss.detach().item() * len(img)

                if percep_loss is not None:
                    loss_percep = percep_loss(img, img_recons, reduce="sum").mean()
                    loss = (
                        1 - args.loss.percep_coeff
                    ) * loss + args.loss.percep_coeff * loss_percep * 100000
                    metrics["loss_percep"] += loss_percep.detach().item() * len(img)

                if latent_dataset is not None:
                    z_true = latent_dataset[idx]
                    z_true = z_true.to(device)
                    loss_latent = EuclideanLoss()(z, z_true, reduce="sum").mean()
                    loss = (
                        1 - args.latent_regularization
                    ) * loss + regularization_coeff * loss_latent * 10000
                    metrics["loss_latent"] += loss_latent.detach().item() * len(img)

                loss.backward()

                optimizer.step()

                metrics["n_samples"] += len(img)
                if args.debug:
                    break

            metrics["loss_recons"] /= metrics["n_samples"]
            metrics["loss_latent"] /= metrics["n_samples"]
            metrics["loss_percep"] /= metrics["n_samples"]
            if generator is not None:
                with torch.no_grad():
                    biodata = biodata_ref.to(device)
                    z = encoder(biodata)
                    img = generator(z)
                    logger.save_image("recons_%.4d" % epoch, img)

            logger.add(metrics)

            if best_loss is None or metrics["loss_recons"] < best_loss:
                best_loss = metrics["loss_recons"]
                logger.save_model("model", encoder)


if __name__ == "__main__":
    config = parse(Config)
    executor = load_executor(config.executor)
    executor(TrainEncoder(), config.train)
