import os
from dataclasses import dataclass
from simple_parsing import parse, subgroups
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm

from crocodile.executor import load_executor, ExecutorConfig, executor_subgroups
from crocodile.loader import load_from_path
from crocodile.dataset import LaurenceDataset, LatentDataset
from crocodile.optimizer import load_optimizer
from crocodile.utils.loss import load_loss
from crocodile.utils.logger import Logger
from crocodile.params import ComputeLatentParams as Params


@dataclass
class Config:
    train: Params
    executor: ExecutorConfig = subgroups(executor_subgroups, default="local")


class ComputeLatent:
    def __call__(self, args: Params, resume=False):
        args.slurm_job_id = os.environ.get("SLURM_JOB_ID")
        device = torch.device("cuda")

        generator = load_from_path(args.generator_path).to(device)

        transform_list = [
            transforms.Resize((int(generator.im_size), int(generator.im_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        trans = transforms.Compose(transform_list)

        dataset = LaurenceDataset(args.dataset, transform=trans)

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        latent_dataset = LatentDataset(len(dataset), dim=generator.num_latent)

        loss_fn = load_loss(args=args.loss)

        torch.manual_seed(1234)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img_ref, _, index_ref = next(iter(dataloader))
        img_ref = img_ref[: args.num_test_samples].to(device)
        index_ref = index_ref[: args.num_test_samples]

        loss_mean = 0
        n_samples = 0
        for j, (img, _, index) in enumerate(dataloader):
            img = img.to(device)
            z = latent_dataset[index]
            z = z.to(device)
            z.requires_grad_()

            optimizer = load_optimizer([z], args.optimizer)
            for i in tqdm(range(args.num_iter), disable=args.debug):
                optimizer.zero_grad()

                img_recons = generator(z)
                loss = loss_fn(img, img_recons)
                loss_sum = loss.sum()
                loss_sum.backward()

                optimizer.step()

                if args.debug:
                    print(i, loss_sum.detach().item())
                    logger.save_image(
                        "recons_%.4d" % i, img_recons[: args.num_test_samples]
                    )

            latent_dataset[index] = z.detach().cpu()
            n_samples += len(img)
            loss = loss_sum.detach().item()
            print("Epoch %i / %i, Loss: %2f" % (j, len(dataloader), loss))
            loss_mean += loss
            logger.save_image("groundtruth_%.4d" % j, img[: args.num_test_samples])
            logger.save_image("recons_%.4d" % j, img_recons[: args.num_test_samples])
            if args.debug:
                break

        loss_mean /= n_samples
        logger.save("latent", latent_dataset)
        logger.add({"loss": loss_mean})


if __name__ == "__main__":
    config = parse(Config)
    executor = load_executor(config.executor)
    executor(ComputeLatent(), config.train)
