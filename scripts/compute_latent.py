from crocodile.executor import load_executor, ExecutorConfig, ExecutorCallable
from crocodile.generator import load_from_path
from crocodile.dataset import LaurenceDataset, LatentDataset
from crocodile.utils.optim import load_optimizer
from crocodile.utils.loss import load_loss
from crocodile.utils.logger import Logger
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from simple_parsing import ArgumentParser
from tqdm import tqdm
from crocodile.params import ComputeLatentParams as Params
import os


class ComputeLatent(ExecutorCallable):
    def __call__(self, args: Params, resume=False):
        args.slurm_job_id = os.environ.get('SLURM_JOB_ID')
        device = torch.device('cuda')

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        trans = transforms.Compose(transform_list)

        generator = load_from_path(
            args.generator_path, args.epoch, device=device)
        args.dataset.resolution = generator.resolution

        dataset = LaurenceDataset(
            args.dataset, transform=trans, target_transform=transforms.ToTensor())

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        latent_dataset = LatentDataset(len(dataset), dim=generator.latent_dim)

        loss_fn = load_loss(args.loss.loss, args.loss)

        torch.manual_seed(1234)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img_ref, _, index_ref = iter(dataloader).next()
        img_ref = img_ref[:args.num_test_samples].to(device)
        index_ref = index_ref[:args.num_test_samples]

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

                optimizer.step(loss=loss)

                if args.debug:
                    print(i, loss_sum.detach().item())
                    logger.save_image("recons_%.4d" %
                                      i, img_recons[:args.num_test_samples])

            latent_dataset[index] = z.detach().cpu()
            n_samples += len(img)
            loss = loss_sum.detach().item()
            print("Epoch %i / %i, Loss: %2f" % (j, len(dataloader), loss))
            loss_mean += loss
            logger.save_image("groundtruth_%.4d" %
                              j, img[:args.num_test_samples])
            logger.save_image("recons_%.4d" %
                              j, img_recons[:args.num_test_samples])
            if args.debug:
                break

        loss_mean /= n_samples
        logger.save("latent", latent_dataset)
        logger.add({"loss": loss_mean})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(ComputeLatent(), args.params)
