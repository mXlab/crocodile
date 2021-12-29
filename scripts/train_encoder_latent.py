from crocodile import generator
from crocodile.executor import load_executor, ExecutorConfig, ExecutorCallable
from crocodile.encoder import load_encoder
from crocodile.generator import load_from_path
from crocodile.dataset import LaurenceDataset, LatentDataset, latent
from crocodile.utils.optim import load_optimizer
from crocodile.utils.loss import load_loss
from crocodile.utils.logger import Logger
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from simple_parsing import ArgumentParser
from tqdm import tqdm
from crocodile.params import TrainEncoderLatentParams as Params
import os


class TrainEncoder(ExecutorCallable):
    def __call__(self, args: Params, resume=False):
        args.slurm_job_id = os.environ.get('SLURM_JOB_ID')
        device = torch.device('cuda')

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        trans = transforms.Compose(transform_list)

        generator = None
        if args.generator_path is not None:
            generator = load_from_path(
                args.generator_path, args.epoch, device=device)
            args.dataset.resolution = generator.resolution

        dataset = LaurenceDataset(
            args.dataset, transform=trans)
        
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        latent_dataset = LatentDataset.load(args.latent_path)

        loss_fn = load_loss(args.loss.loss, args.loss)

        encoder = load_encoder(args.encoder).build(
            dataset.seq_length*dataset.seq_dim, latent_dataset.dim)
        encoder.to(device)

        optimizer = load_optimizer(encoder.parameters(), args.optimizer)

        torch.manual_seed(1234)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img, biodata_ref, index = iter(dataloader).next()
        print(dataset[0][0].size(), dataset[0][1].size())
        print(img.size(), biodata_ref.size(), index.size())
        img = img[:args.num_test_samples]
        biodata_ref = biodata_ref[:args.num_test_samples].float()
        index = index[:args.num_test_samples]
        logger.save_image("groundtruth", img)
        if generator is not  None:
            z = latent_dataset[index]
            z = z.to(device)
            img = generator(z)
            logger.save_image("groundtruth_latent", img)
            
        print("Biodata size: ", biodata_ref.size())


        loss_mean = 0
        n_samples = 0
        for epoch in range(args.num_epochs):
            for _, biodata, index in tqdm(dataloader, disable=args.debug):
                optimizer.zero_grad()

                z = latent_dataset[index]
                z = z.to(device)
                biodata = biodata.float().to(device)

                z_recons = encoder(biodata)
                loss = loss_fn(z, z_recons).mean()
                loss.backward()

                optimizer.step(loss=loss)

                loss_mean += loss.detach().item()*len(index)
                n_samples += len(index)
                if args.debug:
                    print(loss_mean/n_samples)

            loss_mean /= n_samples
            print("Epoch %i / %i, Loss: %2f" % (epoch, args.num_epochs, loss_mean))
            if args.debug:
                break

            logger.add({"loss": loss_mean})
            logger.save_model("{epoch:04d}", encoder)

            if generator is not None:
                biodata = biodata_ref.to(device)
                z = encoder(biodata)
                img = generator(z)
                logger.save_image("recons_%.4d" % epoch, img)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(TrainEncoder(), args.params)
