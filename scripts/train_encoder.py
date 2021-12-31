from crocodile.executor import load_executor, ExecutorConfig, ExecutorCallable
from crocodile.encoder import Encoder
from crocodile.generator import load_from_path
from crocodile.dataset import LaurenceDataset
from crocodile.utils.optim import load_optimizer
from crocodile.utils.loss import load_loss
from crocodile.utils.logger import Logger
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch
from simple_parsing import ArgumentParser
from tqdm import tqdm
from crocodile.params import TrainEncoderParams as Params
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


        loss_fn = load_loss(args.loss.loss, args.loss)

        encoder = Encoder(dataset.seq_dim, dataset.seq_length, generator.latent_dim, args.encoder)
        encoder.to(device)

        optimizer = load_optimizer(encoder.parameters(), args.optimizer)

        torch.manual_seed(1234)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img, biodata_ref, _ = iter(dataloader).next()
        img = img[:args.num_test_samples]
        biodata_ref = biodata_ref[:args.num_test_samples].float()
        logger.save_image("groundtruth", img)

        loss_mean = 0
        n_samples = 0
        for epoch in range(args.num_epochs):
            for img, biodata, _ in tqdm(dataloader, disable=args.debug):
                optimizer.zero_grad()

                img = img.to(device)
                biodata = biodata.float().to(device)

                z = encoder(biodata)
                img_recons = generator(z)
                loss = loss_fn(img, img_recons).mean()
                loss.backward()

                optimizer.step(loss=loss)

                loss_mean += loss.detach().item()*len(img)
                n_samples += len(img)
                if args.debug:
                    break

            loss_mean /= n_samples
            print("Epoch %i / %i, Loss: %.2f" % (epoch, args.num_epochs, loss_mean))
            if generator is not None:
                with torch.no_grad():
                    biodata = biodata_ref.to(device)
                    z = encoder(biodata)
                    img = generator(z)
                    logger.save_image("recons_%.4d" % epoch, img)
                
            logger.add({"loss": loss_mean})
            logger.save_model("%.4d" % epoch, encoder)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(TrainEncoder(), args.params)
