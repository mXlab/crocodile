from crocodile.executor import load_executor, ExecutorConfig, ExecutorCallable
from crocodile.encoder import Encoder
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
from crocodile.params import TrainEncoderParams as Params
import os


class TrainEncoder(ExecutorCallable):
    def __call__(self, args: Params, resume=False):
        args.slurm_job_id = os.environ.get('SLURM_JOB_ID')
        device = torch.device('cuda')
        torch.manual_seed(1234)

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        trans = transforms.Compose(transform_list)

        generator = load_from_path(args.encoder.generator_path, args.encoder.epoch, device=device)
        args.dataset.resolution = generator.resolution

        dataset = LaurenceDataset(
            args.dataset, transform=trans)
        
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        latent_dataset = None
        if args.latent_path is not None:
            latent_dataset = LatentDataset.load(args.latent_path)

        loss_fn = load_loss(args.loss.loss, args.loss)

        encoder = Encoder(dataset.seq_dim, dataset.seq_length, generator.latent_dim, args.encoder)
        encoder.to(device)

        optimizer = load_optimizer(encoder.parameters(), args.optimizer)

        logger = Logger(args.save_dir)
        logger.save_args(args)

        img, biodata_ref, index = iter(dataloader).next()
        img = img[:args.num_test_samples]
        biodata_ref = biodata_ref[:args.num_test_samples].float()
        logger.save_image("groundtruth", img)
        
        if latent_dataset is not None:
            index = index[:args.num_test_samples]
            z = latent_dataset[index]
            z = z.to(device)
            img = generator(z)
            logger.save_image("groundtruth_latent", img)

        regularization_coeff = args.latent_regularization
        best_loss = None
        for epoch in range(args.num_epochs):
            if args.decreasing_regularization:
                regularization_coeff = args.latent_regularization*(1 - epoch/(args.num_epochs -1))
            
            loss_mean = 0
            loss_latent_mean = 0
            n_samples = 0
            for img, biodata, idx in tqdm(dataloader, disable=args.debug):
                optimizer.zero_grad()

                img = img.to(device)
                biodata = biodata.float().to(device)

                z = encoder(biodata)
                img_recons = generator(z)
                loss = loss_fn(img, img_recons, reduce="sum").mean()

                loss_mean += loss.detach().item()*len(img)

                if latent_dataset is not None:
                    z_true = latent_dataset[idx]
                    z_true = z_true.to(device)
                    loss_latent = loss_fn(z, z_true, reduce="sum").mean()
                    loss = (1-args.latent_regularization)*loss + regularization_coeff * loss_latent * 1000
                    loss_latent_mean += loss_latent.detach().item()*len(img)

                loss.backward()

                optimizer.step(loss=loss)
          
                n_samples += len(img)
                if args.debug:
                    break

            loss_mean /= n_samples
            loss_latent_mean /= n_samples
            print("Epoch %i / %i, Loss: %.2f, Loss latent: %.2f" % (epoch, args.num_epochs, loss_mean, loss_latent_mean))
            if generator is not None:
                with torch.no_grad():
                    biodata = biodata_ref.to(device)
                    z = encoder(biodata)
                    img = generator(z)
                    logger.save_image("recons_%.4d" % epoch, img)
                
            logger.add({"loss": loss_mean, "regularization": loss_latent_mean})
            
            if best_loss is None or loss_mean < best_loss:
                best_loss = loss_mean
                logger.save_model("model", encoder)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(TrainEncoder(), args.params)
