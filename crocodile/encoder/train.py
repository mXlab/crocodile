from crocodile.launcher import Launcher
from dataclasses import dataclass
from crocodile.dataset import LaurenceDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from crocodile.generator import ModelType
from .models import EncoderType
import torch.optim as optim
import torch
from simple_parsing import ArgumentParser
import torchvision


class Train(Launcher):
    @dataclass
    class Params(Launcher.Params):
        dataset: LaurenceDataset.Params = LaurenceDataset.Params()
        batch_size: int = 64
        model_path: Path = None
        model_type: ModelType = ModelType.FASTGAN
        encoder_type: EncoderType = EncoderType.MLP
        lr: float = 1e-2
        num_epochs: 100
        log_dir: Path = Path("./results/encoder")
        name: str = "test1"

    @classmethod
    def parse_args(cls):
        parser = ArgumentParser()
        parser.add_argument(cls.Params, dest="train")

        args, _ = parser.parse_known_args()

        encoder_params = EncoderType.load_params(args.encoder_type)
        parser.add_argument(encoder_params, dest="train.encoder")

        args = parser.parse_args()

        return args.train

    def run(self, args):
        device = torch.device('cuda')

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        trans = transforms.Compose(transform_list)

        dataset = LaurenceDataset(
            args.dataset, transform=trans, target_transform=transforms.ToTensor())

        dataloader = DataLoader(dataset, batch_size=args.batch_size)

        generator = ModelType.load(args.model_type)
        generator = generator.load_model(args.model_path, device=device)

        encoder = EncoderType.load(args.encoder_type).init(
            dataset.seq_length*dataset.dim, generator.latent_dim, args.encoder, device=device)

        optimizer = optim.SGD(encoder.params(), lr=args.lr)

        for epoch in range(args.num_epochs):
            for img, label in dataloader:
                optimizer.zero_grad()

                img = img.to(device)
                label = label.to(device)

                z = encoder(img)
                img_recons = generator(z)

                loss = ((img - img_recons)**2).view(len(img), -1).sum(-1).mean()
                loss.backward()

                optimizer.step()

            print(loss)
            torchvision.utils.save_image(img, str(args.log_dir / args.name / "gt_%.4d.png"%epoch))
            torchvision.utils.save_image(img, str(args.log_dir / args.name / "recons_%.4d.png"%epoch))


if __name__ == "__main__":
    args = Train.parse_args()
    trainer = Train(args)
    trainer.launch(args)

