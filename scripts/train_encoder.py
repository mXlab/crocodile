from crocodile.executor import load_executor, ExecutorConfig
from crocodile.encoder import load_encoder, EncoderParams
from crocodile.generator import load_from_path
from dataclasses import dataclass
from crocodile.dataset import LaurenceDataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import torch.optim as optim
import torch
from simple_parsing import ArgumentParser
import torchvision
from typing import Optional


@dataclass
class Params:
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    encoder: EncoderParams = EncoderParams()
    lr: float = 1e-2
    num_epochs: int = 100
    log_dir: Path = Path("./results/encoder")
    name: str = "test_1"

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


def run(args: Params):
    device = torch.device('cuda')

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    dataset = LaurenceDataset(
        args.dataset, transform=trans, target_transform=transforms.ToTensor())

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    generator = load_from_path(args.generator_path, args.epoch, device=device)

    encoder = load_encoder(args.encoder).build(
        dataset.seq_length*dataset.dim, generator.latent_dim, device=device)

    optimizer = optim.SGD(encoder.params(), lr=args.lr)

    args.save_dir.mkdir(exist_ok=True)

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
        torchvision.utils.save_image(
            img, str(args.save_dir / "gt_%.4d.png" % epoch))
        torchvision.utils.save_image(
            img, str(args.save_dir / "recons_%.4d.png" % epoch))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(run, args.params)
