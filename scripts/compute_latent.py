from crocodile.executor import load_executor, ExecutorConfig
from crocodile.generator import load_from_path
from crocodile.dataset import LaurenceDataset, LatentDataset
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import torch
from simple_parsing import ArgumentParser
import torchvision
from typing import Optional
from tqdm import tqdm
import torch.autograd as autograd


@dataclass
class Params:
    generator_path: Path
    epoch: Optional[int] = None
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    batch_size: int = 64
    lr: float = 5e-5
    num_epochs: int = 100
    log_dir: Path = Path("./results/latent")
    name: str = "test_1"

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


def run(args: Params):
    torch.manual_seed(1234)
    device = torch.device('cuda')

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    generator = load_from_path(args.generator_path, args.epoch, device=device)
    args.dataset.resolution = generator.resolution

    dataset = LaurenceDataset(
        args.dataset, transform=trans, target_transform=transforms.ToTensor())

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    latent_dataset = LatentDataset(len(dataset), dim=generator.latent_dim)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    img_ref, _, index_ref = iter(dataloader).next()
    img_ref = img_ref[:10]
    index_ref = index_ref[:10]
    torchvision.utils.save_image(
        img_ref.add(1).mul(0.5), str(args.save_dir / "groundtruth.png"))

    for epoch in range(args.num_epochs):
        for img, label, index in tqdm(dataloader):
            img = img.to(device)
            z = latent_dataset[index]
            z = z.to(device)
            z.requires_grad_()

            img_recons = generator(z)

            loss = ((img - img_recons)**2).view(len(img), -1).sum(-1).mean()
            grad = autograd.grad(loss, z)[0]

            z = z - args.lr*grad
            latent_dataset[index] = z.detach().cpu()

        print(epoch, loss.item())
        with torch.no_grad():
            z = latent_dataset[index_ref].to(device)
            img_recons = generator(z)

        torchvision.utils.save_image(
            img_recons.add(1).mul(0.5), str(args.save_dir / f"recons_{epoch:04d}.png"))
        latent_dataset.save(str(args.save_dir / "latent.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(run, args.params)
