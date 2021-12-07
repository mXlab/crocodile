from crocodile.executor import load_executor, ExecutorConfig
from crocodile.generator import load_from_path
from dataclasses import dataclass
from pathlib import Path
import torch
from simple_parsing import ArgumentParser
from typing import Optional
from tqdm import tqdm
import os
from torchvision import utils as vutils
from crocodile.dataset import LaurenceDataset
from torchvision import transforms


@dataclass
class EvalParams:
    generator_path: Path
    batch_size: int = 64
    epoch: Optional[int] = None
    output_dir: Optional[int] = None
    output_size: int = 64
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.generator_path.parent


def run(args: EvalParams):
    device = torch.device('cuda')
    generator = load_from_path(args.generator_path, args.epoch, device=device)

    resize = transforms.Resize((args.output_size, args.output_size))

    dataset = LaurenceDataset(args.dataset)

    noise = generator.sample_z(len(dataset))
    images = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)//args.batch_size + 1)):
            _noise = noise[i*args.batch_size: (i+1)*args.batch_size]
            images.append(resize(generator(_noise)).cpu())
        images = torch.cat(images)
        torch.save(images, args.output_dir / "images.pt")
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExecutorConfig, dest="executor")
    parser.add_arguments(EvalParams, dest="eval")
    args = parser.parse_args()

    executor = load_executor(args.executor)

    executor(run, args.eval)