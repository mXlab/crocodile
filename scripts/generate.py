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


@dataclass
class EvalParams:
    generator_path: Path
    num_frames: int = 100
    batch_size: int = 16
    epoch: Optional[int] = None
    output_dir: Optional[int] = None

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.generator_path.parent / "eval"


def run(args: EvalParams):
    device = torch.device('cuda')
    generator = load_from_path(args.generator_path, args.epoch, device=device)

    start_noise = generator.sample_z()
    end_noise = generator.sample_z()
    alpha = torch.linspace(
        0, 1, args.num_frames).to(device).view(-1, 1)
    noise = alpha*start_noise + (1-alpha)*end_noise

    args.output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.num_frames//args.batch_size + 1)):
            _noise = noise[i*args.batch_size: (i+1)*args.batch_size]
            img = generator(_noise)[0]
            for j, g_img in enumerate(img):
                vutils.save_image(g_img.add(1).mul(0.5),
                                  os.path.join(args.output_dir, f'{i*args.batch_size+j:04d}.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExecutorConfig, dest="executor")
    parser.add_arguments(EvalParams, dest="eval")
    args = parser.parse_args()

    executor = load_executor(args.executor)

    executor(run, args.eval)
