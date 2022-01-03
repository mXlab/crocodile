from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from crocodile.generator import load_from_path
from crocodile.dataset.sampler import SequenceSampler
from torchvision import transforms
from torch.utils.data import DataLoader

from crocodile.dataset import LaurenceDataset, biodata
from crocodile.encoder import Encoder
from crocodile.executor import ExecutorConfig, load_executor
import torch
from torchvision import utils as vutils
import os
import subprocess
from simple_parsing import ArgumentParser


@dataclass
class Params:
    encoder_path: Path
    generator_path: Path
    seq_length: int = 1800
    batch_size: int = 32
    dataset: LaurenceDataset.Params = LaurenceDataset.Params()
    num_videos: int = 10
    tmp_dir: Optional[Path] = None
    log_dir: Path = Path("./results/videos")
    name: str = "test_1"

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name


def run(params: Params):
    device = torch.device('cuda')

    if params.tmp_dir is None:
        params.tmp_dir = Path(os.environ.get('SLURM_TMPDIR'))

    encoder = Encoder.load(params.encoder_path)
    generator = load_from_path(params.generator_path, encoder.options.epoch, device=device)
    params.dataset.resolution = generator.resolution
    encoder = encoder.to(device)
    encoder.eval()

    transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    dataset = LaurenceDataset(
            params.dataset, transform=trans, target_transform=transforms.ToTensor())

    sampler = SequenceSampler(len(dataset), params.seq_length, shuffle=True)
    dataloader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=4)

    iterator = iter(dataloader)
    for i in range(params.num_videos):
        _, biodata, _ = iterator.next()
        for j in range(biodata//params.batch_size + 1):
            _biodata = biodata[j*params.batch_size:(j+1)*params.batch_size]
            _biodata = _biodata.to(device)
            z = encoder(_biodata)
            imgs = generator(z)

            for k, img in enumerate(imgs):
                vutils.save_image(img.add(1).mul(0.5), 
                    os.path.join( params.tmp_dir / f'{j*params.batch_size+k:04d}.png'))
        
        output_file = params.save_dir / "video_%.4d.mp4"%i
        cmd = "ffmpeg -framerate %.2f -i %s/%%04d.png %s" % (dataset.config.fps, params.tmp_dir, output_file)
        subprocess.run(cmd.split())
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(run, args.params)