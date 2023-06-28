from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from crocodile.generator import load_from_path
from crocodile.dataset.sampler import SequenceSampler
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from crocodile.dataset import load_biodata, LaurenceDataset
from crocodile.encoder import Encoder
from crocodile.executor import ExecutorConfig, load_executor
import torch
from torchvision import utils as vutils
import os
import subprocess
from simple_parsing import ArgumentParser
from tqdm import tqdm
import shutil
import torch.nn.functional as F


@dataclass
class Params:
    encoder_path: Path
    biodata: Path
    config: Path = Path("./data/laurence/")
    seq_length: int = 1000
    batch_size: int = 32
    num_videos: int = 10
    tmp_dir: Optional[Path] = None
    log_dir: Path = Path("./results/videos")
    smoothing: float = 0.2
    name: str = "test_1"

    def __post_init__(self):
        self.save_dir = self.log_dir / self.name
        if self.save_dir.is_dir():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True)

def save_image(img, filename):
    vutils.save_image(img.add(1).mul(0.5), filename)

def run(params: Params):
    device = torch.device('cpu')

    if params.tmp_dir is None:
        params.tmp_dir = Path(os.environ.get('SLURM_TMPDIR'))

    encoder = Encoder.load(params.encoder_path)
    generator = load_from_path(encoder.options.generator_path, device=device)
    encoder = encoder.to(device)
    encoder.eval()

    config = LaurenceDataset.load_config(".data" / "laurence")
    dataset = load_biodata(params.biodata)

    sampler = SequenceSampler(dataset, params.seq_length, shuffle=True)
    iterator = iter(sampler)
    for i in range(params.num_videos):
        idx = next(iterator)
        subset = Subset(dataset, idx)
        dataloader = DataLoader(subset, batch_size=params.batch_size, shuffle=False, num_workers=4)
        image_idx = 0
        if (params.tmp_dir / 'images').is_dir():
            shutil.rmtree(params.tmp_dir / 'images')
        (params.tmp_dir / 'images').mkdir(parents=True)
        
        z_smooth: torch.Tensor = None # TODO: fix this so that z_smooth is always a tensor.
        for biodata in tqdm(dataloader):
            with torch.no_grad():
                biodata = biodata.float().to(device)
                z = encoder(biodata)            
            for _z in z:
                if z_smooth is None:
                    z_smooth = _z
                else:
                    z_smooth = params.smoothing*_z + (1 - params.smoothing)*z_smooth
                img = generator(z_smooth.unsqueeze(0)).cpu()
                save_image(img, params.tmp_dir / ('images/%.4d.png' % image_idx))
                image_idx += 1
        
        output_file = params.save_dir / ("video_%.4d.mp4"%i)
        input_file = params.tmp_dir / "images/%04d.png"
        cmd = "ffmpeg -framerate %.2f -i %s %s" % (config.fps, input_file, output_file)
        subprocess.run(cmd.split())
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Params, dest="params")
    parser.add_arguments(ExecutorConfig, dest="executor")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(run, args.params)