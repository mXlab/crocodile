from crocodile.launcher import Launcher
from crocodile.generator import ModelType
from crocodile.dataset import LaurenceDataset
from dataclasses import dataclass
from pathlib import Path
import torch
from simple_parsing import ArgumentParser
from typing import Optional
from tqdm import tqdm
import os
from torchvision import utils as vutils


class Generate(Launcher):
    @dataclass
    class Params(Launcher.Params):
        model_path: Path = None
        num_frames: int = 100
        batch_size: int = 16
        model_type: ModelType = ModelType.STYLEFORMER
        output_dir: Optional[Path] = None
        name: str = "test1"

        def __post_init__(self):
            if self.output_dir is None:
                self.output_dir = Path("./results/%s/%s" % (
                                       self.model_type.value, self.name))

    def run(self, args):
        model_path = args.eval.model_path.resolve()
        output_dir = args.eval.output_dir.resolve()

        device = torch.device('cuda')

        model = ModelType.load(args.eval.model_type)
        model = model.load_model(model_path, device=device)

        start_noise = model.sample_z()
        end_noise = model.sample_z()
        alpha = torch.linspace(
            0, 1, args.eval.num_frames).to(device).view(-1, 1)
        noise = alpha*start_noise + (1-alpha)*end_noise

        with torch.no_grad():
            for i in tqdm(range(args.eval.num_frames//args.eval.batch_size + 1)):
                _noise = noise[i*args.batch: (i+1)*args.eval.batch_size]
                img = model(_noise)[0]
                for j, g_img in enumerate(img):
                    vutils.save_image(g_img.add(1).mul(0.5),
                                      os.path.join(output_dir, f'{i*args.eval.batch_size+j:04d}.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(LaurenceDataset.Params, dest="dataset")
    parser.add_arguments(Generate.Params, dest="eval")
    args = parser.parse_args()

    launcher = Generate(args.eval)
    launcher.launch(args)
