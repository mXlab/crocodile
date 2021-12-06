from pathlib import Path
import shutil
import torchvision
import torch
from typing import Any
import json
from collections import defaultdict


class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        if self.log_dir.is_dir():
            shutil.rmtree(self.log_dir)
        (self.log_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "models").mkdir(parents=True, exist_ok=True)

        self.results = defaultdict(list)

    def save_args(self, args: Any):
        args.save(self.log_dir / "args.yaml")

    def save_image(self, name: str, image: torch.Tensor):
        torchvision.utils.save_image(
            image.add(1).mul(0.5), str(self.log_dir / f"images/{name}.png"))

    def add(self, obj):
        for key, value in obj.items():
            self.results[key].append(value)
        with open(str(self.log_dir / 'results.json'), 'w') as f:
            json.dump(self.results, f)
        print(obj)

    def save_model(self, name: str, model: Any):
        self.save("models/%s" % name, model)

    def save(self, name: str, obj: Any):
        obj.save(self.log_dir / f"{name}.pth")
