from crocodile.executor import load_executor, ExecutorConfig
from dataclasses import dataclass
from crocodile.mask3d import Face3dModel
from pathlib import Path
import os
import subprocess
from simple_parsing import ArgumentParser


@dataclass
class Params:
    images_path: Path = Path("./results/fastgan/test_1/eval")
    model_type: Face3dModel = Face3dModel.DDFA_V2


def run(args):
    path = args.images_path.resolve()
    if path is None or not path.is_dir():
        raise("Please specify a valid path for the directory containing the images.")

    if args.model_type == Face3dModel.DDFA_V2:
        os.chdir('pygan/face3d/3DDFA_V2')
        command = "python demo.py -f %s -o obj" % (
            path)
        print("Running: %s" % command)
        subprocess.run(command.split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExecutorConfig, dest="executor")
    parser.add_arguments(Params, dest="params")
    args = parser.parse_args()

    executor = load_executor(args.executor)
    executor(run, args.params)
