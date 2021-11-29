from pygan.launcher import Launcher
from dataclasses import dataclass
from pygan.face3d import Face3dModel
from pathlib import Path
import os
import subprocess
from simple_parsing import ArgumentParser


class Face3D(Launcher):
    @dataclass
    class Params(Launcher.Params):
        path: Path = Path("./results/styleformer/test1")
        model_type: Face3dModel = Face3dModel.DDFA_V2

    def __init__(self, args):
        super().__init__(args)

    def run(self, args):
        path = args.path.resolve()
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
    parser.add_arguments(Face3D.Params, dest="train")
    args = parser.parse_args()

    launcher = Face3D(args.train)
    launcher.launch(args.train)
