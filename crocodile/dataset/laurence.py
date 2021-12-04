from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import subprocess
from dataclasses import dataclass
from pathlib import Path
from crocodile.utils.drive import GoogleDrive, check_integrity
from omegaconf import OmegaConf
from .biodata import Biodata


class LaurenceDataset(Dataset):
    FILES = [
        ('1bP5iSAIM2SRJCvi9Ul813RrZ6kV9og3D',
         'videos/001.mov', '6df26e9bfb7825059b390b7bbf66a370'),
        ('16XBEGwn6cTgX7S4WSIFyQVuiX9t7cCSJ', 'config.yaml', None),
        ('1Fvc4gL-ccpsdBpNZUmrqvvswnyfVEumG', 'videos/001.csv', None),
    ]

    @dataclass
    class Params:
        dataset_path: Path = Path("./data")
        resolution: int = 64
        biodata: Biodata.Params = Biodata.Params()
        token: Path = Path("./token.json")

    def __init__(self, args: Params = Params(), transform=None, target_transform=None):
        super().__init__()

        self.path = args.dataset_path / "laurence"
        self.transform = transform
        self.target_transform = target_transform
        self.resolution = args.resolution
        self.token = args.token

        self.download(self.path, self.token)
        self.extract_video(self.path)
        self.process_images(self.path, args.resolution)

        self.config = self.load_config(self.path)

        self.images = self.load_images(self.path / str(args.resolution))
        self.biodata = Biodata(self.path, self.config, args.biodata)
        self.seq_length = self.biodata.seq_length
        self.seq_dim = self.biodata.dim

    def get_path(self):
        return (self.path / str(self.resolution)).resolve()

    @classmethod
    def download(cls, path: Path, token: Path):
        if cls.check_all_integrity(path):
            return

        path.mkdir(exist_ok=True)

        drive = GoogleDrive.connect_to_drive(token)

        for file_id, filename, md5 in cls.FILES:
            drive.download_file(file_id, path / filename, md5)

    @classmethod
    def check_all_integrity(cls, path):
        return all(check_integrity(path / filemame, md5) for _, filemame, md5 in cls.FILES)

    @classmethod
    def check_video_integrity(cls, path, num_frames=None):
        if not path.exists():
            return False
        if num_frames is None:
            return True
        return num_frames == cls.get_num_frames(path)

    @classmethod
    def get_num_frames(cls, path: Path):
        return len(cls.load_images(path))

    @classmethod
    def extract_video(cls, path: Path):
        config = cls.load_config(path)

        path_to_raw = path / "raw"
        if cls.check_video_integrity(path_to_raw, config.num_frames):
            return

        print("Extracting video frames...")

        path_to_raw.mkdir(exist_ok=True)
        command = "ffmpeg -i {} -f image2 {}".format(
            path / config.video_file, path_to_raw / "frame_%07d.png")
        subprocess.run(command.split())

    @classmethod
    def process_images(cls, path: Path, resolution: int):
        config = cls.load_config(path)

        path_img = path / str(resolution)
        if cls.check_video_integrity(path_img, config.num_frames):
            return

        print("Processing frames at resolution %i ..." % resolution)

        path_img.mkdir(exist_ok=True)
        list_images = cls.load_images(path / "raw")
        for i, file in enumerate(tqdm(list_images)):
            img = Image.open(file)
            img = img.crop(box=(config.crop_x, config.crop_y, config.crop_x +
                           config.crop_width, config.crop_y + config.crop_height))
            img = img.resize((resolution, resolution), 3)
            img.save(path_img / ("%.7i.png" % i))

    @staticmethod
    def load_config(path: Path):
        path = path / "config.yaml"
        return OmegaConf.load(path)

    @staticmethod
    def load_images(path: Path):
        return sorted(path.glob("*.png"))

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.transform is not None:
            img = self.transform(img)

        biodata = self.biodata[index]

        if self.target_transform is not None:
            biodata = self.target_transform(biodata)

        return img, biodata

    def convert_index(self, index):
        return int((index - self.config.start_sensor)/self.config.sampling_rate * self.config.fps + self.config.start_video)

    def __len__(self):
        return min(len(self.images), self.convert_index(len(self.biodata)))
