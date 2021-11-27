from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import subprocess
import argparse
from dataclasses import dataclass
from pathlib import Path
from pygan.utils.drive import GoogleDrive, check_integrity
from omegaconf import OmegaConf


class LaurenceDataset(Dataset):
    FILES = [
        ('1bP5iSAIM2SRJCvi9Ul813RrZ6kV9og3D',
         'videos/001.mov', '6df26e9bfb7825059b390b7bbf66a370'),
        ('16XBEGwn6cTgX7S4WSIFyQVuiX9t7cCSJ', 'config.yaml', None)
    ]

    @dataclass
    class Params:
        root: Path = Path("./data")
        resolution: int = 64

    def __init__(self, args: Params = Params(), transform=None):
        super().__init__()

        self.path = args.root / "laurence"
        self.transform = transform
        self.resolution = args.resolution

        self.download(self.path)
        self.extract_video(self.path)
        self.process_images(self.path, args.resolution)

        self.images = self.load_images(self.path / str(args.resolution))

    def get_path(self):
        return (self.path / str(self.resolution)).resolve()

    @classmethod
    def download(cls, path: Path):
        if cls.check_all_integrity(path):
            return

        path.mkdir(exist_ok=True)
        token = path / "token.json"

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

        print("Processing frames at resolution %i ..."% resolution)

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

    @staticmethod
    def load_labels():
        labels_path = args.root / args.labels_filename

        raw_labels = np.loadtxt(os.path.join(labels_path, delimiter=",",
                                             dtype=config.types))

        labels_to_index = {}
        index_to_labels = []
        i = 0
        for row in raw_labels:
            if row[2] not in labels_to_index:
                labels_to_index[row[2]] = i
                index_to_labels.append(row[2])
                i += 1
        num_cat = len(index_to_labels)

    @staticmethod
    def load_biodata():
        print("Loading biodata...")
        self.biodata = True
        signal = np.loadtxt(biodata, delimiter=',')

        if preprocessing is not None:
            print("Preprocessing biodata...")
            self.features = preprocessing(signal)
        else:
            self.features = signal[:, 0:]

        index = (start_data + (np.arange(self.num_frames) - start_img)
                 * sampling_rate/fps).astype(int)
        self.features = self.features[index[index < len(self.features)]]
        self.num_features = self.features.shape[1]

        self.num_samples = len(self.features)
        self.length = min(self.num_samples, self.num_frames)

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_label(self, index):
        label = None
        for row in self.raw_labels:
            if index > row[1]:
                continue
            label = self.labels_to_index[row[2]]
            break
        if label is None:
            raise IndexError("index out of range")

        if self.one_hot:
            target = torch.zeros(len(self.index_to_labels))
            target[label] = 1
        else:
            target = torch.tensor([label]).float()

        if self.biodata:
            feature = self.features[index]
            if self.feature_transform is not None:
                feature = self.feature_transform(feature)
            return img, target, feature
        else:
            return img, target

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_video')
    parser.add_argument('path_to_dataset')
    parser.add_argument('-r', '--resolution', default=None, type=int)
    args = parser.parse_args()
    extract_video(args.path_to_video, args.path_to_dataset)
    if args.resolution is not None:
        resize_images(args.path_to_dataset, args.resolution)
