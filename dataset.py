from torch.utils.data import Dataset
import glob
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm
import subprocess
import argparse

ROOT = "/network/tmp1/berardhu/crocodile/data"

def extract_video(path_to_video, path_to_dataset):
    if not os.path.exists(os.path.join(path_to_dataset, "raw")):
        os.makedirs(os.path.join(path_to_dataset, "raw"))
    command = "ffmpeg -i {} -f image2 {}".format(path_to_video, os.path.join(path_to_dataset, "raw/frame_%07d.png"))
    subprocess.run(command.split())

def resize_images(path_to_dataset, resolution):
    if not os.path.exists(os.path.join(path_to_dataset, str(resolution))):
        os.makedirs(os.path.join(path_to_dataset, str(resolution)))
    list_images = glob.glob(os.path.join(path_to_dataset, "raw/frame_*.png"))
    for i, file in tqdm(enumerate(list_images)):
        img = Image.open(file)
        img = img.resize((resolution, resolution), 3, box=(404, 92, 404+472, 92+472))
        img.save(os.path.join(path_to_dataset, str(resolution), "%.7i.png"%i))



class CrocodileDataset(Dataset):
    def __init__(self, root=ROOT, transform=None, resolution=64):
        super(CrocodileDataset, self).__init__()

        self.transform = transform
        self.resolution = resolution
        self.root = root

        if not os.path.exists(os.path.join(self.root, "raw")):
            raise OSError("Couldn't find raw images. Run dataset.py before running train.py.")

        if not os.path.exists(os.path.join(self.root, str(resolution))):
            os.makedirs(os.path.join(self.root, str(resolution)))

        self.num_frames = len(glob.glob(os.path.join(self.root, "raw/frame_*.png")))
        num_files = len(glob.glob(os.path.join(self.root, str(resolution), "*.png")))
        if num_files != self.num_frames:
            print("Processing image files ...")
            resize_images(self.root, resolution)


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, str(self.resolution), "%.7i.png"%index))

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.num_frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_video')
    parser.add_argument('path_to_dataset')
    parser.add_argument('-r', '--resolution', default=None, type=int)
    args = parser.parse_args()
    extract_video(args.path_to_video, args.path_to_dataset)
    if not args.resolution is None:
        resize_images(args.path_to_dataset, args.resolution)
