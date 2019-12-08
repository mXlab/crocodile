from torch.utils.data import Dataset
import glob
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm
import subprocess
import argparse
import csv

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
        img = img.crop(box=(720, 218, 720+472, 218+472))
        img = img.resize((resolution, resolution), 3)
        img.save(os.path.join(path_to_dataset, str(resolution), "%.7i.png"%i))



class CrocodileDataset(Dataset):
    def __init__(self, root=ROOT, transform=None, resolution=64, one_hot=True):
        super(CrocodileDataset, self).__init__()

        self.transform = transform
        self.one_hot = one_hot
        self.resolution = resolution
        self.root = root

        self.raw_labels = np.loadtxt(os.path.join(self.root, "timestamps.csv"), delimiter=",",
                                 dtype={'names': ('start', 'end', 'emotion'), 'formats': (int, int, "<S8")})
        self.labels_to_index = {}
        self.index_to_labels = []
        i = 0
        for row in self.raw_labels:
            if row[2] not in self.labels_to_index:
                self.labels_to_index[row[2]] = i
                self.index_to_labels.append(row[2])
                i += 1
        self.num_cat = len(self.index_to_labels)

        if not os.path.exists(os.path.join(self.root, "raw")):
            raise OSError("Couldn't find raw images. Run dataset.py before running train.py.")

        if not os.path.exists(os.path.join(self.root, str(resolution))):
            os.makedirs(os.path.join(self.root, str(resolution)))

        list_frames = glob.glob(os.path.join(self.root, "raw/frame_*.png"))
        self.num_frames = len(list_frames)
        list_frames = glob.glob(os.path.join(self.root, str(resolution), "*.png"))
        num_files = len(list_frames)
        if num_files != self.num_frames:
            print("Processing image files ...")
            resize_images(self.root, resolution)



    def __getitem__(self, index):   
        label = None
        for row in self.raw_labels:
            if index > row[1]:
                continue
            label = self.labels_to_index[row[2]]
            break
        if label is None:
            raise IndexError("index out of range")

        img = Image.open(os.path.join(self.root, str(self.resolution), "%.7i.png"%index))

        if self.transform is not None:
            img = self.transform(img)       
        if self.one_hot:
            target = torch.zeros(len(self.index_to_labels))
            target[label] = 1
        else:
            target = torch.tensor([label]).float()

        return img, target

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
