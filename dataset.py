from torch.utils.data import Dataset
import glob
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm

class CrocodileDataset(Dataset):
    path = "/network/tmp1/berardhu/erin_project/data_%i.pt"
    raw_folder = "/network/tmp1/berardhu/erin_project/img1"
    filename = "frame*.png"
    def __init__(self, transform, resolution=64):
        super(CrocodileDataset, self).__init__()

        self.transform = transform

        if not os.path.exists(self.path%resolution):
            print("Processing image files ...")
            image_set = []
            list_images = glob.glob(os.path.join(self.raw_folder, self.filename))
            for file in tqdm(list_images):
                img = Image.open(file)
                img = img.resize((resolution, resolution), 3, box=(404, 92, 404+472, 92+472))
                img = np.asarray(img)
                image_set.append(img)
            image_set = np.stack(image_set)
            image_set = torch.from_numpy(image_set)

            with open(os.path.join(self.path%resolution), 'wb') as f:
                torch.save(image_set, f)

        self.data = torch.load(self.path%resolution)

    def __getitem__(self, index):
        img = self.data[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)
