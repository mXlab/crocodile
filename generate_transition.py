import argparse
import json
import os
import torch
import models
from dataset import CrocodileDataset
import torchvision
import subprocess
import shutil
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument("--epoch", default=None, type=int)
parser.add_argument("--num-samples", default=100, type=int)
config = parser.parse_args()

with open(os.path.join(config.input, "config.json"), 'r') as f:
    args = argparse.Namespace(**json.load(f))

EMA = args.ema
BATCH_SIZE = args.batch_size
NUM_Z = args.num_latent
NUM_FILTERS = args.num_filters
LR_GEN = args.learning_rate_gen
LR_DIS = args.learning_rate_dis
NUM_EPOCHS = args.num_epochs
SEED = args.seed
RESOLUTION = args.resolution
GRADIENT_PENALTY = args.gradient_penalty
torch.manual_seed(SEED)
ROOT = args.path_to_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Init...")

dataset = CrocodileDataset(root=ROOT, resolution=RESOLUTION, one_hot=True)

gen = models.SmallGenerator(NUM_Z+dataset.num_cat, RESOLUTION, NUM_FILTERS, args.num_layers, spectral_norm=args.spectral_norm_gen).to(device)

if config.epoch is None:
    checkpoint = torch.load(os.path.join(config.input, "last_model.chk"))
else:
    checkpoint = torch.load(os.path.join(config.input, "gen/gen_%i.chk"%config.epoch))

gen.load_state_dict(checkpoint["gen_state_dict"])
gen = gen.to(device)

if os.path.exists(os.path.join(config.output, "img")):
    shutil.rmtree(os.path.join(config.output, "img"))
os.makedirs(os.path.join(config.output, "img"))
    

y = torch.eye(dataset.num_cat)
y1 = y[torch.randint(dataset.num_cat, size=(10*10,))]
z1 = torch.zeros(10*10, NUM_Z).normal_()
y1 = torch.cat([z1,y1], -1)

y2 = y[torch.randint(dataset.num_cat, size=(10*10,))]
z2 = torch.zeros(10*10, NUM_Z).normal_()
y2 = torch.cat([z1,y2], -1)

alpha = torch.linspace(0, 1, config.num_samples)
print("Generating...")
for i, alpha in tqdm.tqdm(enumerate(torch.linspace(0, 1, config.num_samples))):
    y = alpha*y1 + (1-alpha)*y2
    y = y.to(device)
    x = gen(y)
    x = x/2 + 0.5
    torchvision.utils.save_image(x, os.path.join(config.output, "img/%.6i.png"%i), nrow=10)

command = "ffmpeg -i {} {}".format(os.path.join(config.output, "img/%06d.png"), os.path.join(config.output, "video.mp4"))
subprocess.run(command.split())
shutil.rmtree(os.path.join(config.output, "img"))
