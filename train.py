from dataset import CrocodileDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.nn.utils import spectral_norm
from torch import optim
import utils
import torch
from torch import autograd
import os
import argparse
import time
import json
import models

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="small", choices=("small",))
parser.add_argument('-e', '--num-epochs', default=1000, type=int)
parser.add_argument('-r', '--resolution', default=128, type=int)
parser.add_argument('-f', '--num-filters', default=128, type=int)
parser.add_argument('-lrd','--learning-rate-dis', default=1e-3, type=float)
parser.add_argument('-lrg','--learning-rate-gen', default=1e-3, type=float)
parser.add_argument('--ema', default=0, type=float)
parser.add_argument('-bs', '--batch-size', default=64, type=int)
parser.add_argument('-z', '--num-latent', default=50, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('-gp', '--gradient-penalty', default=0, type=int)
parser.add_argument('--spectral-norm-gen', action="store_true")
parser.add_argument('-nl', '--num-layers', default=6, type=int)
parser.add_argument('--path-to-dataset', default="/network/tmp1/berardhu/crocodile/data", type=str)
parser.add_argument('--output-path', default="/network/tmp1/berardhu/crocodile/results/", type=str)
args = parser.parse_args()

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
OUTPUT_PATH = os.path.join(args.output_path, "exp_%i/"%int(time.time()))
ROOT = args.path_to_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CrocodileDataset(root=ROOT, transform=transform, resolution=RESOLUTION)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

print("Init...")

if args.model is "small":
    gen = models.SmallGenerator(NUM_Z, RESOLUTION, NUM_FILTERS, args.num_layers, spectral_norm=args.spectral_norm_gen).to(device)
    dis = models.SmallDiscriminator(RESOLUTION, NUM_FILTERS, args.num_layers).to(device)

gen_optimizer = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(dis.parameters(), lr=LR_DIS, betas=(0.5, 0.999))

z_examples = torch.zeros(100, NUM_Z).normal_().to(device)

if not os.path.exists(os.path.join(OUTPUT_PATH, "gen")):
    os.makedirs(os.path.join(OUTPUT_PATH, "gen"))
if not os.path.exists(os.path.join(OUTPUT_PATH, "img")):
    os.makedirs(os.path.join(OUTPUT_PATH, "img"))

dataiter = iter(dataloader)
x_examples = dataiter.next()[:100]
x_examples = x_examples/2 + 0.5
torchvision.utils.save_image(x_examples, os.path.join(OUTPUT_PATH, "examples.png"), nrow=10)

with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

print("Training...")
init_epoch = 0
for epoch in range(NUM_EPOCHS):
    t = time.time()
    for x in dataloader:
        x = x.to(device)
        z = torch.zeros(len(x), NUM_Z).normal_().to(device)

        x_gen = gen(z)
        score_true, score_gen = dis(x), dis(x_gen)
        loss_gen, loss_dis = utils.compute_loss(score_true, score_gen, mode="nsgan")
        if GRADIENT_PENALTY:
            loss_dis += GRADIENT_PENALTY*dis.get_penalty(x, x_gen)

        grad_gen = autograd.grad(loss_gen, gen.parameters(), retain_graph=True)
        grad_dis = autograd.grad(loss_dis, dis.parameters(), retain_graph=True)

        for p, g in zip(gen.parameters(), grad_gen):
            p.grad = g

        for p, g in zip(dis.parameters(), grad_dis):
            p.grad = g

        gen_optimizer.step()
        dis_optimizer.step()

    print("Epoch: %i, Loss dis: %.2e, Loss gen %.2e, Time: %i"%(init_epoch+epoch, loss_dis, loss_gen, time.time()-t))

    x_gen = gen(z_examples)
    x_gen = x_gen/2 + 0.5
    torchvision.utils.save_image(x_gen, os.path.join(OUTPUT_PATH, "img/img_%i.png"%(init_epoch+epoch)), nrow=10)

    torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict()},
                os.path.join(OUTPUT_PATH, "gen/gen_%i.chk"%(init_epoch+epoch)))

    torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict(),
                'dis_state_dict': dis.state_dict(), 'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'dis_optimizer_state_dict': dis_optimizer.state_dict()},
                os.path.join(OUTPUT_PATH, "last_model.chk"))
