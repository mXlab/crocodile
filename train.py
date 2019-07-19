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

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resolution', default=128, type=int)
parser.add_argument('-f', '--num-filters', default=64, type=int)

args = parser.parse_args()

BATCH_SIZE = 128
NUM_Z = 128
NUM_FILTERS = args.num_filters
LR_GEN = 1e-3
LR_DIS = 1e-3
NUM_EPOCHS = 1000
SEED = 1234
RESOLUTION = args.resolution
torch.manual_seed(SEED)
OUTPUT_PATH = "/network/tmp1/berardhu/erin_project/results/exp_%i/"%int(time.time())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CrocodileDataset(transform, resolution=RESOLUTION)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

print("Init...")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.input = nn.Linear(NUM_Z, NUM_FILTERS*(RESOLUTION//8)**2)
        self.network = nn.Sequential(nn.BatchNorm2d(NUM_FILTERS),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(NUM_FILTERS, NUM_FILTERS//2, 4, stride=2, padding=1), # 8x8 -> 16x16
                                     nn.BatchNorm2d(NUM_FILTERS//2),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(NUM_FILTERS//2, NUM_FILTERS//4, 4, stride=2, padding=1), # 16x16 -> 32x32
                                     nn.BatchNorm2d(NUM_FILTERS//4),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(NUM_FILTERS//4, NUM_FILTERS//8, 4, stride=2, padding=1), # 32x32 -> 64x64
                                     nn.BatchNorm2d(NUM_FILTERS//8),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(NUM_FILTERS//8, 3, 3, stride=1, padding=1),
                                     nn.Tanh())

    def forward(self, x):
        x = self.input(x).view(-1, NUM_FILTERS, RESOLUTION//8, RESOLUTION//8)
        x = self.network(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(spectral_norm(nn.Conv2d(3, NUM_FILTERS//8, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//8, NUM_FILTERS//4, 4, stride=2, padding=1)), # 64x64 -> 32x32
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//4, NUM_FILTERS//4, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//4, NUM_FILTERS//2, 4, stride=2, padding=1)), # 32x32 -> 16x16
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//2, NUM_FILTERS//2, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS//2, NUM_FILTERS, 4, stride=2, padding=1)), # 16x16 -> 8x8
                                     nn.LeakyReLU(0.1, True),
                                     spectral_norm(nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3, stride=1, padding=1)),
                                     nn.LeakyReLU(0.1, True))
        self.output = spectral_norm(nn.Linear(NUM_FILTERS*(RESOLUTION//8)**2, 1))

    def forward(self, x):
        x = self.network(x).view(-1, NUM_FILTERS*(RESOLUTION//8)**2)
        x = self.output(x)
        return x

gen = Generator().to(device)
dis = Discriminator().to(device)

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

print("Training...")
init_epoch = 0
for epoch in range(NUM_EPOCHS):
    for x in dataloader:
        x = x.to(device)
        z = torch.zeros(len(x), NUM_Z).normal_().to(device)

        x_gen = gen(z)
        score_true, score_gen = dis(x), dis(x_gen)
        loss_gen, loss_dis = utils.compute_loss(score_true, score_gen, mode="nsgan")

        grad_gen = autograd.grad(loss_gen, gen.parameters(), retain_graph=True)
        grad_dis = autograd.grad(loss_dis, dis.parameters(), retain_graph=True)

        for p, g in zip(gen.parameters(), grad_gen):
            p.grad = g

        for p, g in zip(dis.parameters(), grad_dis):
            p.grad = g

        gen_optimizer.step()
        dis_optimizer.step()

    print("Epoch: %i, Loss dis: %.2e, Loss gen %.2e"%(init_epoch+epoch, loss_dis, loss_gen))

    x_gen = gen(z_examples)
    x_gen = x_gen/2 + 0.5
    torchvision.utils.save_image(x_gen, os.path.join(OUTPUT_PATH, "img/img_%i.png"%(init_epoch+epoch)), nrow=10)

    torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict()},
                os.path.join(OUTPUT_PATH, "gen/gen_%i.chk"%(init_epoch+epoch)))

    torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict(),
                'dis_state_dict': dis.state_dict(), 'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'dis_optimizer_state_dict': dis_optimizer.state_dict()},
                os.path.join(OUTPUT_PATH, "last_model.chk"))
