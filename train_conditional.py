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
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class Config():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('output_path')
        parser.add_argument('-m', '--model', default="small", choices=("small",))
        parser.add_argument('--slurmid', default=None)
        parser.add_argument('--slurm', action="store_true")
        parser.add_argument('-e', '--num-epochs', default=1000, type=int)
        parser.add_argument('-r', '--resolution', default=128, type=int)
        parser.add_argument('-f', '--num-filters', default=256, type=int)
        parser.add_argument('-lrd','--learning-rate-dis', default=5e-3, type=float)
        parser.add_argument('-lrg','--learning-rate-gen', default=2e-3, type=float)
        parser.add_argument('--ema', default=0, type=float)
        parser.add_argument('-bs', '--batch-size', default=64, type=int)
        parser.add_argument('-z', '--num-latent', default=10, type=int)
        parser.add_argument('--seed', default=1234, type=int)
        parser.add_argument('-gp', '--gradient-penalty', default=0, type=int)
        parser.add_argument('--spectral-norm-gen', action="store_true")
        parser.add_argument('-nl', '--num-layers', default=4, type=int)
        parser.add_argument('--path-to-dataset', default="/checkpoint/hberard/crocodile", type=str)

        self.parser = parser

    def parse_args(self):       
        return self.parser.parse_args()

def run(args):
    if args.slurm:
        args.slurmid = "%s_%s"%(os.environ["SLURM_JOB_ID"],os.environ["SLURM_ARRAY_TASK_ID"])
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

    exp_name = "%i_%i"%(int(time.time()), np.random.randint(9999))
    if args.slurmid is not None:
        exp_name = args.slurmid
    OUTPUT_PATH = os.path.join(args.output_path, '%i/%s')%(RESOLUTION,exp_name)
    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_PATH, 'runs'))

    print("Loading dataset...")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CrocodileDataset(root=ROOT, transform=transform, resolution=RESOLUTION, one_hot=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Init...")

    if args.model == "small":
        gen = models.SmallGenerator(NUM_Z+dataset.num_cat, RESOLUTION, NUM_FILTERS, args.num_layers, spectral_norm=args.spectral_norm_gen).to(device)
        dis = models.ConditionalSmallDiscriminator(RESOLUTION, dataset.num_cat, NUM_FILTERS, args.num_layers).to(device)

    gen_optimizer = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(dis.parameters(), lr=LR_DIS, betas=(0.5, 0.999))

    z_examples = torch.zeros(1, 10, NUM_Z).normal_().expand(dataset.num_cat, -1, -1)
    y_examples = torch.eye(dataset.num_cat).unsqueeze(1).expand(-1, 10, -1)
    z_examples = torch.cat([z_examples, y_examples], -1).view(-1, NUM_Z+dataset.num_cat).to(device)

    if not os.path.exists(os.path.join(OUTPUT_PATH, "gen")):
        os.makedirs(os.path.join(OUTPUT_PATH, "gen"))
    if not os.path.exists(os.path.join(OUTPUT_PATH, "img")):
        os.makedirs(os.path.join(OUTPUT_PATH, "img"))

    dataiter = iter(dataloader)
    x_examples,_ = dataiter.next()[:100]
    x_examples = x_examples/2 + 0.5
    torchvision.utils.save_image(x_examples, os.path.join(OUTPUT_PATH, "examples.png"), nrow=10)

    with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    print("Training...")
    init_epoch = 0
    for epoch in range(NUM_EPOCHS):
        t = time.time()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            z = torch.zeros(len(x), NUM_Z).normal_().to(device)
            z = torch.cat([z,y], -1)

            x_gen = gen(z)
            score_true, score_gen = dis(x, y), dis(x_gen, y)
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


        x_gen = x_gen/2 + 0.5
        img = torchvision.utils.make_grid(x_gen, nrow=10)
        writer.add_image('gen_random', img, epoch)

        x_gen = gen(z_examples)
        x_gen = x_gen/2 + 0.5
        img = torchvision.utils.make_grid(x_gen, nrow=10) # First dimension is row, second dimension is column
        writer.add_image('gen', img, epoch)
        torchvision.utils.save_image(x_gen, os.path.join(OUTPUT_PATH, "img/img_%.3i.png"%(init_epoch+epoch)), nrow=10)

        torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict()},
                    os.path.join(OUTPUT_PATH, "gen/gen_%i.chk"%(init_epoch+epoch)))

        torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict(),
                    'dis_state_dict': dis.state_dict(), 'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                    'dis_optimizer_state_dict': dis_optimizer.state_dict()},
                    os.path.join(OUTPUT_PATH, "last_model.chk"))

if __name__ == "__main__":
    run(Config().parse_args())