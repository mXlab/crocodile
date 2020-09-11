from lib.dataset import CrocodileDataset
from lib import utils
from lib import models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.nn.utils import spectral_norm
from torch import optim
import torch
from torch import autograd
import os
import argparse
import time
import json
from lib.fid import FID



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="small", choices=("small", "resnet"))
    parser.add_argument('--mode', default="hinge")
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
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--path-to-dataset', default="/network/tmp1/berardhu/crocodile/data", type=str)
    parser.add_argument('--output-path', default="/network/tmp1/berardhu/crocodile/results/", type=str)
    parser.add_argument('--compute-fid', action="store_true")
    args = parser.parse_args()
    return args


def run(args, logger=None):
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
    dataset = CrocodileDataset(root=ROOT, transform=transform, resolution=RESOLUTION, one_hot=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Init...")

    if args.compute_fid:
        path_to_stats = os.path.join(args.path_to_dataset, str(args.resolution), "crocodile_stats.pt")
        if not os.path.exists(path_to_stats):
            print("Computing stats for dataset...")
            mu, sigma = FID(device=device).compute_stats(dataloader)
            torch.save(dict(mu=mu, sigma=sigma), path_to_stats)

        fid_score = FID(path_to_stats, device=device)

    if args.model == "small":
        gen = models.SmallGenerator(NUM_Z, RESOLUTION, NUM_FILTERS, args.num_layers, spectral_norm=args.spectral_norm_gen)
        dis = models.SmallDiscriminator(RESOLUTION, NUM_FILTERS, args.num_layers)
    elif args.model == "resnet":
        gen = models.ResNetGenerator(NUM_Z, RESOLUTION, NUM_FILTERS, args.num_layers)
        dis = models.ResNetDiscriminator(RESOLUTION, NUM_FILTERS, args.num_layers)

    gen = gen.to(device)
    dis = dis.to(device)

    gen_optimizer = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(dis.parameters(), lr=LR_DIS, betas=(0.5, 0.999))

    z_examples = torch.zeros(100, NUM_Z).normal_().to(device)

    if not os.path.exists(os.path.join(OUTPUT_PATH, "gen")):
        os.makedirs(os.path.join(OUTPUT_PATH, "gen"))
    if not os.path.exists(os.path.join(OUTPUT_PATH, "img")):
        os.makedirs(os.path.join(OUTPUT_PATH, "img"))

    dataiter = iter(dataloader)
    x_examples, _ = dataiter.next()
    x_examples = x_examples/2 + 0.5
    torchvision.utils.save_image(x_examples[:100], os.path.join(OUTPUT_PATH, "examples.png"), nrow=10)

    with open(os.path.join(OUTPUT_PATH, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    print("Training...")
    init_epoch = 0
    for epoch in range(NUM_EPOCHS):
        t = time.time()
        for x, _ in dataloader:
            x = x.to(device)
            x_gen = gen.sample(len(x))

            score_true, score_gen = dis(x), dis(x_gen)
            loss_gen, loss_dis = utils.compute_loss(score_true, score_gen, mode=args.mode)
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

        x_gen = gen(z_examples)
        x_gen = x_gen/2 + 0.5

        fid = 0
        if args.compute_fid:
            fid = fid_score(gen)

        if logger is None:
            print("Epoch: %i, Loss dis: %.2e, Loss gen %.2e, FID: %.2e, Time: %i"%(init_epoch+epoch, loss_dis, loss_gen, fid, time.time()-t))
            torchvision.utils.save_image(x_gen, os.path.join(OUTPUT_PATH, "img/img_%i.png"%(init_epoch+epoch)), nrow=10)
        else:
            scalar_dict = dict(loss=loss_gen, loss_dis=loss_dis, loss_gen=loss_gen, fid=fid)
            logger.write(scalar_dict, epoch)

            x_gen = torchvision.utils.make_grid(x_gen, nrow=10)
            logger.add_image("gen", x_gen, epoch)

        torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict()},
                    os.path.join(OUTPUT_PATH, "gen/gen_%i.chk"%(init_epoch+epoch)))

        torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict(),
                    'dis_state_dict': dis.state_dict(), 'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                    'dis_optimizer_state_dict': dis_optimizer.state_dict()},
                    os.path.join(OUTPUT_PATH, "last_model.chk"))


if __name__ == "__main__":
    args = get_config()
    run(args)
