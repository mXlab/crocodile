###################import statements########################################
from dataset import CrocodileDataset, SequenceSampler
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from torch import optim
import utils
import torch
from torch import autograd
import os
import argparse
import time
import json
import models
from tensorboardX import SummaryWriter
import numpy as np
import logger

###################### Global Variables ###############################
BIODATA_DEFAULT = "LaurenceHBS-Nov919mins1000Hz-Heart+GSR-2channels.csv"
SAMPLING_RATE_DEFAULT = 1000
FPS = 30000/1001
DEFAULT_CONFIG = dict(distance_heart=400, width_heart=100, prominence_heart=0.01,
                      width_eda=600, distance_eda=1800, prominence_eda=0.0014, sampling_rate=1000) 


class Config():
      #This class is an argument parser to configure training process

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
        parser.add_argument('-z', '--num-latent', default=5, type=int)
        parser.add_argument('--seed', default=1234, type=int)
        parser.add_argument('-gp', '--gradient-penalty', default=0, type=int)
        parser.add_argument('--spectral-norm-gen', action="store_true")
        parser.add_argument('-nl', '--num-layers', default=4, type=int)
        parser.add_argument('--path-to-dataset', default="/checkpoint/hberard/crocodile", type=str)
        parser.add_argument('--path-to-biodata', default=None, type=str)
        parser.add_argument('--normalization', default="normalized", choices=("standardized", "normalized"))
        parser.add_argument('--length-sequence', default=150, type=int)
        parser.add_argument('--num-sequences', default=10, type=int)
        parser.add_argument('--num-variations', default=10, type=int)

        self.parser = parser

    def parse_args(self):   # function to parse the arguments       
        args = self.parser.parse_args()

        if args.path_to_biodata is None:
            args.path_to_biodata = os.path.join(args.path_to_dataset, BIODATA_DEFAULT)

        return args


class Preprocessing:
    def __init__(self, distance_heart=None, width_heart=None, prominence_heart=None,
                 distance_eda=None, width_eda=None, prominence_eda=None,
                 sampling_rate=SAMPLING_RATE_DEFAULT, normalization=None):
        self.sampling_rate = sampling_rate
        self.distance_heart = distance_heart
        self.width_heart = width_heart
        self.prominence_heart = prominence_heart

        self.distance_eda = distance_eda
        self.width_eda = width_eda
        self.prominence_eda = prominence_eda
        self.normalization = normalization

    def __call__(self, signal):
        import biodata
        from scipy.signal import find_peaks
        from biosppy.signals.tools import get_heart_rate, smoother

        list_smoothing_heart = [None, 10, 100]
        list_smoothing_rate = [None, 1000, 10000]
        list_smoothing_eda = [None, 10]
        list_features = []

        heart_raw = biodata.enveloppe_filter(signal[:, 1])
        heart_peaks, heart_properties = find_peaks(heart_raw, distance=self.distance_heart, width=self.width_heart, prominence=self.prominence_heart)
        for smoothing in list_smoothing_heart:
            if smoothing is None:
                size = 1
                smoothing = False
            else:
                size = smoothing
                smoothing = True

            bpm = get_heart_rate(heart_peaks, sampling_rate=self.sampling_rate, smooth=smoothing, size=size)
            amplitudes = heart_properties["prominences"]
            if smoothing:
                amplitudes, _ = smoother(signal=amplitudes, kernel='boxcar', size=size, mirror=True)

            bpm = biodata.interpolate(bpm[1], bpm[0], len(signal))
            amplitudes = biodata.interpolate(amplitudes, heart_peaks, len(signal))

            list_features += [bpm, amplitudes]

        eda_raw = biodata.enveloppe_filter(signal[:, 2])
        eda_peaks, eda_properties = find_peaks(eda_raw, distance=self.distance_eda, width=self.width_eda, prominence=self.prominence_eda)
        for smoothing in list_smoothing_eda:
            if smoothing is None:
                size = 1
                smoothing = False
            else:
                size = smoothing
                smoothing = True
            
            intervals = biodata.compute_intervals(eda_peaks, smooth=smoothing, size=size)
            amplitudes = eda_properties["prominences"]
            if smoothing:
                amplitudes, _ = smoother(signal=amplitudes, kernel='boxcar', size=size, mirror=True)
            intervals = biodata.interpolate(intervals, eda_peaks, len(signal))
            amplitudes = biodata.interpolate(amplitudes, eda_peaks, len(signal))

            list_features += [intervals, amplitudes]

        for smoothing in list_smoothing_rate:
            if smoothing is None:
                size = 1
                smoothing = False
            else:
                size = smoothing
                smoothing = True

            rate = biodata.rate_of_change(eda_raw, size=size)
            list_features.append(rate)

        features = np.array(list_features).transpose()
        if self.normalization is None:
            pass
        elif self.normalization == "standardized":
            features = (features - features.mean(0))/features.std(0)
        elif self.normalization == "normalized":
            features = (features - features.min(0))/(features.max(0)-features.min(0))
        else:
            raise ValueError
        
        return torch.tensor(features).float()
    

def run(args):
    if args.slurm:
        args.slurmid = "%s_%s" % (os.environ["SLURM_JOB_ID"],os.environ["SLURM_ARRAY_TASK_ID"])
   
   ### Convert passed arguments to variable ###  
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

    # set device to gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_name = "%i_%i" % (int(time.time()), np.random.randint(9999))
    if args.slurmid is not None:
        exp_name = args.slurmid
    OUTPUT_PATH = os.path.join(args.output_path, '%i/%s') % (RESOLUTION, exp_name)
    #writer = SummaryWriter(log_dir=os.path.join(OUTPUT_PATH, 'runs'))
    writer = logger.Logger(OUTPUT_PATH)

    print("Loading dataset...")

    preprocessing = Preprocessing(**DEFAULT_CONFIG)

    #transform dataset to tensor and normalize it
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   
     #create instance of CrocodileDataset and DataLoader class
    dataset = CrocodileDataset(root=ROOT, transform=transform, feature_transform=None, resolution=RESOLUTION, one_hot=True,
                               biodata=args.path_to_biodata, preprocessing=preprocessing)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    sampler = SequenceSampler(dataset, args.length_sequence, shuffle=False)
    testloader = DataLoader(dataset, batch_sampler=sampler, num_workers=1)

    print("Init...")

#create small generator network and the small discriminator network based on passed arguments
    if args.model == "small":
        gen = models.SmallGenerator(NUM_Z+dataset.num_features, RESOLUTION, NUM_FILTERS, args.num_layers, spectral_norm=args.spectral_norm_gen).to(device)
        dis = models.ConditionalSmallDiscriminator(RESOLUTION, dataset.num_features, NUM_FILTERS, args.num_layers).to(device)


    #run adam algorithm optimization on model
    gen_optimizer = optim.Adam(gen.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(dis.parameters(), lr=LR_DIS, betas=(0.5, 0.999))

 #create missing "gen" and "img" directories if they dont exist
    if not os.path.exists(os.path.join(OUTPUT_PATH, "gen")):
        os.makedirs(os.path.join(OUTPUT_PATH, "gen"))
    if not os.path.exists(os.path.join(OUTPUT_PATH, "img")):
        os.makedirs(os.path.join(OUTPUT_PATH, "img"))

    dataiter = iter(testloader)
    z_examples = torch.zeros(args.num_variations, NUM_Z).normal_()
    x_examples = []
    features_examples = []
    for i in range(args.num_sequences):
        x, _, features = dataiter.next()
        x = x/2 + 0.5
        x_examples.append(x)
        features_examples.append(features)
    x_examples = torch.stack(x_examples)
    features_examples = torch.stack(features_examples)

    features_examples = features_examples.view(args.num_sequences, 1, args.length_sequence, -1).expand(-1, args.num_variations, -1, -1)
    features_examples = features_examples.reshape(args.num_sequences*args.num_variations*args.length_sequence, -1)
    
    z_examples = z_examples.view(1, args.num_variations, 1, -1).expand(args.num_sequences, -1, args.length_sequence, -1)
    z_examples = z_examples.reshape(args.num_sequences*args.num_variations*args.length_sequence, -1)

    #writer.add_video(x_examples, 0, fps=FPS, nrow=args.num_sequences)

    writer.add_hparams(args)


############################ Training Loop ###################################
    print("Training...")
    init_epoch = 0
    for epoch in range(NUM_EPOCHS):
        t = time.time()
        for x, _, features in dataloader:
            x = x.to(device)
            features = features.to(device)
            z = torch.zeros(len(x), NUM_Z).normal_().to(device)
            z = torch.cat([z, features], -1)

            x_gen = gen(z)
            score_true, score_gen = dis(x, features), dis(x_gen, features)
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

        print("Epoch: %i, Loss dis: %.2e, Loss gen %.2e, Time: %i" % (init_epoch+epoch, loss_dis, loss_gen, time.time()-t))

        with torch.no_grad():
            x_gen = x_gen/2 + 0.5
            writer.add_image(x_gen, init_epoch+epoch)

            list_samples = []
            for j in range(25):
                _, _, features = dataiter.next()
                z = torch.zeros(1, NUM_Z).normal_().expand(args.length_sequence, -1)
                z = torch.cat([z, features], -1).to(device)
                x_gen = gen(z)
                x_gen = x_gen/2 + 0.5
                list_samples.append(x_gen.cpu())
            list_samples = torch.stack(list_samples, 0)
            writer.add_video(list_samples, init_epoch+epoch, fps=FPS, nrow=25)

            """
            list_samples = []
            for i in range(0, args.num_sequences*args.num_variations*args.length_sequence, BATCH_SIZE):
                z = z_examples[i:i+BATCH_SIZE]
                features = features_examples[i:i+BATCH_SIZE]
                z = torch.cat([z, features], -1).to(device)
                x_gen = gen(z)
                x_gen = x_gen/2 + 0.5
                list_samples.append(x_gen.cpu())

            list_samples = torch.cat(list_samples, 0).view(args.num_sequences, args.num_variations, args.length_sequence, 3, RESOLUTION, RESOLUTION)
            list_samples = torch.cat([x_examples.unsqueeze(1), list_samples], 1).view(-1, args.length_sequence, 3, RESOLUTION, RESOLUTION)
            writer.add_video(list_samples, init_epoch+epoch, fps=FPS, nrow=args.num_sequences)
            """
       ##saves models to file
        torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict()},
                   os.path.join(OUTPUT_PATH, "gen/gen_%i.chk" % (init_epoch+epoch)))

        torch.save({'epoch': init_epoch+epoch, 'gen_state_dict': gen.state_dict(),
                    'dis_state_dict': dis.state_dict(), 'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                    'dis_optimizer_state_dict': dis_optimizer.state_dict()},
                   os.path.join(OUTPUT_PATH, "last_model.chk"))


########################### Main loop ###################################
if __name__ == "__main__":
    run(Config().parse_args())