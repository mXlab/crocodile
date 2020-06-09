import submitit
from train import run, get_config
import copy
import numpy as np
import time
import os

args = get_config()
args.slurm = True
args.num_epochs = 100

NUM_CONFIG = 25
NUM_SEEDS = 100
partition = "learnfair"
total_time = 24*60 
NAME = "video_only_2"
PROJECT_NAME = "crocodile"

args.resolution = 128


def run_with_logger(args):
    from logger import ExpvizLogger
    log_dir = os.path.join(args.output_path, "exp_%i_%i/"%(int(time.time()), np.random.randint(9999)))
    logger = ExpvizLogger(projectname=PROJECT_NAME, hyperparams=args, log_dir=log_dir, expname=NAME)
    run(args, logger)


def generate_config(num_config):
    list_configs = []
    for i in range(num_config):
        config = copy.deepcopy(args)
        config.num_filters = int(np.random.choice([128, 256, 512]))
        config.learning_rate_dis = 1e-4
        config.learning_rate_gen = 1e-4
        config.batch_size = 128
        config.num_latent = int(np.random.choice([10, 50, 100]))
        config.seed = int(np.random.randint(NUM_SEEDS))
        #config.gradient_penalty = bool(np.random.choice([True, False]))
        #config.spectral_norm_gen = bool(np.random.choice([True, False]))
        config.num_layers = int(np.random.choice([3, 4, 5]))

        list_configs.append(config)
    return list_configs


list_configs = generate_config(NUM_CONFIG)

# the AutoExecutor class is your interface for submitting function to Slurm or
# Chronos the specified folder is used to dump job information, logs and result
# when finished %j is replaced by the job id at runtime
executor = submitit.AutoExecutor(folder="{}/log/%j".format(args.output_path))
# specify sbatch parameters (here it will timeout after 4min, and run on dev)
# This is where you would specify num_gpus=1 for instance
# If you run this on Chronos, the partition argument is ignore 
# (as there's no equivalent for crun)
executor.update_parameters(timeout_min=total_time, partition=partition, gpus_per_node=1,
                           array_parallelism=100)
# The submission interface is identical to concurrent.futures.Executor
jobs = executor.map_array(run_with_logger, list_configs)

print("Running...")
results = [j.results() for j in jobs]
