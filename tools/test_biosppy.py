import csv
import biosppy.signals.bvp as bvp
import biosppy.signals.eda as eda

import numpy as np

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("csv_file", type=str, help="File constaining the raw data")
parser.add_argument('-s', '--sampling-rate', type=int, default=1000, help="The sample rate")
parser.add_argument('-m', '--min-amplitude', type=float, default=0.1, help="Min amplitude (for EDA)")
args = parser.parse_args()

signals = np.loadtxt(args.csv_file, delimiter=',')
t = signals[:,0]
bvp_raw = signals[:,1]
eda_raw = signals[:,2]

bvp_signal = bvp.bvp(bvp_raw, sampling_rate=args.sampling_rate)
eda_signal = eda.eda(eda_raw, sampling_rate=args.sampling_rate, min_amplitude=args.min_amplitude)

print(bvp_signal)
print(eda_signal)
#
# with open(csv_file_name, 'w') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
