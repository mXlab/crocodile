import soundfile
import csv
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("wav_file", type=str, help="File constaining the raw data")
#parser.add_argument("duration", type=str, help="Duration of the wav file")
parser.add_argument("csv_file", type=str, help="File where to output the CSV data")
args = parser.parse_args()

wavfile = args.wav_file
csvfile = args.csv_file

duration = float(19*60)

#data, _ = soundfile.read(wavfile)
data, samplerate = soundfile.read(wavfile)

n_samples = len(data)
#step = duration/n_samples
step = 1.0/samplerate

with open(csvfile, 'w') as csvfile:
    writer = csv.writer(csvfile)
    t = 0
    for d in data:
        writer.writerow([round(t, 3), d[0], d[1]])
        t += step
