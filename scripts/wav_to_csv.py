import soundfile
import csv
import argparse
import subprocess

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("wav_file", type=str, help="File constaining the raw data")
parser.add_argument("--csv_file", default=None, type=str, help="File where to output the CSV data")
parser.add_argument("--n_channels", type=int, default=4)
parser.add_argument("--samplerate", type=int, default=1000)
args = parser.parse_args()

wavfile = args.wav_file
csvfile = args.csv_file
if csvfile is None:
    csvfile = wavfile.replace(".wav", ".csv")


print("Preprocessing wav file...")
subprocess.run(["./scripts/fix_wav_format.sh", wavfile, wavfile, str(args.n_channels), str(args.samplerate)], check=True)

duration = float(19*60)

#data, _ = soundfile.read(wavfile)
data, samplerate = soundfile.read(wavfile)

n_samples = len(data)
#step = duration/n_samples
step = 1.0/samplerate

print("Converting file...")
with open(csvfile, 'w') as csvfile:
    writer = csv.writer(csvfile)
    t = 0
    for d in data:
        writer.writerow([round(t, 3), d[0], d[1]])
        t += step
