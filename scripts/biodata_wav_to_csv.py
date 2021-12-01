import soundfile
import csv

#wavfile = "LaurenceHBS-Nov919mins1000Hz-Heart+GSR-2channels.wav"
#csvfile = "LaurenceHBS-Nov919mins1000Hz-Heart+GSR-2channels.csv"

wavfile = "LaurenceHBSinfo-Nov919mins1000Hz-2channels.wav"
csvfile = "LaurenceHBSinfo-Nov919mins1000Hz-2channels.csv"

duration = float(19*60)

data, _ = soundfile.read(wavfile)

n_samples = len(data)
step = duration/n_samples

with open(csvfile, 'w') as csvfile:
    writer = csv.writer(csvfile)
    t = 0
    for d in data:
        writer.writerow([t, d[0], d[1]])
        t += step
