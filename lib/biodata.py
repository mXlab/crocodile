import numpy as np
from biosppy.signals.tools import smoother


# Utility function to filter the signal by computing the envelope of the signal
def enveloppe_filter(x, threshold=1e-2):
    for i in range(1, len(x)):
        mask = x[i] <= threshold
        x[i] = x[i-1]*mask + x[i]*(~mask)
    return x


def interpolate(x, peaks, length):
    new_signal = np.zeros(length)
    new_signal[peaks] = x
    new_signal = enveloppe_filter(new_signal)
    return new_signal



def rate_of_change(x, size=1):
    rate = [0]*size
    for i in range(size, len(x)-size):
        r = (x[i+size]-x[i-size])/(2*size)
        rate.append(r)
    rate += [0]*size
    assert len(rate) == len(x)
    return np.array(rate)


def compute_intervals(peaks, smooth=False, size=3):
    intervals = [0]
    for i in range(len(peaks)-1):
        intervals.append(peaks[i+1]-peaks[i])
    intervals = np.array(intervals)
    
    if smooth and (len(intervals) > 1):
        intervals, _ = smoother(signal=intervals, kernel='boxcar', size=size, mirror=True)

    return intervals