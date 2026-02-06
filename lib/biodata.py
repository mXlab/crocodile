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


# ============================================================================
# Classes merged from biodatapy/ package
# ============================================================================

class MinMax:
    def __init__(self):
        self.input = None
        self.min = None
        self.max = None
        self.value = None
        self.firstPass = None

        self.reset()

    def reset(self) -> None:
        self.input = 0
        self.min = 0
        self.max = 0
        self.value = 0
        self.firstPass = True

    def adapt(self, lop: float) -> None:
        self.lop = np.clip(lop, 0, 1)
        self.lop = self.lop * self.lop

        self.min += (self.input - self.min) * self.lop
        self.max += (self.input - self.max) * self.lop

    def filter(self, f: float) -> float:
        self.input = f

        if ( self.firstPass ):
            self.firstPass = False
            self.min = f
            self.max = f
        else:
            if ( f > self.max ): self.max = f
            if ( f < self.min ): self.min = f

        if ( self.max == self.min ):
            self.value = 0.5
        else:
            self.value = (f - self.min) / ( self.max - self.min)

        return self.value

    def getMax(self) -> float:
      return self.max

    def getMin(self) -> float:
      return self.min


class Threshold:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        self.triggered = False

    def detect(self, value: float) -> bool:
        if ( value >= self.upper and self.triggered == False ):
            self.triggered = True
            return True
        elif ( value <= self.lower):
            self.triggered = False
        return False


class Lop:
    def __init__(self, alpha_: float = 0.01):
        # Low-pass smoothing factor.
        self.alpha = None

        # Current value.
        self.value = None

        # N. samples seen thus far.
        self.n = None

        # N. samples in calibration phase.
        self.nCalibration = None

        self.setSmoothing(alpha_)
        self.reset()

    # Resets filter.
    def reset(self) -> None:
        self.value = 0
        self.n     = 0

    # Sets smoothing factor to value in [0, 1] (lower value = smoother).
    def setSmoothing(self, alpha_: float) -> None:
        # Constrains the smoothing factor in [0, 1].
        self.alpha = np.clip(alpha_, 0, 1)

        # Rule of thumb that maps the smoothing factor to number of samples.
        self.nCalibration = int(2 / self.alpha - 1)


    # Filters sample and returns smoothed value.
    def filter(self, input: float) -> float:
        # For the first #nCalibration# samples just compute the average.
        if (self.n < self.nCalibration):
            self.n += 1
            self.value = (self.value * (self.n-1) + input) / self.n
        # After that: switch back to exponential moving average.
        else:
            self.value += (input - self.value) * self.alpha
        return self.value


import time

millis = lambda: time.time_ns() // 1000000
micros = lambda: time.time_ns() // 1000

class Heart():
    def __init__(self, stream):
        # This is the data stream from which we read
        self.data_stream = stream

        self.bpmChronoStart = None

        self.heartMinMax = MinMax()
        self.heartThresh = Threshold(0.25, 0.4)
        self.heartMinMaxSmoothing = 0.1

        self.heartSensorAmplitudeLop = Lop(0.001)
        self.heartSensorBpmLop = Lop(0.001)

        self.heartSensorAmplitudeLopValue = None

        self.heartSensorBpmLopValue = None
        self.heartSensorAmplitudeLopValueMinMax = MinMax()
        self.heartSensorAmplitudeLopValueMinMaxSmoothing = 0.001

        self.heartSensorAmplitudeLopValueMinMaxValue = None
        self.heartSensorBpmLopValueMinMax= MinMax()
        self.heartSensorBpmLopValueMinMaxSmoothing = 0.001

        self.heartSensorBpmLopValueMinMaxValue = None

        self.heartSensorFiltered = None
        self.heartSensorAmplitude = None

        self.heartSensorReading = None

        self.bpm = None  # this value is fed to initialize your BPM before a heartbeat is detected

        self.beat = None

        # Internal use.
        self.reset()

    def setAmplitudeSmoothing(self, smoothing: float) -> None:
        self.heartSensorAmplitudeLop.setSmoothing(smoothing)

    def setBpmSmoothing(self, smoothing: float) -> None:
        self.heartSensorBpmLop.setSmoothing(smoothing)

    def setAmplitudeMinMaxSmoothing(self, smoothing: float) -> None:
        self.heartSensorAmplitudeLopValueMinMaxSmoothing = np.clip(smoothing, 0, 1)

    def setBpmMinMaxSmoothing(self, smoothing: float) -> None:
        self.heartSensorBpmLopValueMinMaxSmoothing = np.clip(smoothing, 0, 1)

    def setMinMaxSmoothing(self, smoothing: float) -> None:
        self.heartMinMaxSmoothing = np.clip(smoothing, 0, 1)

    def reset(self) -> None:
        self.heartMinMax.reset()
        self.heartSensorAmplitudeLop.reset()
        self.heartSensorBpmLop.reset()
        self.heartSensorAmplitudeLopValueMinMax.reset()
        self.heartSensorBpmLopValueMinMax.reset()

        self.heartSensorReading = 0
        self.heartSensorFiltered = 0
        self.heartSensorAmplitude = 0
        self.bpmChronoStart = 0

        self.bpm = 60
        self.beat = False

        # Perform one update.
        self.sample()

    """
     * Reads the signal and perform filtering operations. Call this before
     * calling any of the access functions. This function takes into account
     * the sample rate.
    """
    def update(self) -> None:
        self.sample()

    # Get normalized heartrate signal.
    def getNormalized(self) -> float:
        return self.heartSensorFiltered

    # Returns true if a beat was detected during the last call to update().
    def beatDetected(self) -> bool:
        return self.beat

    # Returns BPM (beats per minute).
    def getBPM(self) -> float:
        return self.bpm

    # Returns raw signal as returned by analogRead().
    def getRaw(self) -> int:
        return self.heartSensorReading

    # Returns the average amplitude of signal mapped between 0.0 and 1.0.
    """ For example, if amplitude is average, returns 0.5,
     * if amplitude is below average, returns < 0.5
     * if amplitude is above average, returns > 0.5.
    """
    def amplitudeChange(self) -> float:
        return self.heartSensorAmplitudeLopValueMinMaxValue

    #Returns the average bpm of signal mapped between 0.0 and 1.0.
    """ For example, if bpm is average, returns 0.5,
     * if bpm is below average, returns < 0.5
     * if bpm is above average, returns > 0.5.
    """
    def bpmChange(self) -> float:
        return self.heartSensorBpmLopValueMinMaxValue

    # Performs the actual adjustments of signals and filterings.
    # Internal use: don't use directly, use update() instead.
    def sample(self):
        # Read analog value if needed.
        self.heartSensorReading, ms = self.data_stream.read()

        self.heartSensorFiltered = self.heartMinMax.filter(self.heartSensorReading)
        self.heartSensorAmplitude = self.heartMinMax.getMax() - self.heartMinMax.getMin()
        self.heartMinMax.adapt(self.heartMinMaxSmoothing) # APPLY A LOW PASS ADAPTION FILTER TO THE MIN AND MAX

        self.heartSensorAmplitudeLopValue = self.heartSensorAmplitudeLop.filter(self.heartSensorAmplitude)
        self.heartSensorBpmLopValue =  self.heartSensorBpmLop.filter(self.bpm)

        self.heartSensorAmplitudeLopValueMinMaxValue = self.heartSensorAmplitudeLopValueMinMax.filter(self.heartSensorAmplitudeLopValue)
        self.heartSensorAmplitudeLopValueMinMax.adapt(self.heartSensorAmplitudeLopValueMinMaxSmoothing)
        self.heartSensorBpmLopValueMinMaxValue = self.heartSensorBpmLopValueMinMax.filter(self.heartSensorBpmLopValue)
        self.heartSensorBpmLopValueMinMax.adapt(self.heartSensorBpmLopValueMinMaxSmoothing)

        self.beat = self.heartThresh.detect(self.heartSensorFiltered)

        if ( self.beat ):
            temporaryBpm = 60000. / (ms - self.bpmChronoStart)
            self.bpmChronoStart = ms
            #print(temporaryBpm)
            if ( temporaryBpm > 30 and temporaryBpm < 200 ): # make sure the BPM is within bounds
                self.bpm = temporaryBpm