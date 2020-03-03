import numpy as np
import time
from .minmax import MinMax
from .threshold import Threshold
from .lop import Lop

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


