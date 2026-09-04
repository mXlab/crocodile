import numpy as np


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