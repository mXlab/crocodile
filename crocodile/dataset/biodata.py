from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import Tuple


class Biodata:
    @dataclass
    class Params:
        time_window: int = 5000  # The time window in milliseconds

    def __init__(self, path: Path, sampling_rate: int = 1000, usecols: Tuple[int,...] = (1,2), params: Params = Params()):
        super().__init__()

        self.data = np.loadtxt(path , delimiter=',', usecols=usecols)
        self.window_size = int((params.time_window * sampling_rate / 1000) // 2)

        self.seq_length = self.window_size*2
        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        return self.data[index - self.window_size: index + 2*self.window_size]

    def __len__(self):
        return len(self.data) - 2*self.window_size
