from pathlib import Path
from dataclasses import dataclass
import numpy as np


class Biodata:
    @dataclass
    class Params:
        time_window: int = 5000  # The time window in milliseconds

    def __init__(self, path: Path, config, params: Params = Params()):
        super().__init__()

        self.data = np.loadtxt(path / config.sensor_file,
                               delimiter=',', usecols=config.usecols)
        self.config = config
        self.window_size = int((params.time_window *
                                config.sampling_rate / 1000) // 2)

        self.seq_length = self.window_size*2
        self.dim = self.data.shape[1]

    def convert_index(self, index):
        return int((index - self.config.start_video) / self.config.fps *
                   self.config.sampling_rate + self.config.start_sensor)

    def __getitem__(self, index):
        assert index >= 0  # Only accepts positive index
        index = self.convert_index(index)
        assert index - self.window_size >= 0
        assert index < len(self)
        return self.data[index - self.window_size: index + self.window_size]

    def __len__(self):
        return len(self.data) - self.window_size
