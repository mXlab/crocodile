import numpy as np


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