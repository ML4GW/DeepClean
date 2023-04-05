from dataclasses import dataclass

import numpy as np
from scipy.signal import windows

from deepclean.utils.filtering import Frequency, BandpassFilter


@dataclass
class Cleaner:
    kernel_length: float
    sample_rate: float
    filter_pad: float
    freq_low: Frequency = None
    freq_high: Frequency = None

    def __post_init__(self):
        self.bandpass = BandpassFilter(
            self.freq_low, self.freq_high, self.sample_rate
        )
        self.pad_size = int(self.filter_pad * self.sample_rate)
        self.kernel_size = int(self.kernel_length * self.sample_rate)
        self.window = windows.hann(2 * self.pad_size)

    def __call__(self, noise: np.ndarray, strain: np.ndarray) -> np.ndarray:
        noise[: self.pad_size] *= self.window[: self.pad_size]
        noise[-self.pad_size :] *= self.window[-self.pad_size :]
        noise = self.bandpass(noise)

        start = -self.pad_size - self.kernel_size
        stop = -self.pad_size
        noise = noise[start:stop]
        return strain - noise
