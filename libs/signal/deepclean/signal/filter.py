from typing import List, Union

from scipy import signal

from deepclean.signal import Op


class BandpassFilter(Op):
    def __init__(
        self,
        freq_low: Union[float, List[float]],
        freq_high: Union[float, List[float]],
        sample_rate: float,
        order: int = 8,
    ) -> None:
        try:
            num_bands = len(freq_low)
        except TypeError:
            # freq_low has no `__length__` property, so
            # assume it's a single number
            try:
                # check to make sure freq_high is also
                # a single number
                num_bands = len(freq_high)
            except TypeError:
                # it is, so wrap everything in a list to keep
                # things general
                freq_low = [freq_low]
                freq_high = [freq_high]
                num_bands = 1
            else:
                # it's not, this is a problem
                raise ValueError(
                    "If specifying multiple frequencies for "
                    "freq_high, must do the same for freq_low"
                )
        else:
            # freq_low _does_ have a `__length__` property,
            # so make sure that freq_high has length at all
            # and that this length is the same
            try:
                if num_bands != len(freq_high):
                    raise ValueError(
                        "Number of bands specified to freq_low {} "
                        "doesn't match number of bands specified "
                        "to freq_high {}".format(num_bands, len(freq_high))
                    )
            except TypeError:
                raise ValueError(
                    "If specifying multiple frequencies for "
                    "freq_low, must do the same for freq_high"
                )

        # sort the bands so we can make sure that
        # they don't overlap at all
        freq_high = [freq_high[freq_low.index(f)] for f in sorted(freq_low)]
        freq_low = sorted(freq_low)
        for i in range(num_bands):
            # first make sure that the low freq is
            # lower than the high freq
            if freq_low[i] >= freq_high[i]:
                raise ValueError(
                    "Low frequency {} not less than high frequency {}".format(
                        freq_low[i], freq_high
                    )
                )

            # iterate through all the higher bands and
            # make sure that the current high freq is
            # not higher than the higher low freqs
            if i < (num_bands - 1):
                for j in range(i + 1, num_bands):
                    if freq_high[i] > freq_low[j]:
                        raise ValueError(
                            "Top of frequency band {} greater than "
                            "bottom of higher frequency band {}".format(
                                freq_high[i], freq_low[j]
                            )
                        )

        # create a different filter for each band
        self.sos = []
        for low, high in zip(freq_low, freq_high):
            self.sos.append(signal.butter(
                order,
                [low, high],
                btype="bandpass",
                output="sos",
                fs=sample_rate
            ))

    def __call__(self, x):
        output = 0.
        for sos in self.sos:
            output += signal.sosfiltfilt(sos, y, axis=-1)
        return output
