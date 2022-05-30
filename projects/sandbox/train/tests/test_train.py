import numpy as np
import train


def test_make_fake_sines():
    frequencies = [18.37, 22.14]
    channels = []
    for freq in frequencies:
        freq = str(freq).replace(".", "POINT")
        channel = "FAKE_SINE_FREQ_" + freq + "HZ"
        channels.append(channel)

    t0 = 10
    duration = 100
    sample_rate = 256
    result = train.make_fake_sines(channels, t0, duration, sample_rate)
    assert len(result) == 2

    time = np.arange(t0, t0 + duration, 1 / sample_rate)
    for freq, channel in zip(frequencies, channels):
        y = result[channel]
        assert len(y) == (duration * sample_rate)

        expected = np.sin(2 * np.pi * freq * time)
        assert (expected == y).all()


def test_fetch():
    return


def test_read():
    return


def test_write():
    return


def test_main():
    return
