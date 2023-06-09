from unittest.mock import patch

import numpy as np
import pytest

from deepclean.infer import callback
from deepclean.logging import logger


def add_one(_, x):
    return x + 1


@patch("deepclean.infer.callback.BandpassFilter.__call__", new=add_one)
class TestCleaner:
    def test_call(self):
        x = np.arange(128 * 8).astype("float32")
        cleaner = callback.Cleaner(
            kernel_length=1,
            sample_rate=128,
            filter_pad=0.5,
            freq_low=10,
            freq_high=50,
        )
        y = cleaner(x)
        assert y.shape == (128,)

        expected = np.arange(int(128 * 6.5), int(128 * 7.5)) + 1
        assert (y == expected).all()


@patch("deepclean.infer.callback.BandpassFilter.__call__", new=add_one)
class TestState:
    @pytest.fixture
    def state(self):
        logger.set_logger("test")
        return callback.State(
            name="test",
            frame_length=1,
            memory=8,
            filter_pad=0.5,
            sample_rate=128,
            inference_sampling_rate=16,
            batch_size=8,
            aggregation_steps=4,
            freq_low=10,
            freq_high=20,
        )

    def test_init(self, state):
        assert state.frame_size == 128
        assert state.stride == 8
        assert state.steps_per_frame == 16
        assert state.memory == 1024
        assert state.samples_ahead == 64
        assert state.steps_ahead == 8
        assert state.agg_batches == 0
        assert state.agg_leftover == 4

    def test_validate(self, state):
        x = np.arange(64).astype("float32")
        with pytest.raises(ValueError) as exc:
            state.validate(x[:-1], 0)
        assert str(exc.value).startswith("Noise prediction is of wrong length")

        response = state.validate(x, 0)
        assert response is not None
        assert response.shape == (32,)
        assert state._latest_seen == 0

        expected = np.arange(32, 64)
        assert (response == expected).all()

        response = state.validate(x, 1)
        assert response.shape == (64,)
        assert state._latest_seen == 1
        assert (response == x).all()

    def test_update(self, state):
        x = np.arange(64).astype("float32")
        response = state.update(x, 0)
        assert response is None
        assert state._state.shape == (128,)
        assert (state._state[:32] == x[-32:]).all()
        assert (state._state[32:] == 0).all()

        response = state.update(x + 64, 1)
        assert response is None
        assert state._state.shape == (128,)
        expected = np.arange(32, 128)
        assert (state._state[: 32 + 64] == expected).all()
        assert (state._state[32 + 64 :] == 0).all()

        response = state.update(x + 128, 2)
        assert response is None
        assert state._state.shape == (256,)
        expected = np.arange(32, 64 * 3)
        assert (state._state[: 32 + 64 * 2] == expected).all()
        assert (state._state[32 + 64 * 2 :] == 0).all()

        response = state.update(x + 64 * 3, 3)
        assert response is not None
        assert response.shape == (128,)
        assert (response[64:] == np.arange(64) + 32 + 64 + 1).all()
        assert state._frame_idx == 1
