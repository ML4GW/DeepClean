import logging
import time
from threading import Event, Thread
from queue import Empty, Queue

import numpy as np
import torch


class ChunkedTimeSeriesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        kernel_length: float,
        kernel_stride: float,
        sample_rate: float,
        batch_size: int,
        chunk_length: float,
        num_chunks: int,
        shuffle: bool = True,
        device: torch.device = "cpu"
    ) -> None:
        assert X.shape[-1] == y.shape[-1]

        self.kernel_size = int(kernel_length * sample_rate)
        self.stride_size = int(kernel_stride * sample_rate)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_length = chunk_length
        self.num_chunks = num_chunks

        X = torch.Tensor(X).to(device)
        y = torch.Tensor(y).to(device)
        if chunk_length == -1:
            remainder = (X.shape[-1] - self.kernel_size) % self.stride_size
            if remainder > 0:
                X = X[:, :-remainder]
                y = y[:-remainder]
            self.X = self.unfold(X)
            self.y = self.unfold(y)

        elif chunk_length > 0:
            if chunk_length < 1:
                chunk_length = chunk_length * X.shape[-1] / sample_rate

            split_idx, remainder = self.get_split_idx(
                X.shape[-1], int(chunk_length * sample_rate)
            )
            if remainder > 0:
                X = X[:, :-remainder]
                y = y[:-remainder]

            self.X = torch.split(X, split_idx, dim=-1)
            self.y = torch.split(y, split_idx, dim=-1)
        else:
            self.X = X
            self.y = y

    def get_split_idx(
        self, num_samples: int, samples_per_chunk: int
    ) -> Tuple[Tuple[int, ...], int]:
        # first figure out how to make chunks with
        # roughly the desired length that have
        # an even number of kernels in each chunk
        remainder_per_chunk = (
            samples_per_chunk - self.kernel_size
        ) % self.stride_size
        samples_per_chunk = samples_per_chunk - remainder_per_chunk
        num_chunks = (num_samples - 1) // samples_per_chunk + 1
        lengths = (samples_per_chunk,) * (num_chunks - 1)

        # make sure that the last chunk will have an even
        # number of kernels, even if that means sloughing
        # off a little bit of data from the edge
        last_length = num_samples - samples_per_chunk * (num_chunks - 1)
        remainder = (last_length - self.kernel_size) % self.stride_size
        last_length -= remainder

        # however, if we wont' have enough data for a
        # full kernel in this case, then just cut off
        # all the data in the last chunk
        # TODO: should we try to reallocate some of
        # this data to the existing chunks, or do we
        # assume that the user has tuned their
        # chunk_length to precisely match their
        # memory capacity?
        if last_length >= self.kernel_size:
            lengths = lengths + (last_length,)
        else:
            remainder = last_length + remainder
        return lengths, remainder

    def get_num_kernels(self, num_samples: int) -> int:
        """
        Don't need any ceil in this function since we
        take care beforehand to make sure all tensors
        have an even number of kernels in them
        """

        return (num_samples - self.kernel_size) // self.stride_size + 1

    def unfold(self, chunk: torch.Tensor) -> torch.Tensor:
        if len(chunk.shape) == 1:
            chunk = chunk[None, None, None]
            num_channels = None
        elif len(chunk.shape) == 2:
            chunk = chunk[None, :, None]
            num_channels = chunk.shape[1]
        else:
            raise ValueError(
                "Can't unfold timeseries chunk with shape {}".format(
                    chunk.shape
                )
            )

        num_kernels = self.get_num_kernels(chunk.shape[-1])
        unfold = torch.nn.Unfold(
            kernel_size=(1, num_kernels), dilation=(1, self.stride_size)
        )
        chunk = unfold(chunk)
        if num_channels is not None:
            chunk = chunk.reshape(num_channels, num_kernels, -1)
            chunk = chunk.transpose(1, 0)
        else:
            chunk = chunk[0]
        return chunk

    def grab_next_chunks(self, remainder_X=None, remainder_y=None):
        start = self._chunk_idx * self.num_chunks
        if start >= len(self.X):
            raise StopIteration

        stop = (self._chunk_idx + 1) * self.num_chunks
        chunk_indices = self.chunk_indices[start:stop]

        xs = [self.unfold(self.X[i]) for i in chunk_indices]
        ys = [self.unfold(self.y[i]) for i in chunk_indices]

        if remainder_X is not None:
            xs.insert(0, remainder_X)
            ys.insert(0, remainder_y)

        X = torch.cat(xs, axis=0)
        y = torch.cat(ys, axis=0)

        logging.debug(
            "Loaded chunks of shape {} and {}".format(X.shape, y.shape)
        )

        if self.shuffle:
            idx = torch.randperm(X.shape[0])
            X = X[idx]
            y = y[idx]
        self._chunk_idx += 1

        return X, y

    def chunk_getter(self):
        try:
            while not self._stop_event.is_set():
                try:
                    X, y = self.grab_next_chunks()
                except StopIteration:
                    break

                self._chunk_q.put((X, y))
                while self._chunk_q.unfinished_tasks:
                    time.sleep(1e-5)
            else:
                return
            self._stop_event.set()
        except Exception as e:
            self._stop_event.set()
            self._chunk_q.put(e)

    def __iter__(self):
        if self.chunk_length <= 0:
            if self.chunk_length == -1:
                num_kernels = self.X.shape[0]
            else:
                num_kernels = self.get_num_kernels(self.X.shape[-1])

            if self.shuffle:
                self.idx = torch.randperm(num_kernels)
            else:
                self.idx = torch.arange(num_kernels)
            self._batch_idx = 0
            return self

        if self.shuffle:
            self.chunk_indices = torch.randperm(len(self.X))
        else:
            self.chunk_indices = torch.arange(len(self.X))

        self._stop_event = Event()
        self._chunk_q = Queue()
        self._chunk_thread = Thread(target=self.chunk_getter)
        self._chunk_idx = self._batch_idx = 0
        self._chunk_thread.start()
        self.chunk_X = self.chunk_y = None
        return self

    def __next__(self):
        if self.chunk_length <= 0:
            start = self._batch_idx * self.batch_size
            if start >= self.idx.shape[0]:
                raise StopIteration

            stop = (self._batch_idx + 1) * self.batch_size
            idx = self.idx[start: stop]
            self._batch_idx += 1
            if self.chunk_length == -1:
                return self.X[idx], self.y[idx]
            else:
                idx = idx * self.stride_size
                X = torch.stack([
                    self.X[:, i: i + self.kernel_size] for i in idx
                ])
                y = torch.stack([
                    self.y[i: i + self.kernel_size] for i in idx
                ])
                return X, y

        start = self._batch_idx * self.batch_size
        if self.chunk_X is None or start >= len(self.chunk_X):
            while True:
                try:
                    chunk = self._chunk_q.get_nowait()
                except Empty:
                    if self._stop_event.is_set():
                        self._chunk_thread.join()
                        raise StopIteration
                    time.sleep(1e-6)
                else:
                    if isinstance(chunk, Exception):
                        raise chunk
                    self.chunk_X, self.chunk_y = chunk
                    start = self._batch_idx = 0
                    self._chunk_q.task_done()
                    break

        stop = (self._batch_idx + 1) * self.batch_size
        X = self.chunk_X[start: stop]
        y = self.chunk_y[start: stop]

        # if len(X) < self.batch_size:
        #     try:
        #         self.chunk_X, self.chunk_y = self.grab_next_chunks(X, y)
        #         X = self.chunk_X[:self.batch_size]
        #         y = self.chunk_y[:self.batch_size]
        #     except StopIteration:
        #         self.chunk_X, self.chunk_y = None, None

        self._batch_idx += 1
        return X, y


def ChunkedFrameFileDataset(ChunkedTimeSeriesDataset):
    def __init__(
        self,
        fnames: Union[str, List[str]],
        channels: List[str],
        kernel_length: float,
        kernel_stride: float,
        sample_rate: float,
        batch_size: int,
        chunk_length: float,
        num_chunks: int,
        shuffle: bool = True,
        device: torch.device = "cpu"
    ) -> None:
        if isinstance(fnames, str):
            fnames = [fnames]

        X, y = [], []
        for fname in fnames:
            if fname.split(".")[-1] == "gwf":
                data = self.read_gwf(fname, channels, sample_rate)
            elif fname.split(".")[-1] in ["h5", "hdf5"]:
                data = self.read_hdf5(fname, channels, sample_rate)
            else:
                raise ValueError(f"No parser to read file {fname}.")

            y.append(data[channels[0]])
            X.append(np.stack([data[chan] for chan in sorted(channels[1:])]))

        X = np.concatenate(X, axis=-1)
        y = np.concatenate(y, axis=-1)
        super().__init__(
            X,
            y,
            kernel_length=kernel_length,
            kernel_stride=kernel_stride,
            sample_rate=sample_rate,
            batch_size=batch_size,
            chunk_length=chunk_length,
            num_chunks=num_chunks,
            shuffle=shuffle,
            device=device
        )

    def read_hdf5(self, fname, channels, sample_rate):
        with h5py.File(self.fname, "r") as f:
            data = {}
            for chan, x in f.items():
                try:
                    if x.attrs["sample_rate"] != sample_rate:
                        raise ValueError(
                            "Channel {} has sample rate {}Hz, expected "
                            "sample rate of {}Hz".format(
                                chan, x.attrs["sample_rate"], sample_rate
                            )
                        )
                except KeyError:
                    continue

            if len(data) < len(channels):
                raise ValueError(
                    "Filename {} missing channels {}".format(
                        fname, list(set(channels) - set(data))
                    )
                )
        return data
