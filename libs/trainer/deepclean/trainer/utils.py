import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import h5py
import torch

from deepclean.logging import logger

DataGenerator = Iterable[Tuple[torch.nn.Module, torch.nn.Module]]


@dataclass
class EpochTracker:
    profiler: Optional[torch.profiler.profile] = None

    def __post_init__(self):
        self.samples_seen = 0
        self.loss = 0
        self.start_time = time.time()
        if self.profiler is not None:
            self.profiler.start()

    def update(self, loss, num_samples) -> None:
        if self.profiler is not None:
            self.profiler.step()

        if loss is not None:
            self.loss += loss * num_samples
            self.samples_seen += num_samples

    def stop(self) -> Tuple[float, float, float]:
        if self.profiler is not None:
            self.profiler.stop()

        end_time = time.time()
        duration = end_time - self.start_time
        throughput = self.samples_seen / duration
        loss = self.loss / self.samples_seen
        return loss, duration, throughput


def backward(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Optional[float]:
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    isnan = torch.isnan(loss).item()
    if isnan and scaler is None:
        raise ValueError("Encountered nan loss")
    elif isnan:
        return None
    return loss


def backwardize(forward_fn: Callable) -> Callable:
    @wraps(forward_fn)
    def step(obj, X: torch.Tensor, y: torch.Tensor) -> Optional[float]:
        obj.optimizer.zero_grad(set_to_none=True)  # reset gradient

        # do forward step in mixed precision
        # if a gradient scaler got passed
        with torch.autocast("cuda", enabled=obj.scaler is not None):
            loss = forward_fn(obj, X, y)
        return backward(loss, obj.optimizer, obj.scaler)

    return step


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: Callable,
        learning_rate: float,
        weight_decay: float,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def forward(self, X, y):
        noise_prediction = self.model(X)
        loss = self.criterion(noise_prediction, y)
        return loss

    @backwardize
    def step(self, X, y):
        return self.forward(X, y)

    def __call__(
        self,
        train_data: DataGenerator,
        valid_data: Optional[DataGenerator] = None,
        profiler: Optional[torch.profiler.profile] = None,
    ) -> Tuple[float, Optional[float]]:
        self.model.train()
        tracker = EpochTracker(profiler)
        for witnesses, strain in train_data:
            loss = self.step(witnesses, strain).item()
            tracker.update(loss, len(witnesses))
        train_loss, duration, throughput = tracker.stop()

        logger.info(
            f"Duration {duration:0.2f}s, "
            f"Throughput {throughput:0.1f} samples/s"
        )
        msg = f"Train Loss: {train_loss:.4e}"

        # Evaluate performance on validation set if given
        if valid_data is not None:
            tracker = EpochTracker()
            self.model.eval()
            with torch.no_grad():
                for witnesses, strain in valid_data:
                    loss = self.forward(witnesses, strain).item()
                    tracker.update(loss, len(witnesses))
                valid_loss, _, __ = tracker.stop()
            msg += f", Valid Loss: {valid_loss:.4e}"
        else:
            valid_loss = None
        logger.info(msg)

        return train_loss, valid_loss


@dataclass
class Checkpointer:
    output_directory: Path
    optimizer: torch.optim.Optimizer
    patience: Optional[int] = None
    decay_factor: float = 0.1
    min_lr: Optional[float] = None
    early_stop: Optional[int] = None
    checkpoint_every: Optional[int] = None

    def __post_init__(self):
        if self.patience is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=self.patience,
                factor=self.decay_factor,
                threshold=0.0001,
                min_lr=self.min_lr,
                verbose=True,
            )
        self._i = 0
        self.best = float("inf")
        self.since_last = 0
        self.history = {"train_loss": [], "valid_loss": []}

        if self.checkpoint_every is not None:
            self.checkpoint_dir = self.output_directory / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def __call__(self, train_loss, valid_loss, model) -> bool:
        self._i += 1
        if (
            self.checkpoint_every is not None
            and not self._i % self.checkpoint_every
        ):
            fname = "epoch_" + str(self._i).zfill(4)
            fname = self.checkpoint_dir / fname
            torch.save(model.state_dict, fname)

        self.history["train_loss"].append(train_loss)
        stop = False
        if valid_loss is not None:
            self.history["valid_loss"].append(valid_loss)

            if self.patience is not None:
                self.lr_scheduler.step(valid_loss)

            if valid_loss <= self.best:
                self.best = valid_loss
                self.since_last = 0

                fname = self.output_directory / "weights.pt"
                torch.save(model.state_dict(), fname)
            elif self.early_stop is not None:
                self.since_last += 1
                if self.since_last >= self.early_stop:
                    stop = True

        with h5py.File(self.output_directory / "history.h5", "w") as f:
            group = f.create_group("loss")
            for key, value in self.history.items():
                group.create_dataset(key, data=value)
        return stop
