import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from deepclean.export import PrePostDeepClean
from deepclean.signal import BandpassFilter, StandardScaler
from deepclean.signal.filter import FREQUENCY
from deepclean.trainer import ChunkedTimeSeriesDataset, CompositePSDLoss
from deepclean.trainer.viz import plot_data_asds

torch.set_default_tensor_type(torch.FloatTensor)


def train_for_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_data: ChunkedTimeSeriesDataset,
    valid_data: Optional[ChunkedTimeSeriesDataset] = None,
    profiler: Optional[torch.profiler.profile] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    """Run a single epoch of training"""

    train_loss = 0
    samples_seen = 0
    start_time = time.time()
    model.train()
    for witnesses, strain in train_data:
        optimizer.zero_grad(set_to_none=True)  # reset gradient
        # do forward step in mixed precision
        with torch.autocast("cuda"):
            noise_prediction = model(witnesses)
            loss = criterion(noise_prediction, strain)

        train_loss += loss.item() * len(witnesses)
        samples_seen += len(witnesses)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()
        logging.debug(f"{samples_seen}/{train_data.num_kernels}")

    if profiler is not None:
        profiler.stop()

    end_time = time.time()
    duration = end_time - start_time
    throughput = samples_seen / duration
    train_loss /= samples_seen

    logging.info(
        "Duration {:0.2f}s, Throughput {:0.1f} samples/s".format(
            duration, throughput
        )
    )
    msg = f"Train Loss: {train_loss:.4e}"

    # Evaluate performance on validation set if given
    if valid_data is not None:
        valid_loss = 0
        samples_seen = 0

        model.eval()
        with torch.no_grad():
            for witnesses, strain in valid_data:
                noise_prediction = model(witnesses)
                loss = criterion(noise_prediction, strain)

                valid_loss += loss.item() * len(witnesses)
                samples_seen += len(witnesses)

        valid_loss /= samples_seen
        msg += f", Valid Loss: {valid_loss:.4e}"
    else:
        valid_loss = None

    logging.info(msg)
    return train_loss, valid_loss, duration, throughput


def train(
    architecture: Callable,
    output_directory: str,
    # data params
    X: np.ndarray,
    y: np.ndarray,
    kernel_length: float,
    kernel_stride: float,
    sample_rate: float,
    chunk_length: float = 0,
    num_chunks: int = 1,
    valid_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    # preproc params
    # TODO: make optional
    freq_low: FREQUENCY = 55.0,
    freq_high: FREQUENCY = 65.0,
    filter_order: int = 8,
    # optimization params
    batch_size: int = 32,
    max_epochs: int = 40,
    init_weights: Optional[Path] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: Optional[int] = None,
    factor: float = 0.1,
    early_stop: int = 20,
    # criterion params
    fftlength: float = 2,
    overlap: Optional[float] = None,
    alpha: float = 1.0,
    # misc params
    device: Optional[str] = None,
    profile: bool = False,
) -> float:
    """Train a DeepClean model on in-memory data

    Args:
        architecture:
            A callable which takes as its only input the number
            of witness channels, and returns an initialized torch
            Module
        output_directory:
            Location to save training artifacts like optimized
            weights, preprocessing objects, and visualizations
        X:
            Array containing witness timeseries data
        y:
            Array contain strain timeseries data
        kernel_length:
            Lenght of the input to DeepClean in seconds
        kernel_stride:
            Distance between subsequent samples of kernels from
            `X` and `y` in seconds. The total number of samples
            used for training will then be
            `(X.shape[-1] / sample_rate - kernel_length) // kernel_stride + 1`
        sample_rate:
            The rate at which the timeseries contained in `X`
            and `y` were sampled
        chunk_length:
            Dataloading parameter which dictates how many kernels
            to unroll at once in GPU memory. Default value of `0`
            means that kernels will be sampled at batch-generation
            time instead of pre-computed, which will be slower but
            have a lower memory footprint. Higher values will break
            `X` up into `chunk_length` chunks which will be unrolled
            into kernels in-bulk before iterating through them.
        num_chunks:
            Dataloading parameter which dictates how many chunks
            to unroll at once. Ignored if `chunk_length` is 0. If
            `chunk_length > 0`, indicates that `num_chunks * chunk_length`
            data will be unrolled into kernels in-bulk. Higher values
            can have a larger memory footprint but better randomness
            and higher throughput.
        valid_data:
            Witness and strain timeseries for validating model
            performance at the end of each epoch. If left
            as `None`, validation won't be performed.
        freq_low:
            Lower limit(s) of frequency range(s) over which to optimize
            PSD loss, in Hz. Specify multiple to optimize over
            multiple ranges. In this case, must be same length
            as `freq_high`.
        freq_high:
            Upper limit(s) of frequency range(s) over which to optimize
            PSD loss, in Hz. Specify multiple to optimize over
            multiple ranges. In this case, must be same length
            as `freq_low`.
        filter_order:
            Order of bandpass filter to apply to strain channel
            before training.
        batch_size:
            Sizes of batches to use during training. Validation
            batches will be four times this large.
        max_epochs:
            Maximum number of epochs over which to train.
        init_weights:
            Path to weights with which to initialize network. If
            left as `None`, network will be randomly initialized.
            If `init_weights` is a directory, it will be assumed
            that this directory contains a file called `weights.pt`.
        lr:
            Learning rate to use during training.
        weight_decay:
            Amount of regularization to apply during training.
        patience:
            Number of epochs without improvement in validation
            loss before learning rate is reduced. If left as
            `None`, learning rate won't be scheduled. Ignored
            if `valid_data is None`
        factor:
            Factor by which to reduce the learning rate after
            `patience` epochs without improvement in validation
            loss. Ignored if `valid_data is None` or
            `patience is None`.
        early_stop:
            Number of epochs without improvement in validation
            loss before training terminates altogether. Ignored
            if `valid_data is None`.
        fftlength:
            The length of FFT windows in seconds over which to
            compute the PSD estimate for the PSD loss.
        overlap:
            The overlap between FFT windows in seconds over which
            to compute the PSD estimate for the PSD loss. If left
            as `None`, the overlap will be `fftlength / 2`.
        alpha:
            The relative amount of PSD loss compared to
            MSE loss, scaled from 0. to 1. The loss function
            is computed as `alpha * psd_loss + (1 - alpha) * mse_loss`.
        device:
            Indicating which device (i.e. cpu or gpu) to run on. Use
            `"cuda"` to use the default GPU available, or `"cuda:{i}`"`,
            where `i` is a valid GPU index on your machine, to specify
            a specific GPU (alternatively, consider setting the environment
            variable `CUDA_VISIBLE_DEVICES=${i}` and using just `"cuda"`
            here).
        profile:
            Whether to generate a tensorboard profile of the
            training step on the first epoch. This will make
            this first epoch slower.
    """

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha value must be between 0 and 1")
    os.makedirs(output_directory, exist_ok=True)
    output_directory = Path(output_directory)

    # TODO: is there more complicated device
    # logic we want to do or should "cpu" just
    # be the default?
    device = device or "cpu"

    # Create pipelines for preprocessing both
    # witness and strain data. Witness data
    # just gets normalized on a channel-wise
    # basis, strain data gets normalized then
    # bandpass filtered to ensure only the
    # target frequency range gets optimized
    logging.info("Preprocessing")
    witness_scaler = StandardScaler()
    strain_scaler = StandardScaler()
    bandpass = BandpassFilter(
        freq_low=freq_low,
        freq_high=freq_high,
        sample_rate=sample_rate,
        order=filter_order,
    )

    # fit the pipelines to the data (i.e. compute
    # the mean and standard deviation across
    # the datasets) and save them for later use
    witness_pipeline = witness_scaler
    witness_pipeline.fit(X)
    witness_pipeline.write(output_directory / "witness_pipeline.pkl")

    strain_pipeline = strain_scaler >> bandpass
    strain_pipeline.fit(y)
    strain_pipeline.write(output_directory / "strain_pipeline.pkl")

    # use these preprocessed arrays to
    # instantiate iterable datasets
    train_data = ChunkedTimeSeriesDataset(
        witness_pipeline(X),
        strain_pipeline(y),
        kernel_length=kernel_length,
        kernel_stride=kernel_stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        chunk_length=chunk_length,
        num_chunks=num_chunks,
        shuffle=False,
        device=device,
    )

    if valid_data is not None:
        valid_X, valid_y = valid_data
        valid_data = ChunkedTimeSeriesDataset(
            witness_pipeline(valid_X),
            strain_pipeline(valid_y),
            kernel_length=kernel_length,
            kernel_stride=kernel_stride,
            sample_rate=sample_rate,
            batch_size=batch_size * 2,
            chunk_length=-1,
            shuffle=False,
            device=device,
        )

    # Creating model, loss function, optimizer and lr scheduler
    logging.info("Building and initializing model")
    model = architecture(len(X))
    model.to(device)

    if init_weights is not None:
        # allow us to easily point to the best weights
        # from another run of this same function
        if init_weights.is_dir():
            init_weights = init_weights / "weights.pt"

        logging.debug(
            f"Initializing model weights from checkpoint '{init_weights}'"
        )
        model.load_state_dict(torch.load(init_weights))
    logging.info(model)

    logging.info("Initializing loss and optimizer")
    criterion = CompositePSDLoss(
        alpha,
        sample_rate,
        fftlength=fftlength,
        overlap=None,
        asd=False,
        device=device,
        freq_low=freq_low,
        freq_high=freq_high,
    )

    if alpha > 0:
        # if we have a welch transform to work with,
        # plot the ASDs of training data as they go
        # into the network so we have a sense for
        # what the NN is learning from
        if freq_low is not None:
            # zoom in on the frequency range or
            # ranges we actually care about
            try:
                x_range = (0.9 * min(freq_low), 1.1 * max(freq_high))
            except TypeError:
                x_range = (0.9 * freq_low, 1.1 * freq_high)

        plot_data_asds(
            train_data,
            sample_rate=sample_rate,
            write_dir=output_directory / "train_asds",
            welch=criterion.psd_loss.welch,
            channels=range(len(X) + 1),
            x_range=x_range,
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience,
            factor=factor,
            threshold=0.0001,
            min_lr=lr * factor**2,
            verbose=True,
        )

    # start training
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None
    best_valid_loss = float("inf")
    since_last_improvement = 0
    weights_path = output_directory / "weights.pt"
    history = {"train_loss": [], "valid_loss": []}

    logging.info("Beginning training loop")
    for epoch in range(max_epochs):
        if epoch == 0 and profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    output_directory / "profile"
                ),
            )
            profiler.start()
        else:
            profiler = None

        logging.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")
        train_loss, valid_loss, duration, throughput = train_for_one_epoch(
            model,
            optimizer,
            criterion,
            train_data,
            valid_data,
            profiler,
            scaler,
        )
        history["train_loss"].append(train_loss)

        # do some house cleaning with our
        # validation loss if we have one
        if valid_loss is not None:
            history["valid_loss"].append(valid_loss)

            # update our learning rate scheduler if we
            # indicated a schedule with `patience`
            if patience is not None:
                lr_scheduler.step(valid_loss)

            # save this version of the model weights if
            # we achieved a new best loss, otherwise check
            # to see if we need to early stop based on
            # plateauing validation loss
            if valid_loss < best_valid_loss:
                logging.debug(
                    "Achieved new lowest validation loss, "
                    "saving model weights"
                )
                best_valid_loss = valid_loss

                torch.save(model.state_dict(), weights_path)
                since_last_improvement = 0
            else:
                since_last_improvement += 1
                if since_last_improvement >= early_stop:
                    logging.info(
                        "No improvement in validation loss in {} "
                        "epochs, halting training early".format(early_stop)
                    )
                    break
        else:
            # if we don't have validation data, just
            # always keep the last version of the model
            torch.save(model.state_dict(), weights_path)

    # load in the best version of the model from training
    model.load_state_dict(torch.load(weights_path))
    if valid_data is not None and alpha > 0:
        # if we have validation data and a welch transform,
        # plot the same ASDs above for the validation data,
        # this time using our optimized model and postprocessing
        # pipeline to also plot the residuals against the
        # strain channel
        plot_data_asds(
            valid_data,
            sample_rate=sample_rate,
            write_dir=output_directory / "valid_asds",
            welch=criterion.psd_loss.welch,
            channels=range(len(X) + 1),
            x_range=x_range,
            model=model,
        )

    # now create a version of the model which
    # has the pre- and postprocessing built in
    nn = PrePostDeepClean(model)
    nn.fit(X, y)
    torch.save(nn.state_dict(), weights_path)
    return history
