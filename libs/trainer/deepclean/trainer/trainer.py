import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from deepclean.signal import BandpassFilter, StandardScaler
from deepclean.trainer import ChunkedTimeSeriesDataset, CompositePSDLoss


# Default tensor type
torch.set_default_tensor_type(torch.FloatTensor)


def run_train_step(
    model: torch.nn.Module,
    train_data: ChunkedTimeSeriesDataset,
    valid_data: Optional[ChunkedTimeSeriesDataset] = None,
    profiler: Optional[torch.profiler.Profiler] = None
):
    train_loss = 0
    samples_seen = 0
    model.train()
    for witnesses, strain in train_data:
        optimizer.zero_grad(set_to_none=True)  # reset gradient
        # do forward step in mixed precision
        with torch.autocast("cuda"):
            noise_prediction = model(witnesses)
            loss = criterion(noise_prediction, strain)

        # do backwards pass at full precision
        loss.backward()

        # update weights and add gradient step to
        # profile if we have it turned on
        optimizer.step()
        if profiler is not None:
            profiler.step()

        train_loss += loss.item() * len(witnesses)
        samples_seen += len(witnesses)

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
        msg += f", Valid Loss {valid_loss:.4e}"
    else:
        valid_loss = None

    logging.info(msg)
    return train_loss, valid_loss, duration, throughput


def train(
    architecture: torch.nn.module,
    output_directory: str,

    # data params
    X: np.ndarray,
    y: np.ndarray,
    kernel_length: float,
    kernel_stride: float,
    sample_rate: float,
    chunk_length: float = 0.05,
    num_chunks: int = 4,
    valid_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,

    # preproc params
    filt_fl: Union[float, List[float]] = 55.0,
    filt_fh: Union[float, List[float]] = 65.0,
    filt_order: int = 8,

    # optimization params
    batch_size: int = 32,
    max_epochs: int = 40,
    init_weights: Optional[str] = None,
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
    profile: bool = False
) -> float:
    """Train a DeepClean model on a stretch of data

    If no training data is availble locally, it will be
    downloaded from the NDS2 server, so ensure that you
    have appropriate credentials for this first.

    Args:
        chanslist:
            File containing the channels to use for training separated
            by newlines, with the target channel in the first row
        filt_fl:
            Lower limit of frequency range over which to optimize
            PSD loss, in Hz. Specify multiple to optimize over
            multiple ranges. In this case, must be same length
            as `filt_fh`.
        filt_fh:
            Upper limit of frequency range over which to optimize
            PSD loss, in Hz. Specify multiple to optimize over
            multiple ranges. In this case, must be same length
            as `filt_fl`.
        filt_order:
            Order of bandpass filter to apply to strain channel
            before training.
        train_kernel:
            Length of time dimension of input to network, in seconds.
            I.e. the last dimension of the network input will have
            size `int(fs * train_kernel)`.
        train_stride:
            Length in seconds between frames that can be sampled
            for training.
        pad_mode:
            Deprecated.
        batch_size:
            Sizes of batches to use during training. Validation
            batches will be four times this large.
        max_epochs:
            Maximum number of epochs over which to train.
        lr:
            Learning rate to use during training.
        weight_decay:
            Amount of regularization to apply during training
        fftlength:
            The size of the FFT to use to compute the PSD
            for the PSD loss
        alpha:
            The relative amount of PSD loss compared to
            MSE loss, scaled from 0. to 1. The loss function
            is computed as `alpha * psd_loss + (1 - alpha) * mse_loss`.
        log_file:
            Name of the file in `train_dir` to which to write logs
    """

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha value must be between 0 and 1")
    os.makedirs(output_directory, exists_ok=True)

    # Use GPU if available
    device = "cuda:5"  # dc.nn.utils.get_device(device)

    # Preprocess data
    logging.info("Preprocessing")
    witness_scaler = StandardScaler()
    strain_scaler = StandardScaler()
    bandpass = BandpassFilter(
        freq_low=filt_fl,
        freq_high=filt_fh,
        sample_rate=sample_rate,
        order=filt_order
    )
    witness_pipeline = witness_scaler >> bandpass
    witness_pipeline.fit(X)
    witness_pipeline.write(
        os.path.join(output_directory, "witness_pipeline.pkl")
    )

    strain_pipeline = strain_scaler >> bandpass
    strain_pipeline.fit(y)
    strain_pipeline.write(
        os.path.join(output_directory, "strain_pipeline.pkl")
    )

    X = witness_pipeline(X)
    y = strain_pipeline(y)
    if valid_data is not None:
        valid_X, valid_y = valid_data
        valid_X = witness_pipeline(valid_X)
        valid_y = strain_pipeline(valid_y)

    train_data = ChunkedTimeSeriesDataset(
        X,
        y,
        kernel_length=kernel_length,
        kernel_stride=kernel_stride,
        sample_rate=sample_rate,
        batch_size=batch_size,
        chunk_length=chunk_length,
        num_chunks=num_chunks,
        shuffle=True,
        device=device
    )

    if valid_data is not None:
        valid_data = ChunkedTimeSeriesDataset(
            valid_X,
            valid_y,
            kernel_length=kernel_length,
            kernel_stride=kernel_stride,
            sample_rate=sample_rate,
            batch_size=batch_size * 4,
            chunk_length=-1,
            shuffle=False,
            device=device
        )

    # Creating model, loss function, optimizer and lr scheduler
    logging.info("Building and initializing model")
    model = architecture(len(X))
    model.to(device)

    if init_weights is not None:
        # allow us to easily point to the best weights
        # from another run of this same function
        if os.path.isdir(init_weights):
            init_weights = os.path.join(init_weights, "weights.pt")

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
        overlap=overlap,
        asd=True,
        device=device,
        freq_low=filt_fl,
        freq_high=filt_fh,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    if patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience,
            factor=factor,
            min_lr=lr / factor ** 2
        )

    # start training
    torch.backends.cudnn.benchmark = True
    best_valid_loss = np.inf
    since_last_improvement = 0
    history = {"train_loss": [], "valid_loss": []}

    logging.info("Beginning training loop")
    for epoch in range(max_epochs):
        if epoch == 0 and profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(output_directory, "profile")
                ),
            )
            profiler.start()
        else:
            profiler = None

        logging.info(f"### Epoch {epoch + 1}/{max_epochs} ###")
        train_loss, valid_loss, duration, throughput = run_train_step(
            model, train_data, valid_data, profiler
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

                weights_path = os.path.join(logger.data_subdir, "weights.pt")
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

    return history