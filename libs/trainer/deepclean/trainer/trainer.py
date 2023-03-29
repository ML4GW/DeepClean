from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from deepclean.logging import logger
from deepclean.export.model import DeepClean
from deepclean.trainer import CompositePSDLoss
from deepclean.trainer.analysis import analyze_model
from deepclean.trainer.utils import Checkpointer, Trainer
from deepclean.utils.filtering import BandpassFilter, Frequency
from ml4gw.dataloading import InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity

torch.set_default_tensor_type(torch.FloatTensor)


def get_weights(state_dict: Dict, prefix: str):
    prefix = f"{prefix}."
    weights = {}
    for name, param in state_dict.items():
        if name.startswith(prefix):
            name = name.replace(prefix, "")
            weights[name] = param
    return weights


def load_weights(
    init_weights: Path,
    model: torch.nn.Module,
    preprocessor: ChannelWiseScaler
):
    logger.info(f"Initializing model from checkpoint '{init_weights}'")
    state_dict = torch.load(init_weights)
    weights = get_weights(state_dict, "deepclean")
    model.load_state_dict(weights)

    # if any of the witness channels have been
    # dropped or gone constant since the last
    # training, keep the old preprocessing params
    preproc_params = get_weights(state_dict, "preprocessor")
    mask = preprocessor.std == 0
    for param_name in ["mean", "std"]:
        param = getattr(preprocessor, param_name)
        param[mask] = preproc_params[param_name][mask]


def train(
    architecture: Callable,
    output_directory: Path,
    # data params
    X: np.ndarray,
    y: np.ndarray,
    kernel_length: float,
    kernel_stride: float,
    sample_rate: float,
    valid_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    # preproc params
    freq_low: Frequency = 55.0,
    freq_high: Frequency = 65.0,
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
    use_amp: bool = False,
) -> None:
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
        use_amp:
            Whether to train with mixed precision, which could
            offer speed advantages.
    """

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha value must be between 0 and 1")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = device or "cpu"

    # scale both the inputs and outputs to 0 mean unit variance
    input_scaler = ChannelWiseScaler(len(X)).to(device)
    output_scaler = ChannelWiseScaler().to(device)

    # fit the means and standard deviations _before_ filtering
    input_scaler.fit(X)
    output_scaler.fit(y)

    # Create the model, potentially loading in weights from
    # a file. Do this _before_ performing preprocessing
    # so that we can potentially resolve any bad preprocessing
    # parameters fit from the current witness data
    logger.info("Building and initializing model")
    model = architecture(len(X)).to(device)
    logger.info(model)

    if init_weights is not None:
        # allow us to easily point to the best weights
        # from another run of this same function
        if init_weights.is_dir():
            init_weights = init_weights / "weights.pt"
        load_weights(init_weights, model, input_scaler)
    # get rid of any remaining 0s in the preprocessing module
    input_scaler.std[input_scaler.std == 0] = 1
    print(input_scaler.std)

    # bandpass the strain up front so that we only
    # deal with the frequency ranges we care about
    strain_filter = BandpassFilter(
        freq_low, freq_high, sample_rate, filter_order
    )
    y = strain_filter(y)

    train_data = InMemoryDataset(
        X,
        y=y,
        kernel_size=int(sample_rate * kernel_length),
        batch_size=batch_size,
        stride=int(sample_rate * kernel_stride),
        coincident=True,
        shuffle=True,
        device=device,
    )
    train_data.X = input_scaler(train_data.X)
    train_data.y = output_scaler(train_data.y)
    if valid_data is not None:
        valid_X, valid_y = valid_data
        valid_y = strain_filter(valid_y)

        valid_data = InMemoryDataset(
            valid_X,
            y=valid_y,
            kernel_size=int(sample_rate * kernel_length),
            batch_size=batch_size,
            stride=int(sample_rate * kernel_stride),
            coincident=True,
            shuffle=False,
            device=device,
        )
        valid_data.X = input_scaler(valid_data.X)
        valid_data.y = output_scaler(valid_data.y)

    # now set up the optimization function
    logger.info("Initializing loss and optimizer")
    criterion = CompositePSDLoss(
        alpha,
        sample_rate,
        fftlength=fftlength,
        overlap=overlap,
        asd=True,
        freq_low=freq_low,
        freq_high=freq_high,
    )
    if alpha > 0:
        criterion.psd_loss.to(device)
        criterion.psd_loss.welch.to(device)

    trainer = Trainer(model, criterion, lr, weight_decay, use_amp)
    checkpointer = Checkpointer(
        output_directory,
        trainer.optimizer,
        patience=patience,
        decay_factor=factor,
        min_lr=lr * factor**2,
        early_stop=early_stop,
        checkpoint_every=5,
    )

    torch.backends.cudnn.benchmark = True
    logger.info("Beginning training loop")
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

        logger.info(f"=== Epoch {epoch + 1}/{max_epochs} ===")
        train_loss, valid_loss = trainer(train_data, valid_data, profiler)
        stop = checkpointer(train_loss, valid_loss, model)
        if stop:
            break

    # load in the best version of the model from training
    weights_path = output_directory / "weights.pt"
    model.load_state_dict(torch.load(weights_path))

    # generate some analyses of our model
    logger.info("Performing post-hoc analysis on trained model")
    welch = SpectralDensity(
        sample_rate, fftlength, average="median", fast=True
    )

    # reset batch sizes to be more manageable since
    # gradient computation in analysis is expensive
    valid_data.batch_size = train_data.batch_size = 64
    analyze_model(output_directory, model, welch, train_data, valid_data)

    model = DeepClean(input_scaler, model, output_scaler)
    torch.save(model.state_dict(), weights_path)
    return weights_path
