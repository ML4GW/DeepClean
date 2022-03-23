# DeepClean training
Trains DeepClean using either local hdf5 files or by fetching data from LIGO's Network Data Service (NDS) archive.

Almost all of the actual training is handled by the [`deepclean.trainer`](../../libs/trainer) library, so consult the code there to get a sense for what the actual training loop looks like. The code here is primarily interested in the logic for getting the specified data and preparing it into numpy arrays, as well in setting up logging.

## Installation
Assuming you chose [installation path #1](../../../README.md#1-the-easy-way---pinto) when setting this repo up and have the `pinto` command line utility available, you can install this project simply via

```console
pinto build .
```

from this directory.
Otherwise, you can install this project by first creating then cloning the base deepclean environment. From the root directory of this repo, this looks like

```console
conda env create -f environment.yaml
conda env create -n deepclean-train --clone deepclean-base
```

Then you can install the necessary additional libraries by running in this directory

```console
conda activate deepclean-train
poetry install
```

## Available commands
### `train`
To get a sense for what command line arguments are available, you can either run:

```console
pinto run . train -h
```

if you have the `pinto` command line utility installed, or you can `conda activate deepclean-train` and just run `train -h`.
Either way, you should see something like

```console
usage: main [-h] --channels CHANNELS [CHANNELS ...] --output-directory
            OUTPUT_DIRECTORY --sample-rate SAMPLE_RATE --kernel-length
            KERNEL_LENGTH --kernel-stride KERNEL_STRIDE
            [--chunk-length CHUNK_LENGTH] [--num-chunks NUM_CHUNKS]
            [--filt-fl FILT_FL [FILT_FL ...]]
            [--filt-fh FILT_FH [FILT_FH ...]] [--filt-order FILT_ORDER]
            [--batch-size BATCH_SIZE] [--max-epochs MAX_EPOCHS]
            [--init-weights INIT_WEIGHTS] [--lr LR]
            [--weight-decay WEIGHT_DECAY] [--patience PATIENCE]
            [--factor FACTOR] [--early-stop EARLY_STOP]
            [--fftlength FFTLENGTH] [--overlap OVERLAP] [--alpha ALPHA]
            [--device DEVICE] [--profile] [--data-directory DATA_DIRECTORY]
            [--t0 T0] [--duration DURATION] [--valid-frac VALID_FRAC]
            [--force-download] [--verbose]
            {autoencoder} ...

Train DeepClean on archival NDS2 data

positional arguments:
  {autoencoder}

optional arguments:
  -h, --help            show this help message and exit
  --channels CHANNELS [CHANNELS ...]
                        Either a list of channels to use during training, or a
                        path to a file containing these channels separated by
                        newlines. In either case, it is assumed that the 0th
                        channel corresponds to the strain data. (default:
                        None)
  --output-directory OUTPUT_DIRECTORY
                        Location to which to save training logs and outputs
                        (default: None)
  --sample-rate SAMPLE_RATE
                        Rate at which to resample witness and strain
                        timeseries (default: None)
  --kernel-length KERNEL_LENGTH
                        Lenght of the input to DeepClean in seconds (default:
                        None)
  --kernel-stride KERNEL_STRIDE
                        Distance between subsequent samples of kernels from
                        `X` and `y` in seconds. The total number of samples
                        used for training will then be `(X.shape[-1] /
                        sample_rate - kernel_length) // kernel_stride + 1`
                        (default: None)
  --chunk-length CHUNK_LENGTH
                        Dataloading parameter which dictates how many kernels
                        to unroll at once in GPU memory. Default value of `0`
                        means that kernels will be sampled at batch-generation
                        time instead of pre-computed, which will be slower but
                        have a lower memory footprint. Higher values will
                        break `X` up into `chunk_length` chunks which will be
                        unrolled into kernels in-bulk before iterating through
                        them. (default: 0)
  --num-chunks NUM_CHUNKS
                        Dataloading parameter which dictates how many chunks
                        to unroll at once. Ignored if `chunk_length` is 0. If
                        `chunk_length > 0`, indicates that `num_chunks *
                        chunk_length` data will be unrolled into kernels in-
                        bulk. Higher values can have a larger memory footprint
                        but better randomness and higher throughput. (default:
                        1)
  --filt-fl FILT_FL [FILT_FL ...]
                        Lower limit(s) of frequency range(s) over which to
                        optimize PSD loss, in Hz. Specify multiple to optimize
                        over multiple ranges. In this case, must be same
                        length as `filt_fh`. (default: 55.0)
  --filt-fh FILT_FH [FILT_FH ...]
                        Upper limit(s) of frequency range(s) over which to
                        optimize PSD loss, in Hz. Specify multiple to optimize
                        over multiple ranges. In this case, must be same
                        length as `filt_fl`. (default: 65.0)
  --filt-order FILT_ORDER
                        Order of bandpass filter to apply to strain channel
                        before training. (default: 8)
  --batch-size BATCH_SIZE
                        Sizes of batches to use during training. Validation
                        batches will be four times this large. (default: 32)
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs over which to train.
                        (default: 40)
  --init-weights INIT_WEIGHTS
                        Path to weights with which to initialize network. If
                        left as `None`, network will be randomly initialized.
                        If `init_weights` is a directory, it will be assumed
                        that this directory contains a file called
                        `weights.pt`. (default: None)
  --lr LR               Learning rate to use during training. (default: 0.001)
  --weight-decay WEIGHT_DECAY
                        Amount of regularization to apply during training.
                        (default: 0.0)
  --patience PATIENCE   Number of epochs without improvement in validation
                        loss before learning rate is reduced. If left as
                        `None`, learning rate won't be scheduled. Ignored if
                        `valid_data is None` (default: None)
  --factor FACTOR       Factor by which to reduce the learning rate after
                        `patience` epochs without improvement in validation
                        loss. Ignored if `valid_data is None` or `patience is
                        None`. (default: 0.1)
  --early-stop EARLY_STOP
                        Number of epochs without improvement in validation
                        loss before training terminates altogether. Ignored if
                        `valid_data is None`. (default: 20)
  --fftlength FFTLENGTH
                        The length of FFT windows in seconds over which to
                        compute the PSD estimate for the PSD loss. (default:
                        2)
  --overlap OVERLAP     The overlap between FFT windows in seconds over which
                        to compute the PSD estimate for the PSD loss. If left
                        as `None`, the overlap will be `fftlength / 2`.
                        (default: None)
  --alpha ALPHA         The relative amount of PSD loss compared to MSE loss,
                        scaled from 0. to 1. The loss function is computed as
                        `alpha * psd_loss + (1 - alpha) * mse_loss`. (default:
                        1.0)
  --device DEVICE       Indicating which device (i.e. cpu or gpu) to run on.
                        Use `"cuda"` to use the default GPU available, or
                        `"cuda:{i}`"`, where `i` is a valid GPU index on your
                        machine, to specify a specific GPU (alternatively,
                        consider setting the environment variable
                        `CUDA_VISIBLE_DEVICES=${i}` and using just `"cuda"`
                        here). (default: None)
  --profile             Whether to generate a tensorboard profile of the
                        training step on the first epoch. This will make this
                        first epoch slower. (default: False)
  --data-directory DATA_DIRECTORY
                        Path to existing data stored as HDF5 files. These
                        files are assumed to have the names "training.h5" and
                        "validation.h5", and to contain the channel data at
                        the root level. If these files do not exist and `t0`
                        and `duration` are specified, data will be downloaded
                        and saved to these files. If left as `None`, data will
                        be downloaded but not saved. (default: None)
  --t0 T0               Initial GPS timestamp of the stretch of data on which
                        to train DeepClean. Only necessary if `data_directory`
                        isn't specified or the files indicated above don't
                        exist, or if `force_download == True`. (default: None)
  --duration DURATION   Length of the stretch of data on which to train _and
                        validate_ DeepClean in seconds. Only necessary if
                        `data_directory` isn't specified or the files
                        indicated above don't exist, or if `force_download ==
                        True`. If still specified anyway, the training data in
                        `{data_directory}/training.h5` will be truncated to
                        this length, discarding the _earliest_ data. (default:
                        None)
  --valid-frac VALID_FRAC
                        Fraction of training data to reserve for validation,
                        split chronologically. If not `None` and data needs to
                        be downloaded, the first `(1 - valid_frac) * duration`
                        seconds of data will be used for training, and the
                        last `valid_frac * duration` seconds will be used for
                        validation. If `None` and data needs to be downloaded,
                        none will be reserved for validation. (default: None)
  --force-download      Whether to re-download data even if `data_directory`
                        is not `None` and `"training.h5"` and
                        `"validation.h5"` exist there. (default: False)
  --verbose             Indicates whether to log at DEBUG or INFO level
                        (default: False)
```

To use the default values saved in the `tool.typeo` section of the project directory's [`pyproject.toml`](../pyproject.toml), run

```console
train --typeo ..:train:autoencoder
```

Where here `autoencoder` specifies the network architecture that we'd like to train with (it's the only one for now).
``
