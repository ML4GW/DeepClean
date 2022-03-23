# DeepClean "Online" Inference
Serve a DeepClean model for inference using Triton inference server and then use directories of one-second-long witness and strain frame files to send inference requests to that server, which are then postprocessed in an "online" fashion (more information below).

## Background
### Directory structures
Rather than running inference on a real data replay stream, this script aims to _recreate_ the behavior of such an inference pipeline in a less asynchronous, more easily debuggable fashion to ensure that online cleaning is producing the expected behavior.

As such, the input data is expected to be consist of two directories containing one-second-long gravitational wave frame (GWF) files. The files in one directory should contain the relevant witness channels, and those in the other should contain the relevant strain channel at the matching timestamps. The files should be named in such a way such that doing a lexigraphic sort on the filenames sets them in chronological order, and every file in each directory should have a counterpart in the other.

### Processing steps
Online processing is characterized by the lack of availability of future data, or at least future data beyond the scale of latency your use case is willing to incur while waiting. This necessarily creates some disparity with the offline cleaning scenario, and so we outline the online processing steps (and where they occur), from the loading of a frame to the writing of its cleaned counterpart, explicitly here for the benefit of clarity.

#### 1. Data loading (Client-side)
A new witness frame file and its corresponding strain frame file are read into memory using `gwpy.timeseries.TimeSeriesDict.read`

#### 2. Preprocessing (Client-side)
Both the witness frame and the strain frame are resampled to a fixed sample rate. The witness frame is normalized by the channel-wise mean and standard deviation from the training set.

#### 3. Streaming updates (Client → Server)
The witness frame is broken into `stride_length` segments which are then streamed to the server and update its snapshot state in sequence.


#### 4. DeepClean inference (Server-side)
Each updated snapshot is passed to DeepClean on the server to produce an estimate of the noise at the strain channel for that kernel of data.

#### 5. Online averaging (Server-side)
An aggregation model at the backend of the server takes DeepClean's prediction and aligns it in time with previous predictions to produce an online average of the prediction for each segment. Once a segment has been averaged by `max_latency // stride_length` overlapping predictions, it is streamed back to the client.

This means that there is an induced latency of `max_latency` (more precisely, `stride_length * (max_latency // stride_length)`) between the initial timestamp of the input update streamed _to_ the server and that of the output prediction streamed back _from_ it. Accordingly, we can discard the first `max_latency // stride_length` predictions because they technically correspond to predictions for segments from before the first frame file begins!

#### 6. Accumulation (Client-side)
These segments of online-averaged predictions are accumulated into a one-second-long noise prediction on the client side. Moreover, in order to avoid edge-effects during filtering, `filter_lead_time` seconds worth of data are required to be accumulated before postprocessing can begin.

This leads to two different sources of _fundamental latency_ that come from data modelling constraints, and have _nothing_ to do with inference latency.

- The first is latency incurred waiting for additional predictions on the same segment of data so that it can be averaged and streamed back from the server.
- The second is latency incurred waiting for additional _averaged predictions_ to come _back_ from the server (already delayed by `max_latency` seconds) to provide some padding so that the current frame-to-be-cleaned doesn't suffer edge effects during filtering.

#### 7. Post-processing (Client-side)
The last `filter_memory` seconds worth of noise predictions, the noise predictions for the current frame, and the noise predictions for the next `filter_lead_time` worth of data are concatenated into a single timeseries.

This timeseries is band-pass filtered using the same parameters used to filter the strain channel during training, then un-normalized using the mean and standard deviation of the strain channel from the training set.

The current frame is then sliced out of this padded timeseries and subtracted from the corresponding strain frame.

#### 8. Data writing and cleanup (Client-side)
The cleaned strain frame is written to disk, and the most out-of-date second of data is removed from the `filter_memory` timeseries.

Note that at this point, we've already accumulated `filter_lead_time` seconds of the _next_ frame we need to process.


## Installation
Assuming you chose [installation path #1](../../../README.md##1-the-easy-way---pinto) when setting this repo up and have the `pinto` command line utility available, you can install this project simply via

```console
pinto build .
```

from this directory. Otherwise, you can install this project by first creating then cloning the base deepclean environment. From the root directory of this repo, this looks like

```console
conda env create -f environment.yaml
conda env create -n deepclean-infer --clone deepclean-base
```

Then you can install the necessary additional libraries by running in this directory

```console
conda activate deepclean-infer
poetry install
```

## Available commands
### `infer`
To get a sense for what command line arguments are available, you can either run:

```console
pinto run . infer -h
```

if you have the `pinto` command line utility installed, or you can `conda activate deepclean-infer` and just run `infer -h`.
Either way, you should see something like

```console
usage: main [-h] --url URL --model-repo-dir MODEL_REPO_DIR --model-name MODEL_NAME --train-directory TRAIN_DIRECTORY
            --witness-data-dir WITNESS_DATA_DIR --strain-data-dir STRAIN_DATA_DIR --channels CHANNELS [CHANNELS ...]
            --kernel-length KERNEL_LENGTH --stride-length STRIDE_LENGTH --sample-rate SAMPLE_RATE --max-latency
            MAX_LATENCY --filter-memory FILTER_MEMORY --filter-lead-time FILTER_LEAD_TIME [--sequence-id SEQUENCE_ID]
            [--verbose] [--gpus GPUS [GPUS ...]] [--max-frames MAX_FRAMES]

    Serve up the models from the indicated model repository
    for inference using Triton and stream witness data taken
    from one second-long frame files to clean the corresponding
    strain data in an online fashion.

optional arguments:
  -h, --help            show this help message and exit
  --url URL             Address at which Triton service is being hosted and to which to send requests, including port
                        (default: None)
  --model-repo-dir MODEL_REPO_DIR
                        Directory containing models to serve for inference (default: None)
  --model-name MODEL_NAME
                        Model to which to send streaming inference requests (default: None)
  --train-directory TRAIN_DIRECTORY
                        Directory where pre- and post-processing pipelines were exported during training (default:
                        None)
  --witness-data-dir WITNESS_DATA_DIR
                        A directory containing one-second-long gravitational wave frame files corresponding to witness
                        data as inputs to DeepClean. Files should be named identically except for their end which
                        should take the form `<GPS timestamp>_<length of frame>.gwf`. (default: None)
  --strain-data-dir STRAIN_DATA_DIR
                        A directory containing one-second-long gravitational wave frame files corresponding to the
                        strain data to be cleaned. The same rules about naming conventions apply as those outlined for
                        the files in `witness_data_dir`, with the added stipulation that each timestamp should have a
                        matching file in `witness_data_dir`. (default: None)
  --channels CHANNELS [CHANNELS ...]
                        A list of channel names used by DeepClean, with the strain channel first, or the path to a
                        text file containing this list separated by newlines (default: None)
  --kernel-length KERNEL_LENGTH
                        The length, in seconds, of the input to DeepClean (default: None)
  --stride-length STRIDE_LENGTH
                        The length, in seconds, between kernels sampled at inference time. This, along with the
                        `sample_rate`, dictates the size of the update expected at the snapshotter model (default:
                        None)
  --sample-rate SAMPLE_RATE
                        Rate at which the input kernel has been sampled, in Hz (default: None)
  --max-latency MAX_LATENCY
                        The maximum amount of time, in seconds, allowed during inference to wait for overlapping
                        predictcions for online averaging. For example, if the `stride_length` is 0.002s and
                        `max_latency` is 0.5s, then output segments will be averaged over 250 overlapping kernels
                        before being streamed back from the server. This means there is a delay of `max_latency` (or
                        the greatest multiple of `stride_length` that is less than `max_latency`) seconds between the
                        start timestamp of the update streamed to the snapshotter and the resulting prediction
                        returned by the ensemble model. The online averaging model being served by Triton should have
                        been instantiated with this same value. (default: None)
  --filter-memory FILTER_MEMORY
                        The number of seconds of past data to use when filtering a frame's worth of noise predictions
                        before subtraction to avoid edge effects (default: None)
  --filter-lead-time FILTER_LEAD_TIME
                        The number of seconds of _future_ data required to be available before filtering a frame's
                        worth of noise predictions before subtraction to avoid edge effects (default: None)
  --sequence-id SEQUENCE_ID
                        A unique identifier to give this input/output snapshot state on the server to ensure streams
                        are updated appropriately (default: 1001)
  --verbose             If set, log at `DEBUG` verbosity, otherwise log at `INFO` verbosity. (default: False)
  --gpus GPUS [GPUS ...]
                        The indices of the GPUs to use for inference (default: None)
  --max-frames MAX_FRAMES
                        The maximum number of files from `witness_data_dir` and `strain_data_dir` to clean. (default:
                        None)
```
