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
If you've followed the steps outlined in the root [README](../../../README.md) for installing the DeepClean command line utility, you can install this project simply via

```console
deepclean build .
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
To get a sense for what command line arguments are available, you can run (from the `deepclean-infer` conda environment: `conda activate deepclean-infer`):

```console
infer -h
```
