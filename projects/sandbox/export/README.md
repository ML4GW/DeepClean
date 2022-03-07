# DeepClean Export
Export a trained DeepClean network architecture to a Triton model repository (or "model store"). This will export the trained network as an ONNX binary to the repo and create a model configuration file for it, with options for setting the level of concurrency through that configuration file.

Additionally, this will create snapshotter and aggregation models to insert at either end of DeepClean to handle input state caching and output online averaging for streaming use cases.

The online averaging model works by taking overlapping segments of DeepClean noise predictions and averaging them over a fixed number of updates. The number of updates, rather than being set explicitly, is controlled by the `max_latency` argument of the `export-model` command. This argument works with the `stride_length` argument to indicate the max number of "time", as indicated by the number of `stride_length` streaming updates, over which to average predictions. The "latency" induced is the latency between the initial timestamp of each _input_ stream and the initial timestamp of the corresponding _output_ stream that gets returned, which will be `stride_length * num_updates` seconds behind the former, where `num_updates = max_latency // stride_length`.

## Installation
If you've followed the steps outlined in the root [README](../../../README.md) for installing the DeepClean command line utility, you can install this project simply via

```console
deepclean build .
```

from this directory. Otherwise,

```console
poetry install
```

will work as well.


## Running
You can see a full list of the available command line options by running

```console
poetry run export-model -h
```

Which should return something like

```console
usage: export [-h] --repository-directory REPOSITORY_DIRECTORY --channels CHANNELS [CHANNELS ...] --weights WEIGHTS
              --kernel-length KERNEL_LENGTH --stride-length STRIDE_LENGTH --sample-rate SAMPLE_RATE --max-latency
              MAX_LATENCY [MAX_LATENCY ...] [--streams-per-gpu STREAMS_PER_GPU] [--instances INSTANCES]
              [--platform {onnxruntime_onnx,tensorflow_savedmodel,tensorrt_plan,ensemble}] [--verbose]
              {autoencoder} ...

    Export a DeepClean architecture to model repository
    for streaming inference, including adding models for
    caching input snapshot state as well as aggregated
    output state.

positional arguments:
  {autoencoder}

optional arguments:
  -h, --help            show this help message and exit
  --repository-directory REPOSITORY_DIRECTORY
                        Directory to which to save the models and their configs (default: None)
  --channels CHANNELS [CHANNELS ...]
                        A list of channel names used by DeepClean, with the strain channel first, or the path to a
                        text file containing this list separated by newlines (default: None)
  --weights WEIGHTS     Path to a set of trained weights with which to initialize the network architecture. If this
                        path is a directory, it should contain a file called `"weights.pt"`. (default: None)
  --kernel-length KERNEL_LENGTH
                        The length, in seconds, of the input to DeepClean (default: None)
  --stride-length STRIDE_LENGTH
                        The length, in seconds, between kernels sampled at inference time. This, along with the
                        `sample_rate`, dictates the size of the update expected at the snapshotter model (default:
                        None)
  --sample-rate SAMPLE_RATE
                        Rate at which the input kernel has been sampled, in Hz (default: None)
  --max-latency MAX_LATENCY [MAX_LATENCY ...]
                        The maximum amount of time, in seconds, allowed during inference to wait for overlapping
                        predictcions for online averaging. For example, if the `stride_length` is 0.002s and
                        `max_latency` is 0.5s, then output segments will be averaged over 250 overlapping kernels
                        before being streamed back from the server. This means there is a delay of `max_latency` (or
                        the greatest multiple of `stride_length` that is less than `max_latency`) seconds between the
                        start timestamp of the update streamed to the snapshotter and the resulting prediction
                        returned by the ensemble model. (default: None)
  --streams-per-gpu STREAMS_PER_GPU
                        The number of snapshot states to host per GPU during inference (default: 1)
  --instances INSTANCES
                        The number of concurrent execution instances of the DeepClean architecture to host per GPU
                        during inference (default: None)
  --platform {onnxruntime_onnx,tensorflow_savedmodel,tensorrt_plan,ensemble}
                        The backend framework platform used to host the DeepClean architecture on the inference
                        service. Right now only `"onnxruntime_onnx"` is supported. (default: Platform.ONNX)
  --verbose             If set, log at `DEBUG` verbosity, otherwise log at `INFO` verbosity. (default: False)
```
