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
```
