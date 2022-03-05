import logging
import os
from typing import Callable, List, Optional, Union

import hermes.quiver as qv
import torch

from deepclean.logging import configure_logging
from deepclean.networks import typeo_wrapper


def make_ensemble(
    repo: qv.ModelRepository,
    deepclean: qv.Model,
    ensemble_name: str,
    snapshotter: Optional[qv.Model],
    stream_size: int,
    num_updates: int,
    streams_per_gpu: int = 1,
) -> qv.Model:
    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one and add a snapshotter
        # instance to it, possibly creating one if need be
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)

        if snapshotter is None:
            # there's no snapshotter, so make one
            ensemble.add_streaming_inputs(
                inputs=[deepclean.inputs["witness"]],
                stream_size=stream_size,
                name="snapshotter",
                streams_per_gpu=streams_per_gpu,
            )
        else:
            # pipe the output of the existing snapshotter
            # model to DeepClean's witness input
            ensemble.pipe(
                list(snapshotter.outputs.values())[0],
                deepclean.inputs["witness"],
            )
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has DeepClean
        # and the snapshotter as a part of its models
        if deepclean not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'deepclean'".format(ensemble_name)
            )
        elif snapshotter is None or snapshotter not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'snapshotter'".format(ensemble_name)
            )

    snapshotter = repo.models["snapshotter"]
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e9
    )
    snapshotter.config.write()

    # add a streaming output to the model if it
    # doesn't already have one
    if len(ensemble.config.output) == 0:
        if ensemble_name == "deepclean-stream":
            name = "aggregator"
        else:
            name = "aggregator-" + ensemble.name.replace(
                "deepclean-stream-", ""
            )
        ensemble.add_streaming_output(
            deepclean.outputs["noise"],
            update_size=stream_size,
            num_updates=num_updates,
            name=name,
            streams_per_gpu=streams_per_gpu,
        )
        aggregator = repo.models[name]
        aggregator.config.sequence_batching.max_sequence_idle_microseconds = (
            int(6e9)
        )
        aggregator.config.write()

    ensemble.export_version(None)
    return ensemble


@typeo_wrapper
def export(
    architecture: Callable,
    repository_directory: str,
    channels: Union[str, List[str]],
    weights: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    max_latency: Union[float, List[float]],
    streams_per_gpu: int = 1,
    instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    verbose: bool = False,
) -> None:
    if os.path.isdir(weights):
        output_directory = weights
        weights = os.path.join(output_directory, "weights.pt")
    else:
        output_directory = os.path.dirname(weights)
    configure_logging(os.path.join(output_directory, "export.log"), verbose)

    logging.info(f"Creating model and loading weights from {weights}")
    nn = architecture(len(channels) - 1)
    nn.load_state_dict(torch.load(weights))
    nn.eval()

    repo = qv.ModelRepository(repository_directory)
    try:
        model = repo.models["deepclean"]
        if instances is not None:
            model.config.scale_instance_group(instances)
    except KeyError:
        model = repo.add("deepclean", platform=platform)
        if instances is not None:
            model.config.add_instance_group(count=instances)

    input_shape = (1, len(channels) - 1, int(kernel_length * sample_rate))
    model.export_version(
        nn, input_shapes={"witness": input_shape}, output_names=["noise"]
    )

    if isinstance(max_latency, float):
        max_latency = [max_latency]
        ensemble_name = "deepclean-stream"
    else:
        ensemble_name = "deepclean-stream-{}"

    for latency in max_latency:
        name = ensemble_name.format(latency)
        num_updates = int(latency // stride_length)
        logging.info(
            "Creating ensemble model {} which averages outputs "
            "over {} updates".format(name, num_updates)
        )

        make_ensemble(
            repo,
            model,
            name,
            repo.models.get("snapshotter", None),
            stream_size=int(sample_rate * stride_length),
            num_updates=num_updates,
            streams_per_gpu=streams_per_gpu,
        )


if __name__ == "__main__":
    export()
