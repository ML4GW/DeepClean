import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from ml4gw.transforms import ChannelWiseScaler
from typeo.parser import make_parser

import hermes.quiver as qv
from deepclean.architectures import DeepCleanAE as architecture
from deepclean.utils.channels import ChannelList, get_channels
from deepclean.export import repo as repo_utils
from deepclean.export.model import DeepClean


@dataclass
class Exporter:
    repository_directory: str
    channels: ChannelList
    kernel_length: float
    sample_rate: float
    inference_sampling_rate: float
    batch_size: int
    aggregation_time: float
    streams_per_gpu: int = 2
    platform: qv.Platform = qv.Platform.ONNX
    instances: Optional[int] = None

    @classmethod
    def from_config(cls, config: Optional[Path] = None):
        if config is None:
            config = Path(__file__).parent.parent / "pyproject.toml"

        parser = argparse.ArgumentParser()
        make_parser(cls, parser)
        args = parser.parse_args(["--typeo", config])
        return cls(**vars(args))

    def __post_init__(self):
        self.channels = get_channels(self.channels)
        self.repo = repo_utils.initialize_repo(
            self.repository_directory,
            self.channels[1:],
            self.platform,
            self.instances
        )

        self.num_witnesses = len(self.channels) - 1
        preprocessor = ChannelWiseScaler(self.num_witnesses)
        postprocessor = ChannelWiseScaler()

        preprocessor.built = True
        postprocessor.built = True

        deepclean = architecture(self.num_witnesses)
        self.model = DeepClean(preprocessor, deepclean, postprocessor)

        kernel_size = int(self.sample_rate * self.kernel_length)
        stream_size = int(self.sample_rate / self.inference_sampling_rate)
        num_updates = int(self.aggregation_time * self.inference_sampling_rate)

        input_shape = (self.batch_size, self.num_witnesses, kernel_size)
        preproc = self.repo.add("preprocessor", platform=qv.Platform.ONNX)
        preproc.export_version(
            self.model.preprocessor,
            input_shapes={"witnesses": input_shape},
            output_names=["normalized"]
        )

        self.repo.models["deepclean"].export_version(
            deepclean,
            input_shapes={"normalized": input_shape},
            output_names=["noise_prediction"]
        )

        postproc = self.repo.add("postprocessor", platform=qv.Platform.ONNX)
        postproc.export_version(
            self.model.postprocessor,
            input_shapes={"noise_prediction": (self.batch_size, kernel_size)},
            output_names=["unscaled"]
        )

        production = self.repo.add(
            "deepclean-stream-production", platform=qv.Platform.ENSEMBLE
        )
        production.add_streaming_inputs(
            inputs=[preproc.inputs["witnesses"]],
            stream_size=stream_size,
            name="snapshotter",
            streams_per_gpu=self.streams_per_gpu,
            batch_size=self.batch_size
        )
        production.add_streaming_output(
            postproc.outputs["unscaled"],
            update_size=stream_size,
            num_updates=num_updates,
            batch_size=self.batch_size,
            name="aggregator",
            streams_per_gpu=self.streams_per_gpu
        )
        self.construct_ensemble(production)

        snapshotter = self.repo.models["snapshotter"]
        aggregator = self.repo.models["aggregator"]

        # now set up the non-production model, piping
        # inputs and outputs to the snaphotter and
        # aggregator models explicitly
        canary = self.repo.add(
            "deepclean-stream-canary", platform=qv.Platform.ENSEMBLE
        )
        canary.pipe(
            snapshotter.outputs["preprocessor.witnesses_snapshot"],
            preproc.inputs["witnesses"]
        )
        self.construct_ensemble(canary)
        canary.pipe(
            postproc.outputs["unscaled"],
            aggregator.inputs["update"]
        )

        self.configure_stateful(self.repo.models["snapshotter"])
        self.configure_stateful(self.repo.models["aggregator"])

    def configure_stateful(self, model):
        repo_utils.configure_stateful(model.config, 60)
        model.config.write()

    def construct_ensemble(self, ensemble):
        ensemble.pipe(
            self.repo.models["preprocessor"].outputs["normalized"],
            self.repo.models["deepclean"].inputs["normalized"],
        )
        ensemble.pipe(
            self.repo.models["deepclean"].outputs["noise_prediction"],
            self.repo.models["postprocessor"].inputs["noise_prediction"]
        )
        self.update_ensemble_versions(ensemble, 1)

    def update_ensemble_versions(self, ensemble, version):
        for step in ensemble.config.ensemble_scheduling.step:
            if step.model_name not in ["snapshotter", "aggregator"]:
                step.model_version = version
        ensemble.config.write()

    @property
    def preprocessor(self):
        return self.model.preprocessor

    @property
    def postprocessor(self):
        return self.model.postprocessor

    @property
    def deepclean(self):
        return self.model.deepclean

    def export_version(self, train_dir: Path):
        weights_path = train_dir / "weights.pt"

        state_dict = torch.load(weights_path)
        self.deepclean.load_state_dict(state_dict)

        self.repo.models["preprocessor"].export_version(self.preprocessor)
        self.repo.models["postprocessor"].export_version(self.postprocessor)

        kwargs = {}
        if self.platform == qv.Platform.TENSORRT:
            kwargs["use_fp16"] = True
        export_path = self.repo.models["deepclean"].export_version(
            self.deepclean, **kwargs
        )
        version = int(Path(export_path).parent.name)
        self.update_ensemble_versions(
            self.repo.models["deepclean-stream-canary"], version
        )

        _, start, stop = train_dir.name.split("-")
        self.deepclean.config.parameters["versions"] += "{}={}-{},".format(
            version, start, stop
        )
        self.deepclean.config.write()
