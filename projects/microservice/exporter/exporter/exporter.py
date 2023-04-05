import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import tritonclient.grpc.model_config_pb2 as model_config
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
        self.num_witnesses = len(self.channels) - 1

        # start by initializing a full model instance
        # we'll use to populate weights into
        preprocessor = ChannelWiseScaler(self.num_witnesses)
        postprocessor = ChannelWiseScaler()
        deepclean = architecture(self.num_witnesses)
        self.model = DeepClean(preprocessor, deepclean, postprocessor)

        # set these manually since we don't actually
        # need to do any fitting, but we're going to
        # export a dummy version before we load any weights
        preprocessor.built = True
        postprocessor.built = True

        # define a few helpful values
        kernel_size = int(self.sample_rate * self.kernel_length)
        stream_size = int(self.sample_rate / self.inference_sampling_rate)
        num_updates = int(self.aggregation_time * self.inference_sampling_rate)
        input_shape = (self.batch_size, self.num_witnesses, kernel_size)
        output_shape = (self.batch_size, kernel_size)

        # initialize an empty repository with an
        # associated deepclean entry already in it
        self.repo = repo_utils.initialize_repo(
            self.repository_directory,
            self.channels[1:],
            self.platform,
            self.instances,
            clean=True
        )

        # initialize an ensemble that we'll add steps to
        ensemble = self.repo.add(
            "deepclean-stream", platform=qv.Platform.ENSEMBLE
        )

        # start by creating a repo entry
        # for the preprocessing model
        preproc = self.repo.add("preprocessor", platform=qv.Platform.ONNX)
        preproc.export_version(
            self.model.preprocessor,
            input_shapes={"witnesses": input_shape},
            output_names=["normalized"]
        )
        self.set_version_policy(preproc)

        # now create a snapshotter instance
        # for this preprocessor, mapped to
        # the latest version for the canary model
        ensemble.add_streaming_inputs(
            inputs=[preproc.inputs["witnesses"]],
            stream_size=stream_size,
            name="snapshotter",
            streams_per_gpu=self.streams_per_gpu,
            batch_size=self.batch_size,
            versions=-1
        )

        # now create a second mapping from the
        # snapshot state to the fixed production
        # version of the preprocessor model
        snapshotter = self.repo.models["snapshotter"]
        ensemble.pipe(
            snapshotter.outputs["preprocessor.witnesses_snapshot"],
            preproc.inputs["witnesses"],
            inbound_version=1
        )

        # now export deepclean, and create mappings
        # from each version of the preprocessor to
        # the corresponding version of deepclean
        self.repo.models["deepclean"].export_version(
            deepclean,
            input_shapes={"normalized": input_shape},
            output_names=["noise_prediction"]
        )
        self.set_version_policy(self.repo.models["deepclean"])

        keys = {"canary": -1, "production": 1}
        for key, version in keys.items():
            ensemble.pipe(
                preproc.outputs["normalized"],
                self.repo.models["deepclean"].inputs["normalized"],
                key=f"normalized-{key}",
                outbound_version=version,
                inbound_version=version
            )

        # now do the same for a postprocessing model,
        # adding a streaming output from the model
        # to the back end of our ensemble
        postproc = self.repo.add("postprocessor", platform=qv.Platform.ONNX)
        postproc.export_version(
            self.model.postprocessor,
            input_shapes={"noise_prediction": output_shape},
            output_names=["unscaled"]
        )
        self.set_version_policy(postproc)
        for key, version in keys.items():
            ensemble.pipe(
                self.repo.models["deepclean"].outputs["noise_prediction"],
                postproc.inputs["noise_prediction"],
                key=f"noise_prediction-{key}",
                outbound_version=version,
                inbound_version=version
            )
            ensemble.add_streaming_output(
                postproc.outputs["unscaled"],
                update_size=stream_size,
                num_updates=num_updates,
                batch_size=self.batch_size,
                name=f"aggregator-{key}",
                key=f"unscaled-{key}",
                streams_per_gpu=self.streams_per_gpu,
                version=version
            )
        ensemble.export_version(None)

    def set_version_policy(self, model: qv.Model):
        model.config.version_policy.MergeFrom(
            model_config.ModelVersionPolicy(
                latest=model_config.ModelVersionPolicy.Latest(num_versions=2)
            )
        )
        model.config.write()

    @property
    def preprocessor(self):
        return self.model.preprocessor

    @property
    def postprocessor(self):
        return self.model.postprocessor

    @property
    def deepclean(self):
        return self.model.deepclean

    def update_ensemble_versions(self, version):
        ensemble = self.repo.models["deepclean-stream"]
        for step in ensemble.config.steps:
            if step.model_version == -1:
                continue
            if step.model_name not in ["snapshotter", "aggregator"]:
                step.model_version = version
        ensemble.config.write()

    def export(self):
        self.repo.models["preprocessor"].export_version(self.preprocessor)
        self.repo.models["postprocessor"].export_version(self.postprocessor)

        kwargs = {}
        if self.platform == qv.Platform.TENSORRT:
            kwargs["use_fp16"] = True
        export_path = self.repo.models["deepclean"].export_version(
            self.deepclean, **kwargs
        )
        version = int(Path(export_path).parent.name)
        return version

    def export_weights(self, train_dir: Path):
        weights_path = train_dir / "weights.pt"

        state_dict = torch.load(weights_path)
        self.deepclean.load_state_dict(state_dict)
        version = self.export()

        _, start, stop = train_dir.name.split("-")
        self.deepclean.config.parameters["versions"] += "{}={}-{},".format(
            version, start, stop
        )
        self.deepclean.config.write()
