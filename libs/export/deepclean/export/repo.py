from typing import List, Optional

from tritonclient.grpc import model_config_pb2 as model_config

from hermes import quiver as qv


def configure_stateful(config: model_config.ModelConfig, idle_time: float):
    idle_time = int(idle_time * 10**6)
    config.sequence_batching.max_sequence_idle_microseconds = idle_time


def initialize_repo(
    repo_dir: str,
    channels: List[str],
    platform: qv.Platform,
    instances: Optional[int] = None,
    clean: bool = False,
):
    repo = qv.ModelRepository(repo_dir, clean=clean)
    try:
        deepclean = repo.models["deepclean"]
    except KeyError:
        deepclean = repo.add("deepclean", platform=platform)
        if instances is not None:
            deepclean.config.add_instance_group(count=instances)
    else:
        if instances is not None:
            deepclean.config.scale_instance_group(count=instances)

    deepclean.config.parameters["channels"].string_value = ",".join(channels)
    return repo
