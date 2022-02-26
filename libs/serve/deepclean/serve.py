import logging
from collections.abc import Iterable
from contextlib import contextmanager
from threading import Thread
from typing import Optional

from spython.main import Client as SingularityClient

DEFAULT_IMAGE = (
    "/cvmfs/singularity.opensciencegrid.org/fastml/gwiaas.tritonserver:latest"
)


@contextmanager
def serve(
    model_repo_dir: str,
    image: str = DEFAULT_IMAGE,
    gpus: Optional[Iterable[int]] = None,
    server_args: Optional[Iterable[str]] = None,
    log_file: Optional[str] = None,
) -> None:
    logging.debug(f"Starting instance of singularity image {image}")
    instance = SingularityClient.instance(image, options=["--nv"], quiet=True)

    cmd = ""
    if gpus is not None:
        cmd = "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus)) + " "

    cmd += "/opt/tritonserver/bin/tritonserver "
    cmd += "--model-repository " + model_repo_dir
    if server_args is not None:
        cmd += " ".join(server_args)

    if log_file is not None:
        cmd += f"> {log_file} 2>&1"
    logging.debug(
        f"Executing command '{cmd}' in singularity instance {instance.name}"
    )
    cmd = ["/bin/bash", "-c", cmd]

    thread = Thread(target=SingularityClient.execute, args=[instance, cmd])
    thread.start()

    try:
        yield
    finally:
        logging.debug(f"Stopping singularity instance {instance.name}")
        instance.stop()

        logging.debug(
            f"Singularity instance {instance.name} stopped, "
            "waiting for thread to close"
        )
        thread.join()
        logging.debug("Thread closed, exiting server context")
