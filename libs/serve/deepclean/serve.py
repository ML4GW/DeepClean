import logging
import time
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
    # start a container instance from the specified image with
    # the --nv flag set in order to utilize GPUs
    logging.debug(f"Starting instance of singularity image {image}")
    instance = SingularityClient.instance(image, options=["--nv"], quiet=True)

    # specify GPUs at the front of the command using
    # CUDA_VISIBLE_DEVICES environment variable
    cmd = ""
    if gpus is not None:
        cmd = "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus)) + " "

    # create the base triton server command and
    # point it at the model repository
    cmd += "/opt/tritonserver/bin/tritonserver "
    cmd += "--model-repository " + model_repo_dir

    # add in any additional arguments to the server
    if server_args is not None:
        cmd += " ".join(server_args)

    # if we specified a log file, reroute stdout and stderr
    # to that file (triton primarily uses stderr)
    if log_file is not None:
        cmd += f" > {log_file} 2>&1"

    # execute the command inside the running container
    # instance, wrapping with /bin/bash so that the
    # CUDA_VISIBLE_DEVICE environment variable gets set.
    # Run it in a separate thread so that we can do work
    # while it runs in this same process
    logging.debug(
        f"Executing command '{cmd}' in singularity instance {instance.name}"
    )
    cmd = ["/bin/bash", "-c", cmd]
    thread = Thread(target=SingularityClient.execute, args=[instance, cmd])
    thread.start()

    try:
        time.sleep(10)
        yield
    finally:
        # stop the instance and wait for its thread to terminate
        logging.debug(f"Stopping singularity instance {instance.name}")
        instance.stop()

        logging.debug(
            f"Singularity instance {instance.name} stopped, "
            "waiting for thread to close"
        )
        thread.join()
        logging.debug("Thread closed, exiting server context")
