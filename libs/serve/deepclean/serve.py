import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager
from queue import Empty, Queue
from threading import Thread
from typing import Optional

from spython.main import Client as SingularityClient
from tritonclient import grpc as triton

DEFAULT_IMAGE = (
    "/cvmfs/singularity.opensciencegrid.org/fastml/gwiaas.tritonserver:latest"
)


def run_server(instance, command):
    command = ["/bin/bash", "-c", command]
    response = SingularityClient.execute(instance, command)
    return response


def run_server_as_target(instance, command, queue):
    response = run_server(instance, command)
    queue.put(response)


def wait_for_start(
    instance, queue: Queue, url: str = "localhost:8001", interval: int = 10
) -> None:
    client = triton.InferenceServerClient(url)

    logging.info("Waiting for server to come online")
    start_time, i = time.time(), 1
    while True:
        try:
            if client.is_server_live():
                break
        except triton.InferenceServerException:
            try:
                response = queue.get_nowait()
                raise ValueError(
                    "Server failed to start with return code {return_code} "
                    "and message:\n{message}".format(**response)
                )
            except Empty:
                pass
        finally:
            elapsed = time.time() - start_time
            if elapsed >= (i * interval):
                logging.debug(
                    "Still waiting for server to start, "
                    "{}s elapsed".format(i * interval)
                )
                i += 1
    logging.info("Server online")


@contextmanager
def serve(
    model_repo_dir: str,
    image: str = DEFAULT_IMAGE,
    gpus: Optional[Iterable[int]] = None,
    server_args: Optional[Iterable[str]] = None,
    log_file: Optional[str] = None,
    wait: bool = False,
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
    response_queue = Queue()
    thread = Thread(
        target=run_server_as_target, args=[instance, cmd, response_queue]
    )
    thread.start()

    try:
        if wait:
            wait_for_start(instance, response_queue)
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
