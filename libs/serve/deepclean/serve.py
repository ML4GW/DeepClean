from collections.abc import Iterable
from contextlib import contextmanager
from threading import Thread
from typing import Optional

from spython.main import Client as SingularityClient


@contextmanager
def serve(
    image: str, model_repo_dir: str, gpus: Optional[Iterable[int]], *args
) -> None:
    instance = SingularityClient.instance(image, options=["--nv"], quiet=True)

    cmd = []
    if gpus is not None:
        cmd.append("CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpus)) + " ")

    cmd.append("/opt/tritonserver/bin/tritonserver")
    cmd.append(f"--model-repository {model_repo_dir}")
    cmd.extend(args)

    thread = Thread(target=SingularityClient.execute, args=[instance, cmd])
    thread.start()

    try:
        yield
    finally:
        instance.stop()
        thread.join()
