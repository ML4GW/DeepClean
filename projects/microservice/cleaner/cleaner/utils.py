import time

from tritonclient import grpc as triton
from tritonclient.utils import InferenceServerException

from deepclean.logging import logger


def _wait_on_condition(func, msg, interval):
    start_time = time.time()
    while True:
        if func():
            break
        time.sleep(1)

        elapsed = int(time.time() - start_time)
        if not elapsed % interval:
            logger.info(f"{msg}, {elapsed}s elapsed")


def wait_for_server(url):
    client = triton.InferenceServerClient(url)

    # first wait for server to come online
    def _is_online():
        try:
            return client.is_server_live()
        except InferenceServerException:
            return False

    msg = "Waiting for server to come online"
    _wait_on_condition(_is_online, msg, 10)

    # now wait for the deepclean-stream model to
    # be ready, which means all its constituent
    # models are ready as well
    def _models_loaded():
        return client.is_model_ready("deepclean-stream")

    msg = "Waiting for streaming model to come online"
    _wait_on_condition(_models_loaded, msg, 10)

    # finally, wait for the latest version of
    # the model to move beyond the initial version,
    # which means we're ready to start doing inference
    # to validate this newly exported model
    def _is_trained_model_ready():
        metadata = client.get_model_metadata("deepclean")
        versions = list(map(int, metadata.versions))
        return max(versions) > 1

    msg = "Waiting for first trained DeepClean model to come online"
    _wait_on_condition(_is_trained_model_ready, msg, 10)
