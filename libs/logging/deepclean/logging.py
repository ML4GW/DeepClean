import logging
import sys
from typing import Optional


def configure_logging(
    filename: Optional[str] = None, verbose: bool = False
) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        stream=sys.stdout,
    )

    if filename is not None:
        logger = logging.getLogger()
        logger.addHandler(
            logging.handlers.FileHandler(filename=filename, mode="w")
        )
