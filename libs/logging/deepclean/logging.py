import logging
import os
import sys
from typing import Optional


class Logger:
    def __init__(self):
        self._logger = logging.getLogger()
        self._formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._cache = []

    def set_logger(
        self, name: str, filename: Optional[str] = None, verbose: bool = False
    ) -> None:
        self._logger = logging.getLogger(name)
        if name in self._cache:
            return self._logger
        self._cache.append(name)

        level = logging.DEBUG if verbose else logging.INFO
        self._logger.setLevel(level)

        use_stdout = os.getenv("DEEPCLEAN_LOG_STDOUT", "True")
        if eval(use_stdout):
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(self._formatter)
            handler.setLevel(level)
            self._logger.addHandler(handler)
        if filename is not None:
            handler = logging.FileHandler(filename=filename, mode="w")
            handler.setFormatter(self._formatter)
            handler.setLevel(level)
            self._logger.addHandler(handler)
        return self._logger

    def __getattr__(self, name):
        logger = object.__getattribute__(self, "_logger")
        try:
            return logger.__getattribute__(name)
        except AttributeError as e:
            raise AttributeError from e


logger = Logger()
