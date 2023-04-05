import logging
import os
import sys
from typing import Optional


class Logger:
    def __init__(self):
        self._logger = None
        self._formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._cache = []

    def configure_logger(
        self,
        logger,
        filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        add_stdout: bool = False
    ):
        if verbose is None:
            level = self._logger.level
        else:
            level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)

        if add_stdout:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(self._formatter)
            handler.setLevel(level)
            logger.addHandler(handler)

        if filename is not None:
            handler = logging.FileHandler(filename=filename, mode="w")
            handler.setFormatter(self._formatter)
            handler.setLevel(level)
            logger.addHandler(handler)
        return logger

    def get_logger(
        self,
        name: str,
        filename: Optional[str] = None,
        verbose: Optional[bool] = None,
        add_stdout: bool = False
    ):
        logger = logging.getLogger(name)
        if name in self._cache:
            return logger
        logger = self.configure_logger(logger, filename, verbose, add_stdout)
        self._cache.append(name)
        return logger

    def set_logger(
        self, name: str, filename: Optional[str] = None, verbose: bool = False
    ) -> None:
        if self._logger is None:
            logger = logging.getLogger()
            use_stdout = os.getenv("DEEPCLEAN_LOG_STDOUT", "True")
            use_stdout = eval(use_stdout)
            self.configure_logger(logger, filename, verbose, use_stdout)

            logger = self.get_logger(name, verbose=verbose, add_stdout=False)
        else:
            logger = self.get_logger(name, filename, verbose)
        self._logger = logger
        return logger

    def __getattr__(self, name):
        logger = object.__getattribute__(self, "_logger")
        try:
            return logger.__getattribute__(name)
        except AttributeError as e:
            raise AttributeError from e


logger = Logger()
