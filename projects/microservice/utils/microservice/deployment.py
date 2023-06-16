import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from microservice.frames import FrameCrawler

from deepclean.logging import logger


@dataclass
class Deployment:
    root: Path

    def __post_init__(self):
        for name, prop in self.__class__.__dict__.items():
            if isinstance(prop, property) and name.endswith("directory"):
                dirname = getattr(self, name)
                dirname.mkdir(exist_ok=True, parents=True)

    @property
    def log_directory(self):
        return self.root / "logs"

    @property
    def train_directory(self):
        return self.root / "train"

    @property
    def data_checks_directory(self):
        return self.train_directory / "checks"

    @property
    def repository_directory(self):
        return self.root / "model_repo"

    @property
    def infer_directory(self):
        return self.root / "infer"

    @property
    def frame_directory(self):
        return self.infer_directory / "frames"

    @property
    def storage_directory(self):
        return self.infer_directory / "storage"


@dataclass(frozen=True)
class DataStream:
    root: Path
    ifo: str
    strain_offset: float = 7

    @property
    def detchar(self):
        return self.root / "lldetchar" / self.ifo

    @property
    def hoft(self):
        return self.root / "kafka" / self.ifo

    def crawl(self, t0, timeout):
        crawler = FrameCrawler(self.detchar, t0, timeout, max_dropped_frames=5)
        for fname in crawler:
            strain_fname = fname.name.replace("lldetchar", "llhoft")
            strain_fname = self.hoft / strain_fname

            tick = time.time()
            timeout = 3 + self.strain_offset
            while time.time() < tick + timeout:
                if strain_fname.exists():
                    break
            else:
                raise FileNotFoundError(
                    "Strain file {} still doesn't exist after "
                    "waiting for {} seconds".format(strain_fname, timeout)
                )

            yield fname, strain_fname


@dataclass
class ExportClient:
    endpoint: str

    def __post_init__(self):
        self.logger = logger.get_logger("Export client")
        self._make_request("alive")

    def _make_request(self, target, *args: str):
        url = f"http://{self.endpoint}/{target}"
        if args:
            url += "/" + "/".join(args)

        self.logger.debug(f"Making request to {url}")
        r = requests.get(url)
        r.raise_for_status()
        self.logger.debug(f"Request to {url} successful")
        return r.content

    def export(self, weights_path: Path) -> None:
        weights_dir = weights_path.parent.name
        self._make_request("export", weights_dir)
        return None

    def get_production_version(self) -> int:
        version = self._make_request("production-version")
        return int(version)

    def set_production_version(self, version: Optional[int] = None):
        version = version or -1
        version = self._make_request("production-version/set", str(version))
        return int(version)

    def get_latest_version(self) -> int:
        version = self._make_request("latest-version")
        return int(version)
