import time
from dataclasses import dataclass
from pathlib import Path

from microservice.frames import FrameCrawler


@dataclass
class Deployment:
    root: Path

    def __post_init__(self):
        for d in ["log", "train", "csd", "frame", "repository"]:
            dirname = getattr(self, f"{d}_directory")
            dirname.mkdir(exist_ok=True, parents=True)

    @property
    def log_directory(self):
        return self.root / "logs"

    @property
    def train_directory(self):
        return self.root / "train"

    @property
    def csd_directory(self):
        return self.train_directory / "csds"

    @property
    def frame_directory(self):
        return self.root / "frames"

    @property
    def repository_directory(self):
        return self.root / "model_repo"


@dataclass(frozen=True)
class DataStream:
    root: Path
    field: str

    @property
    def detchar(self):
        return self.root / "lldetchar" / self.field

    @property
    def hoft(self):
        return self.root / "kafka" / self.field

    def crawl(self, t0, timeout):
        crawler = FrameCrawler(self.detchar, t0, timeout)
        for fname in crawler:
            strain_fname = fname.name.replace("lldetchar", "llhoft")
            strain_fname = self.hoft / strain_fname

            start_time = time.time()
            while not strain_fname.exists():
                if (time.time() - start_time) > 10:
                    raise FileNotFoundError(
                        f"Strain file {strain_fname} does not exist"
                    )

            yield fname, strain_fname
