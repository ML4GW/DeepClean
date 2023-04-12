from dataclasses import dataclass
from pathlib import Path

from microservice.frames import FrameCrawler


@dataclass
class Deployment:
    root: Path

    def __post_init__(self):
        for d in ["log", "train", "frame", "repository"]:
            dirname = getattr(self, f"{d}_directory")
            dirname.mkdir(exist_ok=True, parents=True)

    @property
    def log_directory(self):
        return self.root / "logs"

    @property
    def train_directory(self):
        return self.root / "train"

    @property
    def frame_directory(self):
        return self.root / "frames"

    @property
    def repository_directory(self):
        return self.root / "model_repo"


@dataclass(frozen=True)
class DataStream:
    root: Path

    @property
    def detchar(self):
        return self.root / "lldetchar"

    @property
    def hoft(self):
        return self.root / "llhoft"

    def crawl(self, t0, timeout):
        crawler = FrameCrawler(self.detchar, t0, timeout)
        for fname in crawler:
            strain_fname = fname.name.replace("Detchar", "HOFT")
            strain_fname = self.hoft / strain_fname
            yield fname, strain_fname
