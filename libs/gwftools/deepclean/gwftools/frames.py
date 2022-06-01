import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, Union

PATH_LIKE = Union[str, Path]


prefix_re = "[a-zA-Z0-9_:-]+"
t0_re = "[0-9]{10}"
length_re = "[1-9][0-9]{0,3}"
fname_re = re.compile(
    f"(?P<prefix>{prefix_re})-"
    f"(?P<t0>{t0_re})-"
    f"(?P<length>{length_re})"
    ".(?P<suffix>gwf|hdf5|h5)$"
)


def parse_frame_name(fname: PATH_LIKE) -> Tuple[str, int, int]:
    """Use the name of a frame file to infer its initial timestamp and length

    Expects frame names to follow a standard nomenclature
    where the name of the frame file ends {prefix}_{timestamp}-{length}.gwf

    Args:
        fname: The name of the frame file
    Returns:
        The prefix of the frame file name
        The initial GPS timestamp of the frame file
        The length of the frame file in seconds
    """

    if isinstance(fname, Path):
        fname = fname.name

    match = fname_re.search(fname)
    if match is None:
        raise ValueError(f"Could not parse frame filename {fname}")

    prefix, t0, length, _ = match.groups()
    return prefix, int(t0), int(length)


@dataclass
class FrameFileFormat:
    prefix: str
    suffix: Literal["gwf", "hdf5", "h5"] = "gwf"

    @classmethod
    def from_frame_file(cls, frame_file: PATH_LIKE):
        prefix, _, __ = parse_frame_name(frame_file)
        suffix = Path(frame_file).suffix[1:]
        return cls(prefix, suffix)

    def get_name(self, timestamp: int, length: int):
        if int(timestamp) != timestamp:
            raise ValueError(f"Timestamp {timestamp} must be an int")
        elif len(str(timestamp)) != 10:
            raise ValueError(
                "Couldn't create valid GPS timestamp from timestamp {}".format(
                    timestamp
                )
            )

        if length <= 0:
            raise ValueError(f"Length {length} must be greater than 0")
        elif int(length) != length:
            raise ValueError(f"Length {length} must be an int")
        elif not 1 <= len(str(length)) < 5:
            raise ValueError(f"Frame length {length} invalid")

        return f"{self.prefix}-{timestamp}-{length}.{self.suffix}"
