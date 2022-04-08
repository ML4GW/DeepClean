from typing import Iterable, List, Union

ChannelList = Union[str, Iterable[str]]


def get_channels(channels: ChannelList) -> List[str]:
    if isinstance(channels, str):
        try:
            with open(channels, "r") as f:
                channels = [i for i in f.read().splitlines() if i]
        except FileNotFoundError:
            raise FileNotFoundError(f"No channel file {channels} exists")
    channels = list(channels)
    return channels[0] + sorted(channels[1:])
