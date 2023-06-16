from typing import Iterable, List, Union

ChannelList = Union[str, Iterable[str]]


def get_channels(channels: ChannelList) -> List[str]:
    if isinstance(channels, str) or len(channels) == 1:
        if len(channels) == 1:
            channels = channels[0]

        try:
            with open(channels, "r") as f:
                channels = [i for i in f.read().splitlines() if i]
        except FileNotFoundError:
            raise FileNotFoundError(f"No channel file {channels} exists")

    channels = list(channels)
    return channels[:1] + sorted(channels[1:])
