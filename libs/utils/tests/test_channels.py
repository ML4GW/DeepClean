import os
import random

import pytest

from deepclean.gwftools.channels import get_channels


@pytest.fixture
def tmpfile():
    tmpfile = "channels.txt"
    yield tmpfile
    if os.path.exists(tmpfile):
        os.remove(tmpfile)


def test_get_channels(tmpfile):
    channels = list("ZBCDEFG")

    def check_outputs(input_channels):
        assert get_channels(input_channels) == channels

        # make sure that an extra newline will get
        # stripped out when we read the file
        for extra_character in ["", "\n"]:
            with open(tmpfile, "w") as f:
                f.write("\n".join(input_channels))
                f.write(extra_character)

            # verify that parsing the file as-is
            # and as part of length-1 list comes out right
            assert get_channels(tmpfile) == channels
            assert get_channels([tmpfile]) == channels

    check_outputs(channels)

    # make sure that if we reorder the "witness"
    # channels they come out in alphabetical order,
    # with the "strain" channel still first
    shuffled = channels[1:]
    random.shuffle(shuffled)
    check_outputs(channels[:1] + shuffled)
