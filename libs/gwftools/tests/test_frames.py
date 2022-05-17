from pathlib import Path

import pytest

from deepclean.gwftools.frames import FrameFileFormat, parse_frame_name


@pytest.fixture(params=[str, Path])
def path_type(request):
    return request.param


@pytest.fixture(params=[None, ".", "/dev/shm/H1"])
def frame_dir(request):
    return request.param


def test_parse_frame_name(path_type, frame_dir):
    prefix = "H-H1_llhoft"
    if frame_dir is not None:
        full_prefix = str(Path(frame_dir) / prefix)
    else:
        full_prefix = prefix

    tstamp = 1234567890
    for frame_length in [1, 1024, 4096, 8192]:
        frame_name = f"{full_prefix}-{tstamp}-{frame_length}.gwf"
        prfx, t0, length = parse_frame_name(path_type(frame_name))
        assert prfx == prefix
        assert t0 == tstamp
        assert length == frame_length

    # make sure having the wrong file extension raises an error
    for postfix in ["", ".gwff", ".GWF"]:
        with pytest.raises(ValueError):
            parse_frame_name(
                path_type(f"{full_prefix}-{tstamp}-{frame_length}{postfix}")
            )

    # make sure timestamps of incorrect length raises an error
    for bad_tstamp in [12345678901, 123456789, 1234567890.75]:
        with pytest.raises(ValueError):
            parse_frame_name(
                path_type(f"{full_prefix}-{bad_tstamp}-{frame_length}.gwf")
            )

    # make sure bad frame lengths raise an error
    for bad_length in [0, 16384]:
        with pytest.raises(ValueError):
            parse_frame_name(
                path_type(f"{full_prefix}-{tstamp}-{bad_length}.gwf")
            )


def test_frame_file_format(path_type, frame_dir):
    prefix = "H-H1_llhoft"
    tstamp = 1234567890
    length = 1

    file_format = FrameFileFormat(prefix)
    assert file_format.get_name(tstamp, length) == (
        f"{prefix}-{tstamp}-{length}.gwf"
    )

    if frame_dir is not None:
        full_prefix = str(Path(frame_dir) / prefix)
    else:
        full_prefix = prefix
    file_format = FrameFileFormat.from_frame_file(
        path_type(f"{full_prefix}-{tstamp}-{length}.gwf")
    )
    assert file_format.prefix == prefix
    assert file_format.get_name(tstamp + 1, length) == (
        f"{prefix}-{tstamp + 1}-{length}.gwf"
    )
