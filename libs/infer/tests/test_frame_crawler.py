import pytest

from deepclean.gwftools.frames import parse_frame_name
from deepclean.infer.frame_crawler import FrameCrawler, get_prefix


@pytest.fixture(params=[1, 1024, 4096])
def frame_length(request):
    return request.param


@pytest.fixture(params=[None, "name", "prefix", "length"])
def error_type(request):
    return request.param


@pytest.fixture
def prefix():
    return "H-H1_llhoft"


@pytest.fixture
def timestamp():
    return 1234567890


@pytest.fixture(scope="function")
def data_dir(write_dir, frame_length, error_type, prefix, timestamp):
    for i in range(10):
        postfix = ".gwf"
        length = frame_length
        prfx = prefix

        if error_type == "name":
            postfix = ".gwff"
        elif i != 5:
            # for other error types, only make one
            # file not match the rest
            pass
        elif error_type == "prefix":
            prfx = prefix + "1"
        elif error_type == "length":
            length = length + 1

        fname = f"{prfx}_{timestamp + i * length}-{length}{postfix}"
        with open(write_dir / fname, "w"):
            pass
    return write_dir


def test_get_prefix(data_dir, prefix, frame_length, error_type):
    if error_type is None:
        prfx, length = get_prefix(data_dir)
        assert prfx == prefix
        assert length == frame_length
        return
    elif error_type == "name":
        match = "^No valid .gwf files"
    elif error_type == "prefix":
        match = "^Too many prefixes"
    elif error_type == "length":
        match = "^Too many lengths"

    with pytest.raises(ValueError, match=match):
        get_prefix(data_dir)


@pytest.mark.parametrize("start_first", [True, False])
def test_frame_crawler(
    data_dir, prefix, timestamp, frame_length, error_type, start_first
):
    if error_type is not None:
        with pytest.raises(ValueError):
            FrameCrawler(data_dir, data_dir)
        return

    # TODO: add check for mismatched lengths
    # between the two data directories
    crawler = FrameCrawler(
        data_dir, data_dir, start_first=start_first, timeout=0.1
    )
    assert crawler.length == frame_length

    iterated = iter(crawler)
    for i in range(10):
        witness_fname, strain_fname = next(iterated)
        expected_tstamp = timestamp + (i if start_first else 9) * frame_length
        for fname in [witness_fname, strain_fname]:
            assert fname.parent == data_dir

            prfx, t0, length = parse_frame_name(fname)
            assert prfx == prefix
            assert t0 == expected_tstamp
            assert length == frame_length

        if not start_first:
            break

    with pytest.raises(RuntimeError):
        next(iterated)
