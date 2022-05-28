import train


def test_make_fake_sines():
    channels = [train.FAKE_ID + "18POINT37HZ"]
    result = train.make_fake_sines(channels, 10, 100, 256)

    # TODO: forcing this to fail for now so that we
    # need to fill the rest of these out
    assert (result == 0).all()


def test_fetch():
    return


def test_read():
    return


def test_write():
    return


def test_main():
    return
