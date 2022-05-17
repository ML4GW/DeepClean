import numpy as np

from deepclean.infer import pseudo


def test_online_postprocess(
    num_frames,
    frame_length,
    sample_rate,
    memory,
    look_ahead,
    validate_frame,
    postprocessor,
):
    x = np.arange(num_frames * frame_length * sample_rate)
    frames = pseudo.online_postprocess(
        x,
        x,
        frame_length,
        postprocessor,
        memory,
        look_ahead,
        sample_rate,
    )

    # last frame gets cut off
    assert len(frames) == (num_frames - 1)

    for i, frame in enumerate(frames):
        validate_frame(frame, i)
