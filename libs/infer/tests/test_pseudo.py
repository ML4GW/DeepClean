import numpy as np

from deepclean.infer import pseudo


def test_online_postprocess(
    num_frames,
    frame_length,
    sample_rate,
    filter_lead_time,
    filter_memory,
    validate_frame,
    postprocessor,
):
    x = np.arange(num_frames * frame_length * sample_rate)
    frames = pseudo.online_postprocess(
        x,
        x,
        frame_length,
        postprocessor,
        filter_memory,
        filter_lead_time,
        sample_rate,
    )

    # last frame gets cut off
    assert len(frames) == (num_frames - 1)

    for i, frame in enumerate(frames):
        validate_frame(frame, i)
