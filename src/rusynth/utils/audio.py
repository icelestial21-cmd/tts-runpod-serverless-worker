import numpy as np
from numpy.typing import NDArray
from pydub import AudioSegment


def low_pass_filter(
        audio: NDArray,
        sampling_rate: int,
        channels: int,
        cutoff: int
) -> NDArray:
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate = sampling_rate,
        sample_width = audio.dtype.itemsize,
        channels = 1
    )
    audio_segment = audio_segment.low_pass_filter(cutoff)
    return np.array(audio_segment.get_array_of_samples())


def volume_to_dbfs(volume: float) -> float:
    """ Converts volume to dBFS. """
    scale = [
        (0.0, -float('inf')),
        (0.1, -40),
        (0.2, -30),
        (0.3, -24),
        (0.4, -20),
        (0.5, -16),
        (0.6, -12),
        (0.7, -9),
        (0.8, -6),
        (0.9, -3),
        (1.0, 0)
    ]
    if volume <= 0:
        return -float('inf')
    elif volume >= 1:
        return 0
    for (p1, db1), (p2, db2) in zip(scale, scale[1:]):
        if p1 <= volume <= p2:
            return db1 + (db2 - db1) * (volume - p1) / (p2 - p1)


def normalize_audio(
        audio_array,
        sampling_rate = 22050,
        sample_width = 2,
        channels = 1,
        target_dBFS = -10.0
):
    assert audio_array.dtype == np.int16, "Audio array must be of type np.int16"

    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate = sampling_rate,
        sample_width = sample_width,
        channels = channels
    )

    change_in_dBFS = target_dBFS - audio_segment.dBFS
    normalized_audio_segment = audio_segment.apply_gain(change_in_dBFS)

    normalized_audio_array = np.array(
        normalized_audio_segment.get_array_of_samples()
    )

    return normalized_audio_array
