"""Audio helpers: everything is converted to mono float32 at 16 kHz, what wav2vec2 expects."""

from __future__ import annotations

import io
from math import gcd

import numpy as np

SAMPLE_RATE = 16_000


def to_model_input(data, sample_rate: int) -> np.ndarray:
    """Convert raw samples (int or float, mono or stereo, any rate) to mono float32 16 kHz."""
    audio = np.asarray(data)
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype.kind == "u":
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        # (samples, channels) is the usual layout; (channels, samples) if the first axis is tiny
        audio = audio.mean(axis=1) if audio.shape[1] <= audio.shape[0] else audio.mean(axis=0)

    if sample_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(SAMPLE_RATE, int(sample_rate))
        audio = resample_poly(audio, SAMPLE_RATE // g, int(sample_rate) // g)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio(source) -> np.ndarray:
    """Load a file path or raw bytes (e.g. a WAV file from a dataset)."""
    import soundfile as sf

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    data, sample_rate = sf.read(source, dtype="float32")
    return to_model_input(data, sample_rate)
