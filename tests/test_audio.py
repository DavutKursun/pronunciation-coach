import io

import numpy as np
import soundfile as sf

from pronunciation.audio import load_audio, to_model_input


def test_int16_stereo_44k_becomes_mono_float_16k():
    stereo = (np.random.default_rng(0).standard_normal((44100, 2)) * 1000).astype(np.int16)
    audio = to_model_input(stereo, 44100)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == 16000
    assert np.abs(audio).max() <= 1.0


def test_load_audio_from_wav_bytes():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(8000, dtype=np.float32), 8000, format="WAV")
    assert len(load_audio(buffer.getvalue())) == 16000
