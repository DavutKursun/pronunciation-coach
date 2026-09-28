import json

import numpy as np
import pytest

from conftest import VOCAB

torch = pytest.importorskip("torch")


def save_tiny_model(path):
    """A small random wav2vec2 CTC model whose tokenizer config asks for phonemization, like the real one."""
    from transformers import (Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                              Wav2Vec2PhonemeCTCTokenizer)

    vocab_file = path / "vocab.json"
    vocab_file.write_text(json.dumps({t: i for i, t in enumerate(VOCAB)}))
    # Build it with do_phonemize=False (building it with True would itself need eSpeak),
    # then switch the saved config to True, as in facebook/wav2vec2-lv-60-espeak-cv-ft.
    Wav2Vec2PhonemeCTCTokenizer(str(vocab_file), do_phonemize=False).save_pretrained(path)
    config_file = path / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config["do_phonemize"] = True
    config_file.write_text(json.dumps(config))

    Wav2Vec2FeatureExtractor().save_pretrained(path)
    config = Wav2Vec2Config(
        vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=37, conv_dim=(16,) * 7, num_conv_pos_embeddings=16, pad_token_id=0,
    )
    torch.manual_seed(0)
    Wav2Vec2ForCTC(config).save_pretrained(path)


def test_loads_without_espeak(tmp_path, monkeypatch):
    from pronunciation.recognizer import PhonemeRecognizer

    save_tiny_model(tmp_path)
    monkeypatch.setenv("PHONEMIZER_ESPEAK_LIBRARY", str(tmp_path / "missing" / "libespeak-ng.dylib"))

    recognizer = PhonemeRecognizer(str(tmp_path), device="cpu")
    result = recognizer.recognize(np.zeros(16000, dtype=np.float32))

    frames = int(recognizer.model._get_feat_extract_output_lengths(16000))
    assert frames == 49  # one frame every 20 ms
    assert result.log_probs.shape == (frames, len(VOCAB))
    assert recognizer.blank_id == 0
    assert result.seconds == 1.0
