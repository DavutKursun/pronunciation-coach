import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import VOCAB

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import finetune_recognizer as ft  # noqa: E402


def tokenizer(tmp_path):
    from transformers import Wav2Vec2PhonemeCTCTokenizer

    vocab_file = tmp_path / "vocab.json"
    vocab_file.write_text(json.dumps({t: i for i, t in enumerate(VOCAB)}))
    return Wav2Vec2PhonemeCTCTokenizer(str(vocab_file), do_phonemize=False)


def test_balanced_sampler_weighs_native_and_non_native_speech_the_same():
    rows = [{"l1": "Arabic"}] * 10 + [{"l1": "English"}] * 40
    drawn = list(ft.balanced_sampler(rows, seed=1))
    assert len(drawn) == 20                                   # one epoch = twice the non-native sentences
    many = torch.utils.data.WeightedRandomSampler(list(ft.balanced_sampler(rows, 1).weights), 20000,
                                                  generator=torch.Generator().manual_seed(0))
    native_share = np.mean([i >= 10 for i in many])
    assert 0.47 < native_share < 0.53


def test_speech_and_collate_pad_the_labels(tmp_path):
    from transformers import Wav2Vec2FeatureExtractor

    for name, seconds in (("a", 0.5), ("b", 1.0)):
        sf.write(tmp_path / f"{name}.flac", np.zeros(int(16000 * seconds), np.float32), 16000)
    rows = [{"audio": "a.flac", "tokens": "θ ɪ ŋ k"}, {"audio": "b.flac", "tokens": "t ɪ"}]
    data = ft.Speech(rows, tmp_path, tokenizer(tmp_path))
    batch = ft.Collate(Wav2Vec2FeatureExtractor())([data[0], data[1]])
    assert batch["input_values"].shape == (2, 16000)
    assert batch["labels"].tolist()[1][2:] == [-100, -100]
    assert batch["labels"].tolist()[0] == [VOCAB.index(p) for p in ["θ", "ɪ", "ŋ", "k"]]
    assert batch["attention_mask"][0].sum() == 8000


def test_unknown_target_token_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        ft.Speech([{"audio": "a.flac", "tokens": "θ ə1"}], tmp_path, tokenizer(tmp_path))
