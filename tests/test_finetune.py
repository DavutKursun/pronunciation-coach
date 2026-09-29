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


def test_checkpoints_are_saved_in_float16_and_load_as_float32(tmp_path):
    from test_recognizer import save_tiny_model
    from transformers import AutoFeatureExtractor, AutoTokenizer, Wav2Vec2ForCTC

    (tmp_path / "base").mkdir()
    save_tiny_model(tmp_path / "base")
    model = Wav2Vec2ForCTC.from_pretrained(tmp_path / "base")
    extractor = AutoFeatureExtractor.from_pretrained(tmp_path / "base")
    tok = AutoTokenizer.from_pretrained(tmp_path / "base", do_phonemize=False)
    ft.save(model, extractor, tok, tmp_path / "best", {"epoch": 1})

    from safetensors.torch import load_file

    weights = load_file(tmp_path / "best" / "model.safetensors")
    assert all(w.dtype == torch.float16 for w in weights.values() if w.is_floating_point())
    loaded = Wav2Vec2ForCTC.from_pretrained(tmp_path / "best")
    assert next(loaded.parameters()).dtype == torch.float32
    for (name, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-2), name
    assert (tmp_path / "best" / "training_info.json").exists() and (tmp_path / "best" / "vocab.json").exists()


@pytest.mark.parametrize("best_epoch, kept", [(0, True), (2, False), (3, False), (5, True)])
def test_early_checkpoint_is_kept_only_if_a_later_epoch_won(tmp_path, best_epoch, kept):
    (tmp_path / "epoch3").mkdir()
    assert ft.keep_early_checkpoint(tmp_path, best_epoch, early_epoch=3) is kept
    assert (tmp_path / "epoch3").exists() is kept


def test_training_refuses_to_run_without_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["finetune_recognizer.py", "--data", str(tmp_path), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="No GPU"):
        ft.main()


def test_versions_are_recorded():
    v = ft.versions()
    assert {"python", "torch", "transformers", "numpy", "soundfile", "gpus_visible"} <= set(v)
