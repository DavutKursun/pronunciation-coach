import json

import numpy as np
import pytest
from conftest import BLANK, TOKEN_TO_ID, fake_recognition

from pronunciation.cache import cached_log_probs, model_slug
from pronunciation.metrics import paired_bootstrap_diff
from pronunciation.systems import System, load_system


class FakeRecognizer:
    """Counts model runs; log-probabilities are just the audio length in frames."""
    runs = 0

    def __init__(self, model_id):
        self.token_to_id, self.blank_id = TOKEN_TO_ID, BLANK
        self.decoder = type("D", (), {"special_ids": {1, 2, 3}})()

    def log_probs(self, audio):
        FakeRecognizer.runs += 1
        return np.full((len(audio) // 320, len(TOKEN_TO_ID)), -1.0)


def test_cache_runs_the_model_only_for_new_items(tmp_path):
    FakeRecognizer.runs = 0
    audio = {"a": lambda: np.zeros(16000), "b": lambda: np.zeros(8000)}
    decoder, log_probs, seconds = cached_log_probs("org/model", "demo", audio, tmp_path, FakeRecognizer)
    assert FakeRecognizer.runs == 2
    assert log_probs["a"].shape == (50, len(TOKEN_TO_ID)) and seconds["b"] == 0.5
    assert decoder.blank_id == BLANK

    audio["c"] = lambda: np.zeros(3200)
    _, log_probs, _ = cached_log_probs("org/model", "demo", audio, tmp_path, FakeRecognizer)
    assert FakeRecognizer.runs == 3                       # only "c" was new
    assert set(log_probs) == {"a", "b", "c"}
    assert (tmp_path / model_slug("org/model") / "demo.npz").exists()

    cached_log_probs("org/other-model", "demo", {"a": audio["a"]}, tmp_path, FakeRecognizer)
    assert FakeRecognizer.runs == 4                       # another model has its own cache


def test_system_from_json(tmp_path):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({"name": "v1", "decision": "rules", "settings": {"gop_threshold": -2.5}}))
    system = load_system(path)
    assert system.name == "v1" and system.settings == {"gop_threshold": -2.5}
    assert system.recognizer == "facebook/wav2vec2-lv-60-espeak-cv-ft"


def test_rules_system_uses_its_settings():
    think = [["θ", "ɪ", "ŋ", "k"]]
    recognition = fake_recognition(["t", "ɪ", "ŋ", "k"], confidence=0.999)
    strict = System("strict", settings={"gop_threshold": -2.5})
    off = System("off", settings={"gop_threshold": None})
    assert strict.assess("think", think, recognition, TOKEN_TO_ID, BLANK).words[0].issues
    assert off.assess("think", think, recognition, TOKEN_TO_ID, BLANK).words[0].issues
    with pytest.raises(ValueError):
        System("unknown", decision="magic").assess("think", think, recognition, TOKEN_TO_ID, BLANK)


def share(units):
    return sum(u[0] for u in units) / sum(u[1] for u in units)


def test_paired_bootstrap_diff_of_identical_systems_is_zero():
    units = [(3, 10), (5, 10), (1, 10)]
    result = paired_bootstrap_diff(units, units, share)
    assert result["diff"] == 0.0 and result["ci95"] == (0.0, 0.0) and not result["real"]


def test_paired_bootstrap_diff_finds_a_consistent_improvement():
    a = [(3, 10), (5, 10), (1, 10), (4, 10), (2, 10)]
    b = [(5, 10), (7, 10), (3, 10), (6, 10), (4, 10)]    # every speaker +2 of 10
    result = paired_bootstrap_diff(a, b, share, n_resamples=500)
    assert result["a"] == pytest.approx(0.3) and result["b"] == pytest.approx(0.5)
    assert result["diff"] == pytest.approx(0.2)
    assert result["ci95"][0] > 0 and result["real"]


def test_paired_bootstrap_diff_is_not_real_when_speakers_disagree():
    a = [(5, 10), (5, 10), (5, 10), (5, 10)]
    b = [(9, 10), (1, 10), (8, 10), (2, 10)]             # much better for two, much worse for two
    result = paired_bootstrap_diff(a, b, share, n_resamples=500)
    low, high = result["ci95"]
    assert low < 0 < high and not result["real"]


def test_paired_bootstrap_diff_needs_the_same_speakers():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([(1, 2)], [(1, 2), (1, 2)], share)


def test_compare_experiments_pairs_speakers(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import compare_experiments

    # per speaker [tp, fp, fn, tn]; B finds 2 more wrong words for every Turkish speaker
    a = {"name": "a", "units": {"saa_dev_turkish": {"t1": [3, 1, 7, 9], "t2": [4, 1, 6, 9], "t3": [2, 0, 8, 10]}}}
    b = {"name": "b", "units": {"saa_dev_turkish": {"t3": [4, 0, 6, 10], "t1": [5, 1, 5, 9], "t2": [6, 1, 4, 9]}}}
    rows = compare_experiments.compare(a, b, n_resamples=500)
    recall = next(r for r in rows if r["metric"] == "SAA dev, Turkish: recall")
    assert recall["a"] == pytest.approx(9 / 30) and recall["b"] == pytest.approx(15 / 30)
    assert recall["real"] and recall["ci95"][0] > 0
    assert all(r["metric"].startswith("SAA dev, Turkish") for r in rows)   # metrics without units are skipped
