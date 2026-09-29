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


def test_cache_of_a_local_model_is_rebuilt_when_the_model_changes(tmp_path):
    FakeRecognizer.runs = 0
    model = tmp_path / "recognizer"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"first training")
    audio = {"a": lambda: np.zeros(16000)}
    cached_log_probs(str(model), "demo", audio, tmp_path / "cache", FakeRecognizer)
    cached_log_probs(str(model), "demo", audio, tmp_path / "cache", FakeRecognizer)
    assert FakeRecognizer.runs == 1                       # same model: cached
    (model / "model.safetensors").write_bytes(b"retrained in the same folder")
    cached_log_probs(str(model), "demo", audio, tmp_path / "cache", FakeRecognizer)
    assert FakeRecognizer.runs == 2                       # a new model behind the same path: run again


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


def test_v2_3_experiments_change_only_the_recognizer():
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "experiments"
    v1 = load_system(folder / "v1.json")
    for name in ("v2-3-best", "v2-3-epoch3"):
        system = load_system(folder / f"{name}.json")
        assert system.name == name and system.decision == v1.decision == "rules"
        assert system.settings == v1.settings == {"gop_threshold": -2.5, "pattern_threshold": -1.0}
        assert system.recognizer == f"models/recognizer-l2arctic/{name.removeprefix('v2-3-')}"


def test_tuned_experiment_uses_the_thresholds_chosen_on_saa_dev():
    import math
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    system = load_system(root / "experiments" / "v2-3-best-tuned.json")
    # Infinity: typical Turkish-speaker errors are always reported (a GOP is never above 0)
    assert system.settings == {"gop_threshold": -1.0, "pattern_threshold": math.inf}
    chosen = json.loads((root / "results" / "thresholds" / "v2-3-best.json").read_text())["selected"]["thresholds"]
    assert (system.settings["gop_threshold"], system.settings["pattern_threshold"]) == tuple(chosen)


def test_frozen_v2_is_the_system_chosen_on_saa_dev():
    import math
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    v2 = load_system(root / "experiments" / "v2.json")
    assert v2.recognizer == "models/recognizer-l2arctic/best" and v2.decision == "rules"
    assert v2.settings == {"gop_threshold": 0.0, "pattern_threshold": math.inf, "w_margin_threshold": -4.0}
    chosen = json.loads((root / "results" / "thresholds" / "v2.json").read_text())["selected"]["thresholds"]
    assert [v2.settings[k] for k in ("gop_threshold", "pattern_threshold", "w_margin_threshold")] == chosen


def test_generalized_systems_use_the_thresholds_chosen_on_saa_dev():
    from pathlib import Path

    from pronunciation.thresholds import HIDDEN_PATTERNS

    root = Path(__file__).resolve().parents[1]
    for name, recognizer in (("v1-gen", load_system(root / "experiments" / "v1.json").recognizer),
                             ("v2-gen", load_system(root / "experiments" / "v2.json").recognizer)):
        system = load_system(root / "experiments" / f"{name}.json")
        chosen = json.loads((root / "results" / "hidden" / f"{name}.json").read_text())["settings"]
        assert system.decision == "rules" and system.recognizer == recognizer
        assert system.settings == chosen and list(chosen["hidden_thresholds"]) == HIDDEN_PATTERNS


def test_v2_3d_kept_the_frozen_v2():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    decision = json.loads((root / "results" / "hidden" / "decision.json").read_text())
    assert (decision["frozen"], decision["generalized"], decision["replace"]) == ("v2", "v2-gen", False)
    assert "hidden_thresholds" not in load_system(root / "experiments" / "v2.json").settings
