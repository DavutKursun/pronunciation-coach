import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pronunciation.assess import FEATURE_NAMES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def fake_features(n: int, rng) -> pd.DataFrame:
    table = pd.DataFrame(rng.random((n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    table["accuracy"] = (table["phone_accuracy"] * 10).round()
    table["speaker"] = np.arange(n) % 6
    return table


def test_metrics_json_has_word_detection(tmp_path, monkeypatch):
    import train_scorer

    rng = np.random.default_rng(0)
    fake_features(30, rng).to_csv(tmp_path / "train.csv", index=False)
    fake_features(12, rng).to_csv(tmp_path / "test.csv", index=False)
    # 6 words: the experts found 3 wrong (score < 10); we flagged 2 of them and 1 correct word
    pd.DataFrame({"human_accuracy": [10, 10, 10, 8, 5, 9], "our_score": [1, 1, 0.5, 0.5, 0.2, 1],
                  "n_issues": [0, 0, 1, 1, 2, 0]}).to_csv(tmp_path / "test_words.csv", index=False)

    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", ["train_scorer.py", "--features-dir", str(tmp_path),
                                      "--out", str(tmp_path / "scorer.joblib"), "--metrics", str(metrics_path)])
    assert train_scorer.main() == 0

    words = json.loads(metrics_path.read_text())["word_detection"]
    assert words["expert_threshold"] == 10
    assert words["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert words["precision"] == pytest.approx(2 / 3) and words["recall"] == pytest.approx(2 / 3)
    assert -1 <= words["pcc"] <= 1


def test_train_scorer_keeps_other_results(tmp_path, monkeypatch):
    import train_scorer

    rng = np.random.default_rng(1)
    fake_features(30, rng).to_csv(tmp_path / "train.csv", index=False)
    fake_features(12, rng).to_csv(tmp_path / "test.csv", index=False)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"speech_accent_archive": {"test": "kept"}}))
    monkeypatch.setattr(sys, "argv", ["train_scorer.py", "--features-dir", str(tmp_path),
                                      "--out", str(tmp_path / "scorer.joblib"), "--metrics", str(metrics_path)])
    assert train_scorer.main() == 0
    metrics = json.loads(metrics_path.read_text())
    assert metrics["speech_accent_archive"] == {"test": "kept"} and "models" in metrics
