"""A "system" = which recognizer, which decision mechanism, which settings.

Experiments (scripts/run_experiment.py) compare systems on the same data. A system is described
by a small JSON file in experiments/, e.g. experiments/v1.json:

    {"name": "v1", "recognizer": "facebook/wav2vec2-lv-60-espeak-cv-ft",
     "decision": "rules", "settings": {"gop_threshold": -2.5, "pattern_threshold": -1.0}}

Decision mechanisms:
  rules   v1: alignment differences, confirmed with GOP (settings are passed to analyze())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .assess import Assessment, analyze
from .recognizer import DEFAULT_MODEL


@dataclass
class System:
    name: str
    recognizer: str = DEFAULT_MODEL
    decision: str = "rules"
    settings: dict = field(default_factory=dict)

    def assess(self, text: str, raw_word_phones: list[list[str]], recognition,
               token_to_id: dict[str, int], blank_id: int) -> Assessment:
        """Feedback for one recording, from the recognizer's output."""
        if self.decision == "rules":
            return analyze(text, raw_word_phones, recognition, token_to_id, blank_id, **self.settings)
        raise ValueError(f"unknown decision mechanism: {self.decision}")


def load_system(path: str | Path) -> System:
    return System(**json.loads(Path(path).read_text()))
