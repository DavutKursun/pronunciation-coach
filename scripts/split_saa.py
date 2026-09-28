"""Split the Speech Accent Archive speakers into a dev half and a test half.

Only speakers with a text transcription are used (the others cannot be evaluated). Within each
group (Turkish, English) and gender, speakers are sorted by age and taken in pairs; a seeded coin
flip sends one of each pair to dev and the other to test. Both halves then have a similar age
and gender mix. The split only looks at age and gender, never at results.

The test half is not looked at until the final evaluation (roadmap step 8b).

Usage:
    python scripts/split_saa.py          # writes data/saa_split.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "saa_split.json"
SEED = 762


def split(speakers: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    rng = random.Random(seed)
    halves = {"dev": [], "test": []}
    for _, group in speakers.groupby(["group", "gender"], sort=True):
        ordered = group.sort_values(["age", "speaker"]).speaker.tolist()
        for k in range(0, len(ordered), 2):
            pair = ordered[k:k + 2]
            rng.shuffle(pair)
            halves["dev"].append(pair[0])
            if len(pair) == 2:
                halves["test"].append(pair[1])
            else:  # odd one out: give it to the smaller half
                smaller = min(halves, key=lambda h: len(halves[h]))
                halves[smaller].append(halves["dev"].pop())
    return {h: sorted(names) for h, names in halves.items()}


def main() -> None:
    speakers = pd.read_csv(ROOT / "data" / "saa" / "speakers.csv")
    speakers = speakers[speakers.has_transcription]
    halves = split(speakers)
    SPLIT_FILE.write_text(json.dumps({"seed": SEED, **halves}, indent=2) + "\n")

    speakers["half"] = speakers.speaker.map({s: h for h, names in halves.items() for s in names})
    summary = speakers.groupby(["group", "half"]).agg(
        speakers=("speaker", "size"), female=("gender", lambda g: (g == "female").sum()),
        mean_age=("age", "mean"), min_age=("age", "min"), max_age=("age", "max"))
    print(summary.round(1).to_string())
    print(f"saved {SPLIT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
