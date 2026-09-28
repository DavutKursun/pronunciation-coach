"""Split the speechocean762 train speakers into "fit" (80%) and "val" (20%).

v2 models are trained on "fit"; "val" is for model selection and evaluation during development.
The official test split stays locked until the final comparison. The split is by speaker (all 20
recordings of a speaker land on the same side), stratified by gender and child/adult (half of
the speakers are children), with a fixed seed.

Usage:
    python scripts/split_speechocean.py      # writes data/speechocean_split.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "speechocean_split.json"
SEED = 762
VAL_SHARE = 0.2
CHILD_MAX_AGE = 12


def split(speakers: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    """speakers: one row per speaker with "speaker", "gender" and "age" columns."""
    rng = random.Random(seed)
    halves = {"fit": [], "val": []}
    speakers = speakers.assign(child=speakers.age <= CHILD_MAX_AGE)
    for _, group in speakers.groupby(["gender", "child"], sort=True):
        names = sorted(group.speaker)
        rng.shuffle(names)
        n_val = round(len(names) * VAL_SHARE)
        halves["val"] += names[:n_val]
        halves["fit"] += names[n_val:]
    return {h: sorted(names) for h, names in halves.items()}


def main() -> None:
    from datasets import load_dataset

    data = load_dataset("mispeech/speechocean762", split="train")
    table = pd.DataFrame({"speaker": data["speaker"], "gender": data["gender"], "age": data["age"]})
    speakers = table.groupby("speaker", as_index=False).first()
    halves = split(speakers)
    SPLIT_FILE.write_text(json.dumps({"seed": SEED, **halves}, indent=2) + "\n")

    speakers["part"] = speakers.speaker.map({s: h for h, names in halves.items() for s in names})
    speakers["child"] = speakers.age <= CHILD_MAX_AGE
    summary = speakers.groupby("part").agg(speakers=("speaker", "size"), female=("gender", lambda g: (g == "f").sum()),
                                           children=("child", "sum"), mean_age=("age", "mean"))
    print(summary.round(1).to_string())
    print(f"saved {SPLIT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
