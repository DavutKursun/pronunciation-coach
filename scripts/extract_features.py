"""Run the pronunciation pipeline on speechocean762 and save features + expert scores.

speechocean762: 5,000 English utterances by non-native speakers (Mandarin L1, half of them
children), each scored by five experts at sentence, word and phoneme level.
The output CSVs are the training data for scripts/train_scorer.py.

Usage (GPU recommended, e.g. Google Colab; ~10 minutes on a T4):
    python scripts/extract_features.py
    python scripts/extract_features.py --limit 50          # quick test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pronunciation import PronunciationCoach  # noqa: E402
from pronunciation.audio import load_audio  # noqa: E402
from pronunciation.recognizer import DEFAULT_MODEL, PhonemeRecognizer  # noqa: E402

HUMAN_SCORES = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def load_splits(args):
    from datasets import Audio, load_dataset, load_from_disk

    ds = load_from_disk(args.local_path) if args.local_path else load_dataset(args.dataset)
    # decode audio ourselves (soundfile) instead of relying on the datasets audio backend
    return {split: ds[split].cast_column("audio", Audio(decode=False)) for split in args.splits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="mispeech/speechocean762")
    parser.add_argument("--local-path", help="a dataset saved with save_to_disk (for tests)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--limit", type=int, help="only the first N utterances of each split")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    args = parser.parse_args()

    coach = PronunciationCoach(PhonemeRecognizer(args.model), scorer_path=None)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, data in load_splits(args).items():
        if args.limit:
            data = data.select(range(min(args.limit, len(data))))
        rows, word_rows = [], []
        start = time.time()
        for i, example in enumerate(data):
            audio_field = example["audio"]
            audio = load_audio(audio_field["bytes"] if audio_field.get("bytes") else audio_field["path"])
            result = coach.assess(audio, example["text"])

            rows.append({"split": split, "idx": i, "speaker": example.get("speaker"), "text": example["text"],
                         **{s: example[s] for s in HUMAN_SCORES}, **result.features})

            # word-level comparison with the experts (only when the word lists line up)
            human_words = example.get("words") or []
            if len(human_words) == len(result.words):
                for w, (human, ours) in enumerate(zip(human_words, result.words)):
                    word_rows.append({"split": split, "idx": i, "word_idx": w, "text": human["text"],
                                      "human_accuracy": human["accuracy"], "our_score": ours.score,
                                      "n_issues": len(ours.issues)})

            if (i + 1) % 100 == 0 or i + 1 == len(data):
                rate = (i + 1) / (time.time() - start)
                print(f"[{split}] {i + 1}/{len(data)}  ({rate:.1f} utt/s, {(len(data) - i - 1) / rate / 60:.1f} min left)")

        pd.DataFrame(rows).to_csv(args.out_dir / f"{split}.csv", index=False)
        pd.DataFrame(word_rows).to_csv(args.out_dir / f"{split}_words.csv", index=False)
        print(f"saved {len(rows)} utterances and {len(word_rows)} words -> {args.out_dir}/{split}*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
