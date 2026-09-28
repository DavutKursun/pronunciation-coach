"""Word-level analysis on speechocean762 train: do we flag the words the experts flagged?

Five experts score every word 0-10 (speechocean762 README): 10 = the pronunciation is perfect,
7-9 = most phones are correct but have accents, 4-6 = less than 30% of the phones are wrong,
2-3 = more than 30% are wrong. A word is "really wrong" when its score is below 10 (not perfect);
"below 7" (some phone actually wrong) is shown as extra information. We "flagged" a word when our
feedback reports at least one error for it after GOP confirmation. Only utterances where our word
list lines up with the experts' are used.

Reports the word score correlation, the confusion matrix, false alarm rate, recall (catch rate),
precision, F1 and the phoneme pairs behind most false alarms. The calculations live in
pronunciation/metrics.py and are shared with evaluate_saa.py and train_scorer.py.

The model runs once; its log-probabilities are cached in data/cache/, so a rule change can be
re-evaluated in seconds. Only the TRAIN split is used: the test split is kept for the final numbers.

Usage:
    python scripts/evaluate_words.py                 # first 400 train utterances
    python scripts/evaluate_words.py --limit 50
    python scripts/evaluate_words.py --sweep         # compare GOP confirmation thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation import analyze  # noqa: E402
from pronunciation.audio import SAMPLE_RATE, load_audio  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.metrics import (SPEECHOCEAN_WRONG_BELOW, format_confusion, format_detection, pearson,  # noqa: E402
                                   top_pairs, word_detection_metrics)
from pronunciation.recognizer import Decoder  # noqa: E402

DATASET = "mispeech/speechocean762"
SWEEP = [None, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0, -7.0]
CACHE_DIR = ROOT / "data" / "cache"


def cache_paths(limit: int) -> tuple[Path, Path]:
    stem = CACHE_DIR / f"train_{limit}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def build_cache(limit: int) -> None:
    from datasets import Audio, load_dataset

    from pronunciation.recognizer import PhonemeRecognizer

    data = load_dataset(DATASET, split="train").cast_column("audio", Audio(decode=False))
    data = data.select(range(min(limit, len(data))))
    recognizer = PhonemeRecognizer()
    arrays, utterances = {}, []
    for i, example in enumerate(data):
        audio_field = example["audio"]
        audio = load_audio(audio_field["bytes"] if audio_field.get("bytes") else audio_field["path"])
        arrays[f"u{i}"] = recognizer.log_probs(audio)
        utterances.append({
            "text": example["text"], "speaker": example.get("speaker"), "seconds": len(audio) / SAMPLE_RATE,
            "words": [{"text": w["text"], "accuracy": w["accuracy"]} for w in example["words"]],
        })
        if (i + 1) % 100 == 0:
            print(f"cached {i + 1}/{len(data)}")

    npz_path, json_path = cache_paths(limit)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps({
        "vocab": recognizer.token_to_id, "blank_id": recognizer.blank_id,
        "special_ids": sorted(recognizer.decoder.special_ids), "utterances": utterances,
    }))


def load_cache(limit: int):
    npz_path, json_path = cache_paths(limit)
    if not npz_path.exists():
        print(f"Running the model on the first {limit} train utterances (only once)...")
        build_cache(limit)
    meta = json.loads(json_path.read_text())
    arrays = np.load(npz_path)
    utterances = [(u, arrays[f"u{i}"]) for i, u in enumerate(meta["utterances"])]
    return meta["vocab"], meta["blank_id"], set(meta["special_ids"]), utterances


def evaluate(decoder: Decoder, utterances, **analyze_options) -> list[dict]:
    """One row per comparable word: expert score, our score, flagged or not, our error pairs."""
    rows = []
    for utterance, log_probs in utterances:
        text = utterance["text"]
        recognition = decoder(log_probs, utterance["seconds"])
        result = analyze(text, phonemize_words(tokenize(text)), recognition,
                         decoder.token_to_id, decoder.blank_id, **analyze_options)
        if len(result.words) != len(utterance["words"]):
            continue
        for human, ours in zip(utterance["words"], result.words):
            rows.append({"accuracy": human["accuracy"], "our_score": ours.score, "flagged": bool(ours.issues),
                         "pairs": [f"{i.expected or '-'} → {i.heard or '-'}" for i in ours.issues]})
    return rows


def detection(rows: list[dict], expert_threshold: int = SPEECHOCEAN_WRONG_BELOW) -> dict:
    return word_detection_metrics([r["accuracy"] < expert_threshold for r in rows], [r["flagged"] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=400, help="first N train utterances")
    parser.add_argument("--sweep", action="store_true", help="compare GOP confirmation thresholds")
    args = parser.parse_args()

    vocab, blank_id, special_ids, utterances = load_cache(args.limit)
    decoder = Decoder(vocab, blank_id, special_ids)
    if args.sweep:
        print(f"{len(utterances)} train utterances. An error is reported only if the word's GOP is below the threshold.")
        for threshold in SWEEP:
            label = "GOP off" if threshold is None else f"GOP {threshold:.1f}"
            print(format_detection(label, detection(evaluate(decoder, utterances, gop_threshold=threshold))))
        return

    rows = evaluate(decoder, utterances)
    print(f"{len(utterances)} train utterances\n")
    main_metrics = detection(rows)
    print(format_detection("expert < 10", main_metrics))
    print(format_detection("expert < 7", detection(rows, 7)) + "   (extra: some phone actually wrong)")
    print(f"\nConfusion matrix (expert < 10):\n{format_confusion(main_metrics)}")
    print(f"\nPearson correlation of our word score with the expert score: "
          f"{pearson([r['our_score'] for r in rows], [r['accuracy'] for r in rows]):.3f}")

    print("\nTop false-alarm pairs (expected → heard) in words the experts scored 10:")
    for pair, count in top_pairs(r["pairs"] for r in rows if r["accuracy"] == 10):
        print(f"  {count:4}  {pair}")


if __name__ == "__main__":
    main()
