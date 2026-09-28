"""Word-level detection metrics on speechocean762 train: do we flag the words the experts flagged?

Five experts score every word 0-10. We call a word "flagged" when our feedback reports at least
one issue for it. Only utterances where our word list lines up with the experts' are used.

  false alarm rate  flagged words among the words the experts scored 10
  catch rate        flagged words among the words the experts scored below 10 (recall)
  precision         share of flagged words that the experts scored below 10
  F1                harmonic mean of precision and catch rate

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
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation import analyze  # noqa: E402
from pronunciation.audio import SAMPLE_RATE, load_audio  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
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


def evaluate(decoder: Decoder, utterances, **analyze_options) -> list[tuple[int, bool, list[str]]]:
    """One (expert accuracy, flagged, [expected→heard pairs]) row per comparable word."""
    rows = []
    for utterance, log_probs in utterances:
        text = utterance["text"]
        recognition = decoder(log_probs, utterance["seconds"])
        result = analyze(text, phonemize_words(tokenize(text)), recognition,
                         decoder.token_to_id, decoder.blank_id, **analyze_options)
        if len(result.words) != len(utterance["words"]):
            continue
        for human, ours in zip(utterance["words"], result.words):
            pairs = [f"{i.expected or '-'} → {i.heard or '-'}" for i in ours.issues]
            rows.append((human["accuracy"], bool(ours.issues), pairs))
    return rows


def summarize(rows) -> dict[str, float]:
    ok = [flagged for accuracy, flagged, _ in rows if accuracy == 10]
    bad = [flagged for accuracy, flagged, _ in rows if accuracy < 10]
    false_alarm = sum(ok) / len(ok) if ok else 0.0
    catch = sum(bad) / len(bad) if bad else 0.0
    n_flagged = sum(ok) + sum(bad)
    precision = sum(bad) / n_flagged if n_flagged else 0.0
    f1 = 2 * precision * catch / (precision + catch) if precision + catch else 0.0
    return {"words": len(rows), "expert_ok": len(ok), "expert_bad": len(bad),
            "false_alarm": false_alarm, "catch": catch, "precision": precision, "f1": f1}


def print_summary(stats: dict[str, float]) -> None:
    print(f"words {stats['words']} (expert = 10: {stats['expert_ok']}, expert < 10: {stats['expert_bad']})")
    print(f"false alarm {stats['false_alarm']:.1%}   catch {stats['catch']:.1%}   "
          f"precision {stats['precision']:.1%}   F1 {stats['f1']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=400, help="first N train utterances")
    parser.add_argument("--sweep", action="store_true", help="compare GOP confirmation thresholds")
    args = parser.parse_args()

    vocab, blank_id, special_ids, utterances = load_cache(args.limit)
    decoder = Decoder(vocab, blank_id, special_ids)
    if args.sweep:
        print(f"{len(utterances)} train utterances. An error is reported only if the word's GOP is below the threshold.")
        print(f"{'threshold':>9}  {'false alarm':>11}  {'catch':>6}  {'precision':>9}  {'F1':>5}")
        for threshold in SWEEP:
            stats = summarize(evaluate(decoder, utterances, gop_threshold=threshold))
            label = "off" if threshold is None else f"{threshold:.1f}"
            print(f"{label:>9}  {stats['false_alarm']:>11.1%}  {stats['catch']:>6.1%}  "
                  f"{stats['precision']:>9.1%}  {stats['f1']:>5.3f}")
        return

    rows = evaluate(decoder, utterances)
    print(f"{len(utterances)} train utterances")
    print_summary(summarize(rows))

    false_alarm_pairs = Counter(p for accuracy, _, pairs in rows if accuracy == 10 for p in pairs)
    print("\nTop false-alarm pairs (expected → heard) in words the experts scored 10:")
    for pair, count in false_alarm_pairs.most_common(15):
        print(f"  {count:4}  {pair}")


if __name__ == "__main__":
    main()
