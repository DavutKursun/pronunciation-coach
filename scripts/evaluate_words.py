"""Word-level analysis on speechocean762: do we flag the words the experts flagged?

Five experts score every word 0-10 (speechocean762 README): 10 = the pronunciation is perfect,
7-9 = most phones are correct but have accents, 4-6 = less than 30% of the phones are wrong,
2-3 = more than 30% are wrong. A word is "really wrong" when its score is below 10 (not perfect);
"below 7" (some phone actually wrong) is shown as extra information. We "flagged" a word when our
feedback reports at least one error for it after GOP confirmation. Only utterances where our word
list lines up with the experts' are used.

Reports the word score correlation, the confusion matrix, false alarm rate, recall (catch rate),
precision, F1 and the phoneme pairs behind most false alarms. The calculations live in
pronunciation/metrics.py and are shared with evaluate_saa.py and train_scorer.py.

Uses the "val" speakers of the train split (data/speechocean_split.json, made by
split_speechocean.py): "fit" is training data for v2, the official test split stays locked.
Model output is cached per model (pronunciation/cache.py), so a rule change re-runs in seconds.
(v1's -2.5 GOP threshold was chosen on the first 400 train utterances, 20 speakers, before this
split existed; some of them are val speakers.)

Usage:
    python scripts/evaluate_words.py                 # val speakers
    python scripts/evaluate_words.py --sweep         # compare GOP confirmation thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation.audio import load_audio  # noqa: E402
from pronunciation.cache import cached_log_probs  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.metrics import (SPEECHOCEAN_WRONG_BELOW, format_confusion, format_detection, pearson,  # noqa: E402
                                   top_pairs, word_detection_metrics)
from pronunciation.recognizer import DEFAULT_MODEL, Decoder  # noqa: E402
from pronunciation.systems import System  # noqa: E402

DATASET = "mispeech/speechocean762"
SPLIT_FILE = ROOT / "data" / "speechocean_split.json"
SWEEP = [None, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0, -5.0, -7.0]
V1 = System("v1")   # rules with the default GOP thresholds


def load(part: str = "val", model_id: str = DEFAULT_MODEL):
    """Decoder, and (utterance info, log-probabilities, seconds) for the utterances of one part of train."""
    from datasets import Audio, load_dataset

    speakers = set(json.loads(SPLIT_FILE.read_text())[part])
    data = load_dataset(DATASET, split="train").cast_column("audio", Audio(decode=False))
    indices = [i for i, s in enumerate(data["speaker"]) if s in speakers]

    def audio_of(i: int):
        field = data[i]["audio"]
        return load_audio(field["bytes"] if field.get("bytes") else field["path"])

    audio = {f"u{i}": (lambda i=i: audio_of(i)) for i in indices}
    decoder, log_probs, seconds = cached_log_probs(model_id, f"speechocean_train_{part}", audio)
    info = data.select(indices).select_columns(["text", "speaker", "words"])
    utterances = [({"text": row["text"], "speaker": row["speaker"],
                    "words": [{"text": w["text"], "accuracy": w["accuracy"], "phones": w["phones"],
                               "phones-accuracy": w["phones-accuracy"]} for w in row["words"]]},
                   log_probs[f"u{i}"], seconds[f"u{i}"]) for i, row in zip(indices, info)]
    return decoder, utterances


def evaluate(decoder: Decoder, utterances, system: System = V1) -> list[dict]:
    """One row per comparable word: speaker, expert score, our score, flagged or not, our error pairs."""
    rows = []
    for utterance, log_probs, seconds in utterances:
        text = utterance["text"]
        result = system.assess(text, phonemize_words(tokenize(text)), decoder(log_probs, seconds),
                               decoder.token_to_id, decoder.blank_id)
        if len(result.words) != len(utterance["words"]):
            continue
        for human, ours in zip(utterance["words"], result.words):
            rows.append({"speaker": utterance["speaker"], "accuracy": human["accuracy"], "our_score": ours.score,
                         "flagged": bool(ours.issues),
                         "pairs": [f"{i.expected or '-'} → {i.heard or '-'}" for i in ours.issues]})
    return rows


def confusion_units(rows: list[dict], expert_threshold: int = SPEECHOCEAN_WRONG_BELOW) -> dict[str, list[int]]:
    """Per speaker: [tp, fp, fn, tn], the unit for bootstrap intervals and system comparisons."""
    units: dict[str, list[int]] = {}
    for r in rows:
        wrong = r["accuracy"] < expert_threshold
        cell = (0 if r["flagged"] else 2) + (0 if wrong else 1)   # tp, fp, fn, tn
        units.setdefault(r["speaker"], [0, 0, 0, 0])[cell] += 1
    return units


def detection(rows: list[dict], expert_threshold: int = SPEECHOCEAN_WRONG_BELOW) -> dict:
    return word_detection_metrics([r["accuracy"] < expert_threshold for r in rows], [r["flagged"] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true", help="compare GOP confirmation thresholds")
    args = parser.parse_args()

    decoder, utterances = load("val")
    if args.sweep:
        print(f"{len(utterances)} val utterances. An error is reported only if the word's GOP is below the threshold.")
        for threshold in SWEEP:
            label = "GOP off" if threshold is None else f"GOP {threshold:.1f}"
            system = System(label, settings={"gop_threshold": threshold})
            print(format_detection(label, detection(evaluate(decoder, utterances, system))))
        return

    rows = evaluate(decoder, utterances)
    print(f"{len(utterances)} val utterances ({len({r['speaker'] for r in rows})} speakers)\n")
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
