"""Word-level evaluation on the Speech Accent Archive: real Turkish speakers, expert transcriptions.

A paragraph word is "really wrong" when the expert transcription differs from the expected
(eSpeak en-us) phonemes outside our accepted variants (see pronunciation/saa.py). We compare that
with the words our feedback flags:

  false alarm   flagged words among the words the expert found correct
  recall        flagged words among the words the expert found wrong (catch rate)
  precision     share of flagged words that the expert found wrong
  error level   every expert error on its own: did we report an error on the same sound?
                ("things" said "tins" has two: θ → t and z → s)
  per pattern   word- and error-level catch rates for each Turkish-speaker pattern

The calculations are shared with evaluate_words.py (pronunciation/metrics.py).

The model runs once per half; its output is cached in data/cache/. Only the dev half is used while
we change rules. The test half is for the final evaluation (roadmap step 8b) and needs --final.

Usage:
    python scripts/download_saa.py && python scripts/split_saa.py    # once
    python scripts/evaluate_saa.py                                   # dev half
    python scripts/evaluate_saa.py --gop-sweep                       # compare GOP thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation import analyze  # noqa: E402
from pronunciation.assess import prepare_expected  # noqa: E402
from pronunciation.audio import SAMPLE_RATE, load_audio  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.metrics import error_level_catch, format_confusion, format_detection, top_pairs, word_detection_metrics  # noqa: E402
from pronunciation.phonemes import FUNCTION_WORDS, TIPS  # noqa: E402
from pronunciation.recognizer import Decoder  # noqa: E402
from pronunciation.saa import PARAGRAPH, align_words, expert_labels, narrow_to_broad, parse_transcription  # noqa: E402

SAA_DIR = ROOT / "data" / "saa"
SPLIT_FILE = ROOT / "data" / "saa_split.json"
CACHE_DIR = ROOT / "data" / "cache"


def load(half: str):
    """Decoder, and (speaker, log-probabilities, seconds, transcription) for every speaker of the half."""
    speakers = json.loads(SPLIT_FILE.read_text())[half]
    npz_path, meta_path = CACHE_DIR / f"saa_{half}.npz", CACHE_DIR / f"saa_{half}.json"
    if not npz_path.exists():
        from pronunciation.recognizer import PhonemeRecognizer

        print(f"Running the model on the {half} speakers (only once)...")
        recognizer = PhonemeRecognizer()
        arrays, seconds = {}, {}
        for speaker in speakers:
            audio = load_audio(SAA_DIR / f"{speaker}.mp3")
            arrays[speaker] = recognizer.log_probs(audio)
            seconds[speaker] = len(audio) / SAMPLE_RATE
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        meta_path.write_text(json.dumps({"vocab": recognizer.token_to_id, "blank_id": recognizer.blank_id,
                                         "special_ids": sorted(recognizer.decoder.special_ids), "seconds": seconds}))
    meta = json.loads(meta_path.read_text())
    arrays = np.load(npz_path)
    decoder = Decoder(meta["vocab"], meta["blank_id"], set(meta["special_ids"]))
    data = [(s, arrays[s], meta["seconds"][s], (SAA_DIR / f"{s}.txt").read_text(encoding="utf-8")) for s in speakers]
    return decoder, data


def evaluate(decoder: Decoder, data, **analyze_options) -> pd.DataFrame:
    """One row per paragraph word per speaker: what the expert heard and what we reported."""
    words = tokenize(PARAGRAPH)
    raw = phonemize_words(words)
    word_phones = prepare_expected(words, raw)[0]
    word_is_function = [w.lower() in FUNCTION_WORDS for w in words]

    rows = []
    for speaker, log_probs, seconds, transcription in data:
        expert_words = [narrow_to_broad(w) for w in parse_transcription(transcription)]
        expert = align_words(word_phones, expert_words, word_is_function)
        labels = expert_labels(words, word_phones, expert)
        result = analyze(PARAGRAPH, raw, decoder(log_probs, seconds), decoder.token_to_id, decoder.blank_id,
                         **analyze_options)
        for k, (label, ours) in enumerate(zip(labels, result.words)):
            rows.append({
                "speaker": speaker, "group": speaker.rstrip("0123456789"), "word_idx": k, "word": words[k],
                "expert_wrong": label.wrong, "expert_tips": sorted(label.tips), "expert_pairs": label.pairs,
                "expert_errors": label.errors, "our_errors": ours.issues,
                "flagged": bool(ours.issues), "our_tips": sorted({i.tip for i in ours.issues if i.tip}),
                "our_pairs": [f"{i.expected or '-'} → {i.heard or '-'}" for i in ours.issues],
                "dismissed": bool(ours.dismissed) and not ours.issues, "gop": ours.gop,
            })
    return pd.DataFrame(rows)


def detection(rows: pd.DataFrame) -> dict:
    """Word-level metrics; words the speaker skipped are left out."""
    rows = rows[rows.expert_wrong.notna()]
    return word_detection_metrics(rows.expert_wrong.astype(bool), rows.flagged)


def pattern_table(rows: pd.DataFrame) -> pd.DataFrame:
    """For every Turkish-speaker pattern the expert heard: word level (flagged, right tip) and error level."""
    by_tip = error_level_catch(zip(rows.expert_errors, rows.our_errors))["by_tip"]
    out = []
    for tip in TIPS:
        hit = rows[rows.expert_tips.map(lambda tips: tip in tips)]
        if len(hit):
            out.append({"pattern": tip, "words": len(hit), "word_caught": hit.flagged.mean(),
                        "right_tip": hit.our_tips.map(lambda tips: tip in tips).mean(),
                        "errors": by_tip[tip]["errors"], "error_caught": by_tip[tip]["found"] / by_tip[tip]["errors"],
                        "hidden_by_gop": int(hit.dismissed.sum())})
    return pd.DataFrame(out).set_index("pattern")


def report(rows: pd.DataFrame) -> None:
    for group in ("english", "turkish"):
        print(format_detection(group, detection(rows[rows.group == group])))
    skipped = rows.expert_wrong.isna().sum()
    print(f"(paragraph words the speaker skipped or could not be matched: {skipped}, left out)")

    turkish = rows[(rows.group == "turkish") & rows.expert_wrong.notna()]
    print(f"\nTurkish speakers, confusion matrix:\n{format_confusion(detection(turkish))}")

    errors = error_level_catch(zip(turkish.expert_errors, turkish.our_errors))
    print(f"\nTurkish speakers, error level: {errors['expert_errors']} expert errors, "
          f"we found {errors['found']} ({errors['catch']:.1%}), {errors['found_exact']} of them exactly ({errors['exact']:.1%})")
    print(f"words with 2+ expert errors: {errors['multi_error_words']}  -> we found all {errors['multi_all']}, "
          f"some {errors['multi_some']}, none {errors['multi_none']}")

    percent = "{:.0%}".format
    print("\nTurkish speakers, per pattern the expert heard:")
    print(pattern_table(turkish).to_string(formatters={"word_caught": percent, "right_tip": percent, "error_caught": percent}))

    english_ok = rows[(rows.group == "english") & (rows.expert_wrong == False)]  # noqa: E712
    print("\nEnglish speakers: top false-alarm pairs (ours, expert found the word correct):")
    for pair, n in top_pairs(english_ok.our_pairs, 10):
        print(f"  {n:4}  {pair}")

    missed = turkish[turkish.expert_wrong.astype(bool) & ~turkish.flagged]
    print(f"\nTurkish speakers: expert heard it, we did not ({len(missed)} words, {int(missed.dismissed.sum())} hidden by GOP):")
    for pair, n in top_pairs(missed.expert_pairs, 10):
        print(f"  {n:4}  {pair}")
    extra = turkish[~turkish.expert_wrong.astype(bool) & turkish.flagged]
    print(f"\nTurkish speakers: we reported it, the expert did not ({len(extra)} words):")
    for pair, n in top_pairs(extra.our_pairs, 10):
        print(f"  {n:4}  {pair}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true", help="evaluate the held-out test half (step 8b only)")
    parser.add_argument("--gop-sweep", action="store_true", help="compare GOP confirmation thresholds")
    args = parser.parse_args()

    half = "test" if args.final else "dev"
    decoder, data = load(half)
    print(f"Speech Accent Archive, {half} half: {len(data)} speakers\n")
    if args.gop_sweep:
        for threshold in [None, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0]:
            rows = evaluate(decoder, data, gop_threshold=threshold)
            print(format_detection(f"GOP {threshold}", detection(rows[rows.group == "turkish"])))
        return
    rows = evaluate(decoder, data)
    report(rows)
    rows.drop(columns=["expert_errors", "our_errors"]).to_csv(CACHE_DIR / f"saa_{half}_words.csv", index=False)


if __name__ == "__main__":
    main()
