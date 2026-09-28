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

Two labels are reported: the main one (accepted variants such as the flap in "better" or "æn"
for "and" are not errors) and the raw one (every deviation the expert wrote is an error).
95% confidence intervals come from resampling speakers (bootstrap).

The calculations are shared with evaluate_words.py (pronunciation/metrics.py).

The model runs once per half; its output is cached in data/cache/. Only the dev half is used while
we change rules. The test half is for the final evaluation (roadmap step 8b) and needs --final,
which reports both halves side by side and writes them to results/metrics.json.

Usage:
    python scripts/download_saa.py && python scripts/split_saa.py    # once
    python scripts/evaluate_saa.py                                   # dev half
    python scripts/evaluate_saa.py --gop-sweep                       # compare GOP thresholds
    python scripts/evaluate_saa.py --final                           # step 8b only
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
from pronunciation.assess import prepare_expected  # noqa: E402
from pronunciation.audio import load_audio  # noqa: E402
from pronunciation.cache import cached_log_probs  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.metrics import (bootstrap_ci, detection_from_counts, error_level_catch, format_confusion,  # noqa: E402
                                   format_detection, match_errors, top_pairs, word_detection_metrics)
from pronunciation.phonemes import FUNCTION_WORDS, TIPS  # noqa: E402
from pronunciation.recognizer import DEFAULT_MODEL, Decoder  # noqa: E402
from pronunciation.saa import (PARAGRAPH, align_words, expert_labels, narrow_to_broad,  # noqa: E402
                               parse_transcription, raw_wrong)
from pronunciation.systems import System  # noqa: E402

SAA_DIR = ROOT / "data" / "saa"
SPLIT_FILE = ROOT / "data" / "saa_split.json"
CACHE_DIR = ROOT / "data" / "cache"
V1 = System("v1")   # rules with the default GOP thresholds
METRICS_FILE = ROOT / "results" / "metrics.json"
LABELS = {"main": "expert_wrong", "raw": "expert_wrong_raw"}
DEFINITIONS = {
    "main": "a word is wrong when the expert transcription differs from the expected phonemes "
            "outside our accepted variants (flap, reduced vowels, weak forms of function words...)",
    "raw": "a word is wrong when the expert transcription differs from the expected phonemes at all "
           "(only narrow-IPA detail and vowel length are ignored)",
}


def load(half: str, model_id: str = DEFAULT_MODEL):
    """Decoder, and (speaker, log-probabilities, seconds, transcription) for every speaker of the half."""
    speakers = json.loads(SPLIT_FILE.read_text())[half]
    audio = {s: (lambda s=s: load_audio(SAA_DIR / f"{s}.mp3")) for s in speakers}
    decoder, log_probs, seconds = cached_log_probs(model_id, f"saa_{half}", audio)
    data = [(s, log_probs[s], seconds[s], (SAA_DIR / f"{s}.txt").read_text(encoding="utf-8")) for s in speakers]
    return decoder, data


def evaluate(decoder: Decoder, data, system: System = V1) -> pd.DataFrame:
    """One row per paragraph word per speaker: what the expert heard and what the system reported."""
    words = tokenize(PARAGRAPH)
    raw = phonemize_words(words)
    word_phones = prepare_expected(words, raw)[0]
    word_is_function = [w.lower() in FUNCTION_WORDS for w in words]

    rows = []
    for speaker, log_probs, seconds, transcription in data:
        expert_words = [narrow_to_broad(w) for w in parse_transcription(transcription)]
        expert = align_words(word_phones, expert_words, word_is_function)
        labels = expert_labels(words, word_phones, expert)
        result = system.assess(PARAGRAPH, raw, decoder(log_probs, seconds), decoder.token_to_id, decoder.blank_id)
        for k, (label, ours) in enumerate(zip(labels, result.words)):
            rows.append({
                "speaker": speaker, "group": speaker.rstrip("0123456789"), "word_idx": k, "word": words[k],
                "expert_wrong": label.wrong, "expert_wrong_raw": raw_wrong(word_phones[k], expert[k]),
                "expert_tips": sorted(label.tips), "expert_pairs": label.pairs,
                "expert_errors": label.errors, "our_errors": ours.issues,
                "flagged": bool(ours.issues), "our_tips": sorted({i.tip for i in ours.issues if i.tip}),
                "our_pairs": [f"{i.expected or '-'} → {i.heard or '-'}" for i in ours.issues],
                "dismissed": bool(ours.dismissed) and not ours.issues, "gop": ours.gop,
                "level": ours.level, "error_prob": ours.error_prob,
            })
    return pd.DataFrame(rows)


def detection(rows: pd.DataFrame, label: str = "expert_wrong") -> dict:
    """Word-level metrics; words the speaker skipped are left out."""
    rows = rows[rows[label].notna()]
    return word_detection_metrics(rows[label].astype(bool), rows.flagged)


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


def confusion_units(rows: pd.DataFrame, label: str = "expert_wrong") -> dict[str, list[int]]:
    """Per speaker: [tp, fp, fn, tn], the unit for bootstrap intervals and system comparisons."""
    rows = rows[rows[label].notna()]
    units = {}
    for name, speaker in rows.groupby("speaker"):
        wrong, flagged = speaker[label].astype(bool), speaker.flagged
        units[name] = [int((wrong & flagged).sum()), int((~wrong & flagged).sum()),
                       int((wrong & ~flagged).sum()), int((~wrong & ~flagged).sum())]
    return units


def error_units(rows: pd.DataFrame, keep=lambda error: True) -> dict[str, list[int]]:
    """Per speaker: [expert errors we found, expert errors], counting only the errors `keep` selects."""
    units = {}
    for name, speaker in rows[rows.expert_wrong.notna()].groupby("speaker"):
        found = total = 0
        for expert, ours in zip(speaker.expert_errors, speaker.our_errors):
            kept = [e for e in expert if keep(e)]
            found += match_errors(kept, ours)[0]
            total += len(kept)
        units[name] = [found, total]
    return units


def with_ci(rows: pd.DataFrame, label: str) -> dict:
    """Word-level metrics of one group, with 95% intervals from resampling its speakers."""
    units = list(confusion_units(rows, label).values())
    result = detection(rows, label)
    result["speakers"] = len(units)
    result["ci95"] = {key: bootstrap_ci(units, lambda u, key=key: detection_from_counts(*np.sum(u, axis=0))[key])
                      for key in ("false_alarm", "recall", "precision", "f1")}
    return result


def error_level_with_ci(rows: pd.DataFrame) -> dict:
    result = error_level_catch(zip(rows.expert_errors, rows.our_errors))
    units = list(error_units(rows).values())
    result["ci95"] = {"catch": bootstrap_ci(units, lambda u: sum(x[0] for x in u) / max(sum(x[1] for x in u), 1))}
    return result


def summarize(rows: pd.DataFrame) -> dict:
    """Everything that goes to metrics.json for one half."""
    turkish = rows[(rows.group == "turkish") & rows.expert_wrong.notna()]
    summary = {"definitions": DEFINITIONS}
    for version, label in LABELS.items():
        summary[version] = {group: with_ci(rows[rows.group == group], label) for group in ("english", "turkish")}
    summary["main"]["turkish_error_level"] = error_level_with_ci(turkish)
    if rows.level.notna().any():   # learned detector: red words alone
        summary["main"]["turkish_red"] = detection(turkish.assign(flagged=turkish.level == "red"))
    summary["main"]["turkish_patterns"] = pattern_table(turkish).reset_index().to_dict(orient="records")
    return summary


def report(rows: pd.DataFrame) -> None:
    for group in ("english", "turkish"):
        print(format_detection(group, detection(rows[rows.group == group])))
    for group in ("english", "turkish"):
        print(format_detection(f"{group} raw", detection(rows[rows.group == group], "expert_wrong_raw")))
    skipped = rows.expert_wrong.isna().sum()
    print(f"(raw: every deviation the expert wrote is an error; words the speaker skipped or could not be "
          f"matched: {skipped}, left out)")

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


def cell(m: dict, key: str) -> str:
    low, high = m["ci95"][key]
    return f"{m[key]:.1%} [{low:.1%}–{high:.1%}]" if key != "f1" else f"{m[key]:.3f} [{low:.3f}–{high:.3f}]"


def markdown_tables(halves: dict[str, dict]) -> str:
    """Dev and test side by side, ready for the README (values as computed, 95% CI in brackets)."""
    dev, test = halves["dev"], halves["test"]
    header = (f"| | Dev ({dev['main']['english']['speakers']} English / {dev['main']['turkish']['speakers']} Turkish speakers) "
              f"| Test ({test['main']['english']['speakers']} English / {test['main']['turkish']['speakers']} Turkish speakers) |\n"
              "| --- | --- | --- |\n")
    lines = []
    for version in ("main", "raw"):
        lines.append(f"**{version.capitalize()} labels** ({DEFINITIONS[version]})\n\n" + header + "".join([
            f"| English speakers: false alarm | {cell(dev[version]['english'], 'false_alarm')} | {cell(test[version]['english'], 'false_alarm')} |\n",
            f"| Turkish speakers: false alarm | {cell(dev[version]['turkish'], 'false_alarm')} | {cell(test[version]['turkish'], 'false_alarm')} |\n",
            f"| Turkish speakers: recall | {cell(dev[version]['turkish'], 'recall')} | {cell(test[version]['turkish'], 'recall')} |\n",
            f"| Turkish speakers: precision | {cell(dev[version]['turkish'], 'precision')} | {cell(test[version]['turkish'], 'precision')} |\n",
            f"| Turkish speakers: F1 | {cell(dev[version]['turkish'], 'f1')} | {cell(test[version]['turkish'], 'f1')} |\n",
        ]))
    d, t = dev["main"]["turkish_error_level"], test["main"]["turkish_error_level"]
    lines[0] += f"| Turkish speakers: error-level catch | {cell(d, 'catch')} | {cell(t, 'catch')} |\n"
    patterns = {"dev": {p["pattern"]: p for p in dev["main"]["turkish_patterns"]},
                "test": {p["pattern"]: p for p in test["main"]["turkish_patterns"]}}
    table = ("**Per pattern, Turkish speakers** (errors the expert heard → share we found on the same sound)\n\n"
             "| Pattern | Dev | Test |\n| --- | --- | --- |\n")
    for tip in TIPS:
        cells = [f"{patterns[h][tip]['error_caught']:.0%} of {patterns[h][tip]['errors']}" if tip in patterns[h] else "–"
                 for h in ("dev", "test")]
        if cells != ["–", "–"]:
            table += f"| {TIPS[tip]['title']} | {cells[0]} | {cells[1]} |\n"
    return "\n".join(lines) + "\n" + table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true", help="evaluate the held-out test half (step 8b only)")
    parser.add_argument("--gop-sweep", action="store_true", help="compare GOP confirmation thresholds (dev half)")
    parser.add_argument("--metrics", type=Path, default=METRICS_FILE, help="where --final writes its results")
    args = parser.parse_args()

    half = "test" if args.final else "dev"
    decoder, data = load(half)
    print(f"Speech Accent Archive, {half} half: {len(data)} speakers\n")
    if args.gop_sweep and not args.final:
        for threshold in [None, -1.0, -1.5, -2.0, -2.5, -3.0, -4.0]:
            rows = evaluate(decoder, data, System(f"gop {threshold}", settings={"gop_threshold": threshold}))
            print(format_detection(f"GOP {threshold}", detection(rows[rows.group == "turkish"])))
        return
    rows = evaluate(decoder, data)
    report(rows)
    rows.drop(columns=["expert_errors", "our_errors"]).to_csv(CACHE_DIR / f"saa_{half}_words.csv", index=False)

    if args.final:
        dev_decoder, dev_data = load("dev")
        halves = {"dev": summarize(evaluate(dev_decoder, dev_data)), "test": summarize(rows)}
        metrics = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
        metrics["speech_accent_archive"] = halves
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2))
        print(f"\n{markdown_tables(halves)}\nSaved to {args.metrics} under \"speech_accent_archive\"")


if __name__ == "__main__":
    main()
