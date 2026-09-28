"""Synthetic test set of typical Turkish-speaker errors: does the coach catch them, and only them?

Every pattern has a few words, each said in a carrier sentence twice: correctly
("Please say think again.") and with the error ("Please say tink again."). The error is made
by editing the word's phonemes, so exactly one kind of sound changes:

  eSpeak NG   phoneme input: "Please say [[t'INk]] again."
  Kokoro-82M  misaki phoneme override: "Please say [think](/tˈɪŋk/) again."

Each recording goes through the full pipeline with the real model and we check the target word:
is it flagged when said correctly (false alarm), flagged with the right tip when said wrongly,
and did GOP confirmation hide a real error?

Synthetic speech is robotic and the recognizer may treat it differently from a human voice, so
this is a regression check for the rules, not a measure of real-world accuracy.

Kokoro needs its own environment, because its G2P (misaki[en]) installs phonemizer-fork, which
would replace the phonemizer package this project uses:
    python3.12 -m venv .venv-tts && .venv-tts/bin/pip install kokoro soundfile

Usage:
    python scripts/synthetic_errors.py                # eSpeak NG voices
    python scripts/synthetic_errors.py --kokoro       # eSpeak NG and Kokoro voices
    python scripts/synthetic_errors.py --details      # also print every recording
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation import analyze  # noqa: E402
from pronunciation.audio import SAMPLE_RATE, load_audio  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.recognizer import Decoder  # noqa: E402

OUT_DIR = ROOT / "data" / "synthetic"
CACHE = OUT_DIR / "log_probs.npz"      # model output per recording, so rule changes re-run in seconds
CACHE_META = OUT_DIR / "model.json"
KOKORO_PYTHON = ROOT / ".venv-tts" / "bin" / "python"
PREFIX, SUFFIX = "Please say", "again."
TARGET = 2  # index of the target word in "Please say <word> again."

ESPEAK_VOICES = ["en-us+m1", "en-us+m3", "en-us+f2", "en-us+f4"]   # two male, two female
ESPEAK_SPEEDS = [130, 175, 220]                                    # words per minute (175 = default)
KOKORO_VOICES = ["am_michael", "am_fenrir", "af_heart", "af_bella"]
KOKORO_SPEEDS = [0.8, 1.0, 1.2]

Rules = list[tuple[str, str]]  # (regex, replacement) pairs applied in order to the word's phonemes
FINAL_DEVOICING_ESPEAK: Rules = [("b$", "p"), ("d$", "t"), ("g$", "k"), ("v$", "f"), ("z$", "s")]
FINAL_DEVOICING_MISAKI: Rules = [("b$", "p"), ("d$", "t"), ("ɡ$", "k"), ("v$", "f"), ("z$", "s")]


@dataclass
class Pattern:
    name: str
    tip: str            # the tip id (phonemes.TIPS) the coach should give
    words: list[str]
    espeak: Rules       # on eSpeak phoneme mnemonics (T = θ, D = ð, I = ɪ, i: = iː, a = æ, U = ʊ, N = ŋ)
    misaki: Rules       # on Kokoro's misaki phonemes (IPA-like, long vowels without ː)


PATTERNS = [
    Pattern("th → t", "th_voiceless", ["think", "thank", "three", "both"], [("T", "t")], [("θ", "t")]),
    Pattern("th → s", "th_voiceless", ["think", "thank", "three", "both"], [("T", "s")], [("θ", "s")]),
    Pattern("ð → d", "th_voiced", ["this", "that", "they", "mother"], [("D", "d")], [("ð", "d")]),
    Pattern("ð → z", "th_voiced", ["this", "that", "they", "mother"], [("D", "z")], [("ð", "z")]),
    Pattern("w → v", "w", ["west", "wine", "window"], [("w", "v")], [("w", "v")]),
    Pattern("ɪ → iː", "short_i", ["ship", "sit", "fill"], [(r"(?<![aeO])I", "i:")], [("ɪ", "i")]),
    Pattern("iː → ɪ", "long_ee", ["sheep", "seat", "feel"], [("i:", "I")], [("i", "ɪ")]),
    Pattern("æ → ɛ", "ae", ["bad", "man", "cat"], [(r"a(?![IU#])", "E")], [("æ", "ɛ")]),
    Pattern("ʊ → uː", "short_u", ["pull", "full", "book"], [(r"(?<![aoO])U", "u:")], [("ʊ", "u")]),
    Pattern("final devoicing", "final_voicing", ["bed", "dog", "bag", "love"],
            FINAL_DEVOICING_ESPEAK, FINAL_DEVOICING_MISAKI),
    Pattern("extra vowel before cluster", "epenthesis", ["school", "street", "sport", "speak"],
            [("^", "I")], [("^", "ɪ")]),
    Pattern("ŋ + g", "ng", ["sing", "long", "singer"], [("N", "Ng")], [("ŋ", "ŋɡ")]),
]


def apply_rules(phonemes: str, rules: Rules) -> str:
    for pattern, replacement in rules:
        phonemes = re.sub(pattern, replacement, phonemes)
    return phonemes


def espeak(*args: str) -> str:
    return subprocess.run(["espeak-ng", *args], check=True, capture_output=True, text=True).stdout.strip()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def plan_recordings(engines: list[str]) -> list[dict]:
    """One row per recording: which sentence, voice, speed and version (correct or error)."""
    rows = []
    for engine in engines:
        voices, speeds = (ESPEAK_VOICES, ESPEAK_SPEEDS) if engine == "espeak" else (KOKORO_VOICES, KOKORO_SPEEDS)
        for pattern in PATTERNS:
            for word in pattern.words:
                for voice in voices:
                    for speed in speeds:
                        for version in ("correct", "error"):
                            name = f"{word}_{voice.replace('+', '_')}_{speed}_{version}.wav"
                            rows.append({"engine": engine, "pattern": pattern.name, "tip": pattern.tip,
                                         "word": word, "voice": voice, "speed": speed, "version": version,
                                         "path": OUT_DIR / engine / slug(pattern.name) / name})
    return rows


def synthesize_espeak(rows: list[dict], regenerate: bool) -> None:
    patterns = {p.name: p for p in PATTERNS}
    for row in rows:
        mnemonic = espeak("-v", "en-us", "-q", "-x", row["word"]).replace(" ", "")
        if row["version"] == "error":
            mnemonic = apply_rules(mnemonic, patterns[row["pattern"]].espeak)
        row["phonemes"] = espeak("-v", "en-us", "-q", "--ipa", f"[[{mnemonic}]]").replace(" ", "")
        if row["path"].exists() and not regenerate:
            continue
        row["path"].parent.mkdir(parents=True, exist_ok=True)
        text = f"{PREFIX} [[{mnemonic}]] {SUFFIX}"
        espeak("-v", row["voice"], "-s", str(row["speed"]), "-w", str(row["path"]), text)


def synthesize_kokoro(rows: list[dict], regenerate: bool) -> None:
    if not KOKORO_PYTHON.exists():
        sys.exit(f"Kokoro needs its own environment ({KOKORO_PYTHON} not found), see the docstring.")
    patterns = {p.name: p for p in PATTERNS}
    jobs = [{"path": str(row["path"]), "voice": row["voice"], "speed": row["speed"], "prefix": PREFIX,
             "word": row["word"], "suffix": SUFFIX,
             "rules": patterns[row["pattern"]].misaki if row["version"] == "error" else [],
             "skip": row["path"].exists() and not regenerate} for row in rows]
    jobs_file = OUT_DIR / "kokoro_jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps(jobs))
    subprocess.run([str(KOKORO_PYTHON), str(ROOT / "scripts" / "kokoro_tts.py"), str(jobs_file)], check=True)
    for row, job in zip(rows, json.loads(jobs_file.read_text())):
        row["phonemes"] = job.get("phonemes", "")


def load_log_probs(rows: list[dict]) -> tuple[Decoder, dict[str, np.ndarray], dict[str, float]]:
    """Model output for every recording, from the cache when possible."""
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    seconds = {}
    missing = [row for row in rows if str(row["path"].relative_to(ROOT)) not in cache]
    if missing or not CACHE_META.exists():
        from pronunciation.recognizer import PhonemeRecognizer

        recognizer = PhonemeRecognizer()
        CACHE_META.write_text(json.dumps({"vocab": recognizer.token_to_id, "blank_id": recognizer.blank_id,
                                          "special_ids": sorted(recognizer.decoder.special_ids)}))
        for k, row in enumerate(missing, 1):
            cache[str(row["path"].relative_to(ROOT))] = recognizer.log_probs(load_audio(row["path"]))
            if k % 200 == 0:
                print(f"model {k}/{len(missing)}")
        if missing:
            np.savez_compressed(CACHE, **cache)
    meta = json.loads(CACHE_META.read_text())
    decoder = Decoder(meta["vocab"], meta["blank_id"], set(meta["special_ids"]))
    for row in rows:
        seconds[str(row["path"].relative_to(ROOT))] = len(load_audio(row["path"])) / SAMPLE_RATE
    return decoder, cache, seconds


def analyze_recordings(rows: list[dict]) -> None:
    decoder, cache, seconds = load_log_probs(rows)
    for row in rows:
        key = str(row["path"].relative_to(ROOT))
        text = f"{PREFIX} {row['word']} {SUFFIX}"
        words = tokenize(text)
        result = analyze(text, phonemize_words(words), decoder(cache[key], seconds[key]),
                         decoder.token_to_id, decoder.blank_id)
        word = result.words[TARGET]
        row.update({
            "flagged": bool(word.issues),
            "right_tip": any(i.tip == row["tip"] for i in word.issues),
            "hidden_by_gop": not word.issues and bool(word.dismissed),
            "gop": word.gop,
            "heard": " ".join(word.heard),
            "carrier_flags": sum(bool(w.issues) for i, w in enumerate(result.words) if i != TARGET),
        })


def pair_table(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (word, voice, speed): the correct and the error recording side by side."""
    keys = ["engine", "pattern", "word", "voice", "speed"]
    correct = results[results.version == "correct"].set_index(keys)
    error = results[results.version == "error"].set_index(keys)
    return pd.DataFrame({
        "false_alarm": correct.flagged,
        "caught": error.flagged,
        "right_tip": error.right_tip,
        "gop": error.gop.round(1),
        "hidden_by_gop": error.hidden_by_gop,
        "heard_error": error.heard,
    }).reset_index()


def pattern_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    order = [p.name for p in PATTERNS]
    summary = pairs.groupby("pattern").agg(
        pairs=("caught", "size"),
        catch=("caught", "mean"),
        right_tip=("right_tip", "mean"),
        false_alarm=("false_alarm", "mean"),
        hidden_by_gop=("hidden_by_gop", "sum"),
    ).reindex(order)
    return summary


def print_summary(title: str, pairs: pd.DataFrame) -> None:
    summary = pattern_summary(pairs)
    total = pd.DataFrame({"pairs": [len(pairs)], "catch": [pairs.caught.mean()],
                          "right_tip": [pairs.right_tip.mean()], "false_alarm": [pairs.false_alarm.mean()],
                          "hidden_by_gop": [pairs.hidden_by_gop.sum()]}, index=["ALL"])
    table = pd.concat([summary, total])
    print(f"\n{title}")
    print(f"{'pattern':28} {'pairs':>5} {'catch':>6} {'right tip':>9} {'false alarm':>11} {'hidden by GOP':>13}")
    for name, r in table.iterrows():
        print(f"{name:28} {int(r.pairs):>5} {r.catch:>6.0%} {r.right_tip:>9.0%} {r.false_alarm:>11.0%} "
              f"{int(r.hidden_by_gop):>13}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kokoro", action="store_true", help="also synthesize with Kokoro-82M")
    parser.add_argument("--regenerate", action="store_true", help="synthesize again even if the files exist")
    parser.add_argument("--details", action="store_true", help="print every recording pair")
    args = parser.parse_args()

    engines = ["espeak", "kokoro"] if args.kokoro else ["espeak"]
    rows = plan_recordings(engines)
    if args.regenerate:
        CACHE.unlink(missing_ok=True)
    synthesize_espeak([r for r in rows if r["engine"] == "espeak"], args.regenerate)
    if args.kokoro:
        synthesize_kokoro([r for r in rows if r["engine"] == "kokoro"], args.regenerate)
    print(f"{len(rows)} recordings in {OUT_DIR.relative_to(ROOT)}/")

    analyze_recordings(rows)
    results = pd.DataFrame(rows)
    results["path"] = results.path.map(lambda p: str(p.relative_to(ROOT)))
    results.to_csv(OUT_DIR / "results.csv", index=False)
    pairs = pair_table(results)
    pairs.to_csv(OUT_DIR / "pairs.csv", index=False)

    if args.details:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(pairs.to_string(index=False))
    for engine in engines:
        print_summary(f"{engine}: per pattern (target word only)", pairs[pairs.engine == engine])
    if len(engines) > 1:
        print_summary("all engines", pairs)
    carrier = results.groupby("engine").carrier_flags.mean()
    print("\nflagged carrier words (please/say/again) per recording:",
          ", ".join(f"{e} {v:.2f}" for e, v in carrier.items()))
    print(f"details: {(OUT_DIR / 'pairs.csv').relative_to(ROOT)}, {(OUT_DIR / 'results.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
