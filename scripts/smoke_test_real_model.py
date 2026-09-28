"""Smoke test with the real wav2vec2 model: synthesize practice sentences with eSpeak NG and analyze them.

Synthetic speech is robotic, so recognition will not be perfect. The goal is to check that the
real model loads and gives output that mostly matches the expected phonemes.

Usage:
    python scripts/smoke_test_real_model.py
    PC_DEVICE=cpu python scripts/smoke_test_real_model.py --n 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation import PronunciationCoach  # noqa: E402
from pronunciation.audio import load_audio  # noqa: E402
from pronunciation.recognizer import PhonemeRecognizer  # noqa: E402


def synthesize(text: str, path: Path) -> None:
    subprocess.run(["espeak-ng", "-v", "en-us", "-w", str(path), text], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5, help="number of sentences from data/sentences.json")
    args = parser.parse_args()

    sentences = [s["text"] for s in json.loads((ROOT / "data/sentences.json").read_text())][: args.n]

    start = time.perf_counter()
    coach = PronunciationCoach(recognizer=PhonemeRecognizer())
    print(f"Device: {coach.recognizer.device}   model loaded in {time.perf_counter() - start:.1f} s\n")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(sentences, 1):
            wav = Path(tmp) / f"sentence_{i}.wav"
            synthesize(text, wav)
            audio = load_audio(wav)

            start = time.perf_counter()
            result = coach.assess(audio, text)
            seconds = time.perf_counter() - start

            expected = " ".join(p for w in result.words for p in w.expected)
            print(f"{i}. {text}")
            print(f"   expected: {expected}")
            print(f"   heard:    {' '.join(result.heard_phones)}\n")
            rows.append((i, len(audio) / 16_000, result.phone_accuracy, seconds))

    print(f"{'#':>2}  {'audio (s)':>9}  {'phone_accuracy':>14}  {'analysis (s)':>12}")
    for i, audio_s, accuracy, seconds in rows:
        print(f"{i:>2}  {audio_s:>9.2f}  {accuracy:>14.3f}  {seconds:>12.2f}")


if __name__ == "__main__":
    main()
