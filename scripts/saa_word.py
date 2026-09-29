"""One paragraph word on SAA dev: what every expert wrote and what each system heard and reported.

Used to check a word-specific rule on real speakers before adding it, e.g. how native speakers
say "into" (the dictionary form ɪntuː, the weak form ɪntə). Only the dev half is used.

Usage:
    python scripts/saa_word.py into experiments/v1.json experiments/v2.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_saa  # noqa: E402
from pronunciation.phonemes import is_vowel  # noqa: E402
from pronunciation.systems import load_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("word", help="a word of the paragraph, e.g. into")
    parser.add_argument("systems", nargs="+", type=Path)
    args = parser.parse_args()

    tables = []
    for path in args.systems:
        system = load_system(path)
        decoder, data = evaluate_saa.load("dev", system.recognizer)
        rows = evaluate_saa.evaluate(decoder, data, system)
        tables.append((system.name, rows[rows.word.str.lower() == args.word.lower()].set_index(["speaker", "word_idx"])))

    first = tables[0][1]
    for group in ("english", "turkish"):
        print(f"\n{group} speakers: '{args.word}'")
        last_vowels = Counter()
        for key, row in first[first.group == group].iterrows():
            vowels = [p for p in (row.expert_phones or []) if is_vowel(p)]
            last_vowels[vowels[-1] if vowels else "(skipped)"] += 1
            heard = "   ".join(f"{name} /{' '.join(t.loc[key].heard_phones)}/ {t.loc[key].our_pairs or ''}" for name, t in tables)
            print(f"  {key[0]:10} expert [{row.expert_ipa}] -> /{' '.join(row.expert_phones or [])}/   {heard}")
        print(f"  last vowel the expert wrote: {dict(last_vowels)}")


if __name__ == "__main__":
    main()
