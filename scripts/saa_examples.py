"""Examples from SAA dev where two systems disagree, for the README and for explaining the model.

  new catches        Turkish speakers: words the expert marked wrong that system B reports with the
                     expert's own error and system A missed (one per error pattern first)
  new false alarms   US English speakers: words the expert found correct that B flags and A did not

For each word: the expected sounds, the expert's transcription (narrow IPA as written), what each
system's recognizer heard in the word, and what B reports. Only the dev half is used.

Usage:
    python scripts/saa_examples.py experiments/v1.json experiments/v2-3-best.json --out results/v2-3_saa_examples.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_saa  # noqa: E402
from pronunciation.metrics import match_errors  # noqa: E402
from pronunciation.systems import System, load_system  # noqa: E402

# the patterns to show first, in this order: (expected, heard) pairs of the expert
PREFERRED = [("z", "s"), ("ð", "d"), ("θ", "t"), ("w", "v"), ("ɹ", "ɾ"), ("ɹ", "r"), ("ɪ", "i"), ("v", "f"), ("d", "t")]


def rows_for(system: System) -> pd.DataFrame:
    decoder, data = evaluate_saa.load("dev", system.recognizer)
    return evaluate_saa.evaluate(decoder, data, system).set_index(["speaker", "word_idx"])


def exact_pairs(row) -> list[tuple]:
    """The expert's errors in this word that B reported exactly (same sound, same replacement)."""
    return [(e.expected, e.heard) for e in row.expert_errors if match_errors([e], row.our_errors)[1]]


def new_catches(a: pd.DataFrame, b: pd.DataFrame, n: int) -> list[tuple]:
    wanted = (b.group == "turkish") & (b.expert_wrong == True) & ~a.flagged & b.flagged  # noqa: E712
    candidates = [(key, pairs) for key, row in b[wanted].sort_index().iterrows() if (pairs := exact_pairs(row))]
    chosen, used = [], set()
    for pattern in PREFERRED:                     # one example per pattern first
        for key, pairs in candidates:
            if pattern in pairs and key not in used and len(chosen) < n:
                chosen.append(key)
                used.add(key)
                break
    chosen += [key for key, _ in candidates if key not in used][: n - len(chosen)]
    return chosen


def new_false_alarms(a: pd.DataFrame, b: pd.DataFrame, n: int) -> list[tuple]:
    wanted = (b.group == "english") & (b.expert_wrong == False) & ~a.flagged & b.flagged  # noqa: E712
    return list(b[wanted].sort_index().index[:n])


def table(keys: list[tuple], a: pd.DataFrame, b: pd.DataFrame, names: tuple[str, str]) -> str:
    lines = [f"| Speaker | Word | Expected | Expert wrote | {names[0]} heard | {names[1]} heard | {names[1]} reports |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for key in keys:
        ra, rb = a.loc[key], b.loc[key]
        reported = ", ".join(f"{i.expected or '-'} → {i.heard or '-'}" for i in rb.our_errors)
        lines.append(f"| {key[0]} | {rb.word} | /{' '.join(rb.expected_phones)}/ | [{rb.expert_ipa or '-'}] | "
                     f"/{' '.join(ra.heard_phones)}/ | /{' '.join(rb.heard_phones)}/ | {reported} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="the reference system, e.g. experiments/v1.json")
    parser.add_argument("b", type=Path, help="the new system")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    system_a, system_b = load_system(args.a), load_system(args.b)
    a, b = rows_for(system_a), rows_for(system_b)
    names = (system_a.name, system_b.name)
    text = (f"### Turkish speakers: real errors {names[1]} catches and {names[0]} missed\n\n"
            + table(new_catches(a, b, args.n), a, b, names)
            + f"\n\n### US English speakers: new false alarms of {names[1]}\n\n"
            + table(new_false_alarms(a, b, args.n), a, b, names) + "\n")
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
