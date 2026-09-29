"""Build the fine-tuning manifests from L2-ARCTIC (non-native) and CMU ARCTIC (native US) speech.

L2-ARCTIC (after scripts/prepare_l2arctic.py): every annotated sentence becomes a target: our eSpeak
expected tokens with the annotated errors applied (pronunciation/l2arctic.py). Sentences with an
"err" label, an unreliable alignment or a token the recognizer does not know are left out.

CMU ARCTIC (bdl, slt, clb, rms, same sentences): the target is simply our expected tokens, so the
model keeps hearing native speech as correct. Sentences annotated for the L2-ARCTIC dev and test
speakers are removed. A few sentences annotated for no L2 speaker at all form a native dev set.

Writes data/finetune/{l2arctic_train,l2arctic_dev,native_train,native_dev}.jsonl and a report,
results/l2arctic_data.json. The test speakers' manifest is only built with --final (v2-4).

Usage:
    python scripts/build_finetune_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation.assess import prepare_expected  # noqa: E402
from pronunciation.g2p import phonemize_words, tokenize  # noqa: E402
from pronunciation.l2arctic import SPEAKERS, build_target, load_speakers, parse_textgrid, word_annotations  # noqa: E402
from pronunciation.recognizer import DEFAULT_MODEL  # noqa: E402

DATA = ROOT / "data"
L2_DIR, CMU_DIR, OUT_DIR = DATA / "l2arctic", DATA / "cmu_arctic", DATA / "finetune"
NATIVE = ["bdl", "slt", "clb", "rms"]
NATIVE_DEV_SENTENCES = 25
SEED = 762
# (expected, heard) pairs that are typical Turkish-speaker errors, for the report
TURKISH_PAIRS = [("z", "s"), ("θ", "t"), ("θ", "s"), ("ð", "d"), ("ð", "z"), ("ɪ", "i"), ("ɪ", "iː"),
                 ("iː", "ɪ"), ("w", "v"), ("v", "w"), ("æ", "ɛ"), ("ʊ", "uː"), ("d", "t"), ("b", "p"), ("ɡ", "k")]


def vocabulary() -> set[str]:
    from transformers import AutoTokenizer

    return set(AutoTokenizer.from_pretrained(DEFAULT_MODEL, do_phonemize=False).get_vocab())


def annotated_ids(speaker: str) -> set[str]:
    return {p.stem for p in (L2_DIR / speaker / "annotation").glob("*.TextGrid")}


def l2_rows(speakers: list[str], vocab: set[str], reasons: Counter) -> list[dict]:
    rows = []
    for speaker in speakers:
        for grid in sorted((L2_DIR / speaker / "annotation").glob("*.TextGrid")):
            tiers = parse_textgrid(grid.read_text(encoding="utf-8"))
            try:
                words = word_annotations(tiers["words"], tiers["phones"])
            except ValueError:
                reasons["bad label"] += 1
                continue
            texts = [w.text for w in words]
            raw = phonemize_words(texts)
            target = build_target(words, raw)
            if target.ok and not set(target.tokens) | set(target.canonical) <= vocab:
                target.ok, target.reason = False, "token not in vocab"
            reasons[target.reason or "kept"] += 1
            if not target.ok:
                continue
            _, flat, _, flags = prepare_expected(texts, raw)
            assert flat == target.expected
            audio = L2_DIR / speaker / f"{grid.stem}.flac"
            rows.append({
                "id": f"{speaker}/{grid.stem}", "speaker": speaker, "l1": SPEAKERS[speaker][0],
                "gender": SPEAKERS[speaker][1], "audio": str(audio.relative_to(DATA)),
                "seconds": sf.info(audio).duration, "text": " ".join(texts),
                "tokens": " ".join(target.tokens), "canonical": target.canonical,
                "expected": target.expected, "perceived": target.perceived, "function": flags,
                "errors": [[e.kind, e.expected, e.heard, e.tip] for e in target.errors],
            })
    return rows


def native_rows(sentence_ids: set[str], vocab: set[str], reasons: Counter) -> list[dict]:
    rows = []
    for speaker in NATIVE:
        folder = CMU_DIR / f"cmu_us_{speaker}_arctic"
        prompts = dict(re.findall(r'\( (\S+) "(.*)" \)', (folder / "etc" / "txt.done.data").read_text()))
        for utt in sorted(sentence_ids & set(prompts)):
            texts = tokenize(prompts[utt])
            raw = phonemize_words(texts)
            tokens = [t for word in raw for t in word]
            if not set(tokens) <= vocab:
                reasons["token not in vocab"] += 1
                continue
            _, flat, _, flags = prepare_expected(texts, raw)
            audio = folder / "wav" / f"{utt}.wav"
            rows.append({
                "id": f"{speaker}/{utt}", "speaker": speaker, "l1": "English", "gender": "",
                "audio": str(audio.relative_to(DATA)), "seconds": sf.info(audio).duration,
                "text": " ".join(texts), "tokens": " ".join(tokens), "canonical": tokens,
                "expected": flat, "perceived": flat, "function": flags, "errors": [],
            })
    return rows


def error_report(rows: list[dict]) -> dict:
    errors = [e for row in rows for e in row["errors"]]
    pairs = Counter(f"{e[1] or '-'} → {e[2] or '-'}" for e in errors)
    return {
        "sentences": len(rows), "hours": round(sum(r["seconds"] for r in rows) / 3600, 2),
        "errors": len(errors), "sentences_with_errors": sum(bool(r["errors"]) for r in rows),
        "by_kind": dict(Counter(e[0] for e in errors)),
        "top_pairs": pairs.most_common(20),
        "turkish_tips": dict(Counter(e[3] for e in errors if e[3]).most_common()),
        "turkish_pairs": {f"{a} → {b}": pairs[f"{a} → {b}"] for a, b in TURKISH_PAIRS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final", action="store_true", help="also build the locked test speakers' manifest (v2-4)")
    args = parser.parse_args()

    vocab = vocabulary()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"reasons": {}}
    parts = {part: load_speakers(part) for part in ("train", "dev")}
    for part, speakers in parts.items():
        reasons: Counter = Counter()
        rows = l2_rows(speakers, vocab, reasons)
        write(OUT_DIR / f"l2arctic_{part}.jsonl", rows)
        report[f"l2arctic_{part}"] = error_report(rows)
        report["reasons"][f"l2arctic_{part}"] = dict(reasons)
    if args.final:
        rows = l2_rows(load_speakers("test", final=True), vocab, Counter())
        write(OUT_DIR / "l2arctic_test.jsonl", rows)

    # which annotated sentences the L2-ARCTIC parts share (reading only the file names)
    ids = {part: set().union(*(annotated_ids(s) for s in speakers)) for part, speakers in parts.items()}
    ids["test"] = set().union(*(annotated_ids(s) for s in load_speakers("test", final=True)))
    report["sentence_overlap"] = {
        "train_sentences": len(ids["train"]), "dev_sentences": len(ids["dev"]), "test_sentences": len(ids["test"]),
        "dev_also_in_train": len(ids["dev"] & ids["train"]), "test_also_in_train": len(ids["test"] & ids["train"]),
    }

    prompts = re.findall(r"\( (\S+) ", (CMU_DIR / "cmu_us_bdl_arctic" / "etc" / "txt.done.data").read_text())
    unused = sorted(set(prompts) - ids["train"] - ids["dev"] - ids["test"])
    native_dev = set(random.Random(SEED).sample(unused, NATIVE_DEV_SENTENCES))
    native_train = set(prompts) - ids["dev"] - ids["test"] - native_dev
    reasons = Counter()
    for part, sentence_ids in (("native_train", native_train), ("native_dev", native_dev)):
        rows = native_rows(sentence_ids, vocab, reasons)
        write(OUT_DIR / f"{part}.jsonl", rows)
        report[part] = {"sentences": len(rows), "hours": round(sum(r["seconds"] for r in rows) / 3600, 2)}
    report["reasons"]["native"] = dict(reasons)

    (ROOT / "results" / "l2arctic_data.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
