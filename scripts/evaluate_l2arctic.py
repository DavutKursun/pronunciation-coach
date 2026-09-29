"""Phone-level evaluation of a recognizer on L2-ARCTIC: PER and mispronunciation detection (MDD).

For every annotated sentence (data/finetune/l2arctic_<part>.jsonl, from build_finetune_data.py):
  PER       recognized phonemes vs what the annotator heard (edit distance / length)
  MDD       expected vs heard vs recognized, per expected sound: TA, FR, FA, TR, precision, recall,
            F1, diagnosis accuracy (pronunciation/metrics.py)
Native speech (CMU ARCTIC native dev): PER vs the expected phonemes, and how many sounds would be
flagged (false rejections).

The recognizer's output is decoded like in the app (restricted to English and learner phonemes)
and cached per model, so a model is run only once per data set. The dev speakers are used during
development; the test speakers are locked until v2-4 and need --final.

Usage:
    python scripts/evaluate_l2arctic.py                                  # the original model
    python scripts/evaluate_l2arctic.py --model models/recognizer-l2arctic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation.audio import load_audio  # noqa: E402
from pronunciation.cache import cached_log_probs  # noqa: E402
from pronunciation.metrics import mdd_counts, mdd_metrics, phone_error_rate, phone_states  # noqa: E402
from pronunciation.phonemes import normalize  # noqa: E402
from pronunciation.recognizer import DEFAULT_MODEL  # noqa: E402

DATA = ROOT / "data"
MANIFESTS = DATA / "finetune"
FOCUS = [("z", "s"), ("θ", "t"), ("ð", "d"), ("ɪ", "iː"), ("w", "v"), ("æ", "ɛ")]


def read_manifest(name: str) -> list[dict]:
    return [json.loads(line) for line in (MANIFESTS / f"{name}.jsonl").read_text().splitlines()]


def recognize(model_id: str, dataset: str, rows: list[dict]) -> dict[str, list[str]]:
    """Recognized (normalized) phonemes for every row, from the per-model cache."""
    audio = {row["id"]: (lambda path=DATA / row["audio"]: load_audio(path)) for row in rows}
    decoder, log_probs, seconds = cached_log_probs(model_id, dataset, audio)
    return {key: normalize(decoder(log_probs[key], seconds[key]).phones) for key in audio}


def evaluate(rows: list[dict], recognized: dict[str, list[str]]) -> dict:
    counts, by_l1 = Counter(), {}
    focus = Counter()
    for row in rows:
        c = mdd_counts(row["expected"], row["perceived"], recognized[row["id"]], row["function"])
        counts += c
        by_l1.setdefault(row["l1"], Counter()).update(c)
        focus.update(focus_catch(row, recognized[row["id"]]))
    return {
        "sentences": len(rows),
        "per": phone_error_rate((row["perceived"], recognized[row["id"]]) for row in rows),
        "per_vs_expected": phone_error_rate((row["expected"], recognized[row["id"]]) for row in rows),
        "mdd": mdd_metrics(counts),
        "mdd_by_l1": {l1: mdd_metrics(c) for l1, c in sorted(by_l1.items())},
        "focus": {f"{a} → {b}": {"errors": focus[(a, b, "n")], "detected": focus[(a, b, "detected")],
                                 "diagnosed": focus[(a, b, "diagnosed")]} for a, b in FOCUS},
    }


def focus_catch(row: dict, recognized: list[str]) -> Counter:
    """For typical Turkish-speaker errors (z → s, θ → t, ...) that the annotator heard: did the recognizer
    output something other than the expected sound there (detected), and was it the heard sound (diagnosed)?"""
    wanted = set(FOCUS)
    p_state, _ = phone_states(row["expected"], row["perceived"], row["function"])
    r_state, _ = phone_states(row["expected"], recognized, row["function"])
    out = Counter()
    for expected, p, r in zip(row["expected"], p_state, r_state):
        if (expected, p) in wanted:
            out[(expected, p, "n")] += 1
            out[(expected, p, "detected")] += r is not None
            out[(expected, p, "diagnosed")] += r == p
    return out


def native(rows: list[dict], recognized: dict[str, list[str]]) -> dict:
    counts = Counter()
    for row in rows:
        counts += mdd_counts(row["expected"], row["expected"], recognized[row["id"]], row["function"])
    m = mdd_metrics(counts)
    return {"sentences": len(rows), "per": phone_error_rate((r["expected"], recognized[r["id"]]) for r in rows),
            "false_rejection_rate": m["false_rejection_rate"]}


def report(model_id: str, l2: dict, nat: dict) -> None:
    m = l2["mdd"]
    print(f"Model: {model_id}\n")
    print(f"L2-ARCTIC ({l2['sentences']} sentences): PER vs heard {l2['per']:.1%}, vs expected {l2['per_vs_expected']:.1%}")
    print(f"MDD: precision {m['precision']:.1%}  recall {m['recall']:.1%}  F1 {m['f1']:.1%}  "
          f"diagnosis accuracy {m['diagnosis_accuracy']:.1%}  (TA {m['ta']}, FR {m['fr']}, FA {m['fa']}, TR {m['tr']})")
    for l1, x in l2["mdd_by_l1"].items():
        print(f"  {l1:11} precision {x['precision']:.1%}  recall {x['recall']:.1%}  F1 {x['f1']:.1%}")
    print("Typical Turkish-speaker errors heard by the annotator (expected → heard): detected / right sound")
    for pair, x in l2["focus"].items():
        if x["errors"]:
            print(f"  {pair:8} {x['errors']:4} errors   detected {x['detected'] / x['errors']:.0%}   "
                  f"right sound {x['diagnosed'] / x['errors']:.0%}")
    print(f"\nNative US speakers ({nat['sentences']} sentences): PER {nat['per']:.1%}, "
          f"sounds flagged although said right {nat['false_rejection_rate']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="a Hugging Face id or a local folder")
    parser.add_argument("--final", action="store_true", help="evaluate the locked test speakers (v2-4 only)")
    parser.add_argument("--out", type=Path, help="save the results as JSON")
    args = parser.parse_args()

    part = "test" if args.final else "dev"
    if args.final and not (MANIFESTS / "l2arctic_test.jsonl").exists():
        sys.exit("Build the test manifest first: python scripts/build_finetune_data.py --final")
    rows = read_manifest(f"l2arctic_{part}")
    native_rows = read_manifest("native_dev")
    l2 = evaluate(rows, recognize(args.model, f"l2arctic_{part}", rows))
    nat = native(native_rows, recognize(args.model, "native_dev", native_rows))
    report(args.model, l2, nat)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": args.model, "part": part, "l2arctic": l2, "native": nat}, indent=2,
                                       ensure_ascii=False) + "\n")
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
