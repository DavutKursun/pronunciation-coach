"""Run one system (recognizer + decision mechanism + settings) on every development set.

  SAA dev          Turkish and US English speakers, expert IPA (main and raw labels, error level,
                   per pattern, 95% speaker-bootstrap intervals)
  speechocean val  the 25 "val" speakers of speechocean762 train
  synthetic        the synthetic error set (eSpeak NG and, if synthesized, Kokoro voices)

The SAA test half and the speechocean762 test split are never touched here (locked until v2-4).
Results, including per-speaker counts for scripts/compare_experiments.py, go to
results/experiments/<name>.json. Model output is cached per model, so only the first run of a
new recognizer is slow.

Usage:
    python scripts/run_experiment.py experiments/v1.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_saa  # noqa: E402
import evaluate_words  # noqa: E402
import synthetic_errors  # noqa: E402
from pronunciation.metrics import format_detection, pearson  # noqa: E402
from pronunciation.systems import System, load_system  # noqa: E402

OUT_DIR = ROOT / "results" / "experiments"
# v2 targets (CLAUDE.md), measured here on SAA dev; the final check is on the SAA test half
TARGETS = [
    ("Turkish precision", "turkish_precision", ">=", 0.70),
    ("Turkish recall", "turkish_recall", ">=", 0.50),
    ("English false alarm", "english_false_alarm", "<=", 0.025),
    ("final devoicing catch", "final_voicing_catch", ">=", 0.60),
    ("ð catch", "th_voiced_catch", ">=", 0.40),
    ("θ catch", "th_voiceless_catch", ">=", 0.75),
]


def git_version() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def catch(units: dict[str, list[int]]) -> float:
    found, total = (sum(u[i] for u in units.values()) for i in (0, 1))
    return found / total if total else 0.0


def run_saa(system: System) -> tuple[dict, dict]:
    decoder, data = evaluate_saa.load("dev", system.recognizer)
    rows = evaluate_saa.evaluate(decoder, data, system)
    summary = evaluate_saa.summarize(rows)
    summary["canonical_bias"] = evaluate_saa.recognized_bias(decoder, data)
    return summary, evaluate_saa.saa_units(rows)


def run_speechocean(system: System) -> tuple[dict, dict]:
    decoder, utterances = evaluate_words.load("val", system.recognizer)
    rows = evaluate_words.evaluate(decoder, utterances, system)
    summary = {"expert_below_10": evaluate_words.detection(rows), "expert_below_7": evaluate_words.detection(rows, 7),
               "word_score_pcc": pearson([r["our_score"] for r in rows], [r["accuracy"] for r in rows]),
               "utterances": len(utterances)}
    return summary, {"speechocean_val": evaluate_words.confusion_units(rows)}


def run_synthetic(system: System) -> tuple[dict, dict]:
    engines = [e for e in ("espeak", "kokoro") if (synthetic_errors.OUT_DIR / e).exists()]
    rows = [r for r in synthetic_errors.plan_recordings(engines) if r["path"].exists()]
    synthetic_errors.analyze_recordings(rows, system)
    pairs = synthetic_errors.pair_table(pd.DataFrame(rows))
    kokoro = pairs[pairs.engine == "kokoro"]
    keys = kokoro.pattern + "|" + kokoro.word + "|" + kokoro.voice + "|" + kokoro.speed.astype(str)
    units = {"synthetic_kokoro_catch": {k: [int(c), 1] for k, c in zip(keys, kokoro.caught)},
             "synthetic_kokoro_false_alarm": {k: [int(f), 1] for k, f in zip(keys, kokoro.false_alarm)}}
    summary = {}
    for engine, part in pairs.groupby("engine"):
        table = synthetic_errors.pattern_summary(part)
        table.loc["ALL"] = [len(part), part.caught.mean(), part.right_tip.mean(), part.false_alarm.mean(),
                            part.hidden_by_gop.sum()]
        summary[engine] = table.reset_index().rename(columns={"index": "pattern"}).to_dict(orient="records")
    return summary, units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", type=Path, help="system definition, e.g. experiments/v1.json")
    args = parser.parse_args()
    system = load_system(args.system)

    saa, saa_units = run_saa(system)
    speechocean, so_units = run_speechocean(system)
    synthetic, synthetic_units = run_synthetic(system)
    units = {**saa_units, **so_units, **synthetic_units}

    values = {
        "turkish_precision": saa["main"]["turkish"]["precision"],
        "turkish_recall": saa["main"]["turkish"]["recall"],
        "english_false_alarm": saa["main"]["english"]["false_alarm"],
        "final_voicing_catch": catch(units["saa_dev_final_voicing"]),
        "th_voiced_catch": catch(units["saa_dev_th_voiced"]),
        "th_voiceless_catch": catch(units["saa_dev_th_voiceless"]),
    }
    targets = [{"target": label, "value": values[key], "goal": f"{op} {goal:.1%}",
                "met": values[key] >= goal if op == ">=" else values[key] <= goal} for label, key, op, goal in TARGETS]

    result = {"name": system.name, "system": vars(system), "code": git_version(),
              "date": datetime.datetime.now().isoformat(timespec="seconds"),
              "saa_dev": saa, "speechocean_val": speechocean, "synthetic": synthetic,
              "targets_on_saa_dev": targets, "z_to_s_catch": catch(units["saa_dev_z_to_s"]),
              "pattern_catch_on_saa_dev": {name: catch(units[f"saa_dev_{name}"]) for name in evaluate_saa.PATTERNS},
              "units": units}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{system.name}.json"
    out.write_text(json.dumps(result, indent=2, default=float))

    print(f"\nSystem {system.name} (code {result['code']})\n")
    for group in ("english", "turkish"):
        print(format_detection(f"SAA {group}", saa["main"][group]))
    print(format_detection("SO val", speechocean["expert_below_10"]))
    for engine, table in synthetic.items():
        total = table[-1]
        print(f"synthetic {engine}: catch {total['catch']:.1%}, right tip {total['right_tip']:.1%}, "
              f"false alarm {total['false_alarm']:.1%}")
    print("\nTargets on SAA dev (aim above them: v1 lost about 9 points from dev to test):")
    for t in targets:
        print(f"  {t['target']:22} {t['value']:6.1%}  goal {t['goal']:7}  {'met' if t['met'] else 'not met'}")
    print("Error-level catch per pattern (Turkish speakers, SAA dev): " + ", ".join(
        f"{name} {value:.0%}" for name, value in result["pattern_catch_on_saa_dev"].items()))
    for group, bias in saa["canonical_bias"].items():
        print(f"Canonical bias, SAA dev {group}: PER vs expert {bias['per_vs_heard']:.1%}, "
              f"vs expected {bias['per_vs_expected']:.1%} (gap {bias['gap']:+.1%})")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
