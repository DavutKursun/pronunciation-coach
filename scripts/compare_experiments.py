"""Compare two experiments (results/experiments/*.json) on the same speakers.

For every metric: the value for A and B, the difference B - A and its 95% interval from a paired
bootstrap over speakers (pronunciation.metrics.paired_bootstrap_diff). An improvement counts as
real only when the interval does not contain zero.

Usage:
    python scripts/compare_experiments.py results/experiments/v1.json results/experiments/v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation.metrics import detection_from_counts, paired_bootstrap_diff  # noqa: E402

# (label, units key in the experiment file, statistic): confusion units are [tp, fp, fn, tn] per
# speaker, catch units [found, errors] per speaker
METRICS = [
    ("SAA dev, Turkish: recall", "saa_dev_turkish", "recall"),
    ("SAA dev, Turkish: precision", "saa_dev_turkish", "precision"),
    ("SAA dev, Turkish: F1", "saa_dev_turkish", "f1"),
    ("SAA dev, Turkish: false alarm", "saa_dev_turkish", "false_alarm"),
    ("SAA dev, Turkish: error-level catch", "saa_dev_turkish_errors", "catch"),
    ("SAA dev, Turkish: final devoicing catch", "saa_dev_final_voicing", "catch"),
    ("SAA dev, Turkish: z → s catch", "saa_dev_z_to_s", "catch"),
    ("SAA dev, Turkish: ð catch", "saa_dev_th_voiced", "catch"),
    ("SAA dev, Turkish: θ catch", "saa_dev_th_voiceless", "catch"),
    ("SAA dev, English: false alarm", "saa_dev_english", "false_alarm"),
    ("speechocean val: recall", "speechocean_val", "recall"),
    ("speechocean val: precision", "speechocean_val", "precision"),
    ("speechocean val: false alarm", "speechocean_val", "false_alarm"),
    ("speechocean val: F1", "speechocean_val", "f1"),
]


def statistic(name: str):
    if name == "catch":
        return lambda units: sum(u[0] for u in units) / max(sum(u[1] for u in units), 1)
    return lambda units: detection_from_counts(*np.sum(units, axis=0))[name]


def compare(a: dict, b: dict, n_resamples: int = 2000) -> list[dict]:
    rows = []
    for label, key, name in METRICS:
        units_a, units_b = a["units"].get(key), b["units"].get(key)
        if not units_a or not units_b:
            continue
        speakers = sorted(set(units_a) & set(units_b))
        result = paired_bootstrap_diff([units_a[s] for s in speakers], [units_b[s] for s in speakers],
                                       statistic(name), n_resamples=n_resamples)
        rows.append({"metric": label, "speakers": len(speakers), **result})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    a, b = json.loads(args.a.read_text()), json.loads(args.b.read_text())

    print(f"| Metric | {a['name']} | {b['name']} | Difference | 95% CI | Real? |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in compare(a, b):
        low, high = r["ci95"]
        print(f"| {r['metric']} | {r['a']:.1%} | {r['b']:.1%} | {r['diff']:+.1%} | [{low:+.1%}, {high:+.1%}] | "
              f"{'yes' if r['real'] else 'no'} |")


if __name__ == "__main__":
    main()
