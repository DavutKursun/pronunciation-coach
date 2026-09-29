"""Plot the v2-3 fine-tuning curves for the README, in a light and a dark version.

Reads results/finetune/loss.csv (the loss of every optimizer step) and train_log.jsonl (dev
metrics per epoch; epoch 0 is the original model) and writes results/finetune/training_curves.png
and training_curves_dark.png: three small multiples over the same epochs, each with one y axis
(never two scales on one plot):

  training loss                 CTC loss, 50-step moving average
  L2-ARCTIC dev: detection F1   with the original model, the epoch-3 checkpoint and the best one labelled
  phone error rate              L2-ARCTIC dev (against what the annotators heard) and native dev

Colors are the first three slots of the dataviz reference palette, checked with its validator for
both modes (the light aqua is below 3:1 on the surface, so its line carries a direct label).

Usage:
    python scripts/plot_training.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "results" / "finetune"
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "l2": "#2a78d6", "native": "#eb6834", "train": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "l2": "#3987e5", "native": "#d95926", "train": "#199e70"},
}


def load() -> tuple[list[dict], np.ndarray, np.ndarray]:
    epochs = [e for e in map(json.loads, (FOLDER / "train_log.jsonl").read_text().splitlines()) if "epoch" in e]
    rows = list(csv.DictReader((FOLDER / "loss.csv").open()))
    steps = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    return epochs, steps, loss


def style(ax, theme: dict, title: str) -> None:
    ax.set_facecolor(theme["surface"])
    ax.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=10)
    ax.grid(axis="y", color=theme["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0, pad=6)
    ax.set_xlabel("epoch (0 = original model)", color=theme["ink2"], fontsize=9)
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_xlim(-0.6, 17.2)


def line(ax, x, y, color: str, theme: dict, markers: bool = True, label: str | None = None) -> None:
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label,
            marker="o" if markers else None, markersize=5, markeredgewidth=1, markeredgecolor=theme["surface"])
    # markersize 5 pt = 10 px and a 1 pt (2 px) ring in the surface color at 150 dpi


def note(ax, x, y, text: str, theme: dict, dx: float, dy: float, ha: str = "left") -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=8.5, color=theme["ink2"])


def plot(theme_name: str, epochs: list[dict], steps: np.ndarray, loss: np.ndarray) -> Path:
    theme = THEMES[theme_name]
    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=theme["surface"])
    x = [e["epoch"] for e in epochs]
    steps_per_epoch = next(e["step"] for e in epochs if e["epoch"] == 1)

    ax = axes[0]
    style(ax, theme, "Training loss (CTC, 50-step average)")
    smooth = np.convolve(loss, np.ones(50) / 50, mode="valid")
    at = (steps[49:]) / steps_per_epoch
    line(ax, at, smooth, theme["train"], theme, markers=False)
    note(ax, at[-1], smooth[-1], f"{smooth[-1]:.2f}", theme, 6, 0)
    ax.set_ylim(0, max(1.4, smooth.max() * 1.1))

    ax = axes[1]
    style(ax, theme, "L2-ARCTIC dev: detection F1")
    f1 = [e["l2_dev_f1"] for e in epochs]
    line(ax, x, f1, theme["l2"], theme)
    best = max(range(1, len(f1)), key=lambda k: f1[k])
    note(ax, 0, f1[0], f"original {f1[0]:.1%}", theme, 6, -14)
    note(ax, 3, f1[3], f"epoch 3: {f1[3]:.1%}", theme, 8, -12)
    note(ax, best, f1[best], f"best, epoch {best}: {f1[best]:.1%}", theme, 0, 14, ha="center")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.6)

    ax = axes[2]
    style(ax, theme, "Phone error rate")
    for key, color, label in (("l2_dev_per", "l2", "L2-ARCTIC dev (vs what was heard)"),
                              ("native_dev_per", "native", "native US dev")):
        values = [e[key] for e in epochs]
        line(ax, x, values, theme[color], theme, label=label)
        note(ax, x[-1], values[-1], f"{values[-1]:.1%}", theme, 8, 0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.25)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(theme["ink2"])

    fig.tight_layout(w_pad=3)
    out = FOLDER / ("training_curves.png" if theme_name == "light" else "training_curves_dark.png")
    fig.savefig(out, dpi=150, facecolor=theme["surface"])
    plt.close(fig)
    return out


def main() -> None:
    epochs, steps, loss = load()
    for theme in THEMES:
        print(f"Saved {plot(theme, epochs, steps, loss).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
