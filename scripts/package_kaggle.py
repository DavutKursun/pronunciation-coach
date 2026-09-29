"""Pack the fine-tuning data and code into one zip for a private Kaggle dataset.

Contents: 16 kHz FLAC audio of the L2-ARCTIC train and dev speakers and of the CMU ARCTIC native
sentences, the manifests (with audio paths inside the zip), the code the training script needs
(pronunciation/ and scripts/finetune_recognizer.py) and the data licenses. The L2-ARCTIC test
speakers are never packed (they stay locked until v2-4).

Usage:
    python scripts/package_kaggle.py        # -> data/finetune/kaggle/pronunciation-coach-finetune.zip
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from importlib.metadata import version
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation.audio import SAMPLE_RATE, to_model_input  # noqa: E402
from pronunciation.l2arctic import load_speakers  # noqa: E402

DATA = ROOT / "data"
MANIFESTS = DATA / "finetune"
OUT = MANIFESTS / "kaggle" / "pronunciation-coach-finetune.zip"
PARTS = ["l2arctic_train", "l2arctic_dev", "native_train", "native_dev"]
# installed on Kaggle with the versions used here (torch stays Kaggle's own, built for its GPU)
PINNED = ["transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy", "scipy", "soundfile"]
README = """Pronunciation Coach: fine-tuning data for the phoneme recognizer (v2-3)
https://github.com/DavutKursun/pronunciation-coach

manifests/*.jsonl   one sentence per line; "tokens" is the training target, "audio" a path in this folder
audio/              16 kHz mono FLAC
pronunciation/, scripts/finetune_recognizer.py   the training code
requirements-finetune.txt   package versions the code was tested with (torch: Kaggle's own)

Data (see licenses/):
- L2-ARCTIC (Zhao et al., Interspeech 2018), CC BY-NC 4.0: non-commercial use only. Only the
  hand-annotated sentences of the train and dev speakers; the test speakers are not included.
- CMU ARCTIC (Kominek & Black, Carnegie Mellon University): speakers bdl, slt, clb, rms, converted
  to FLAC (a modification, marked as such); see licenses/CMU_ARCTIC_COPYING.
A model fine-tuned on this data is CC BY-NC 4.0.
"""


def packed_path(row: dict) -> str:
    if row["l1"] == "English":
        speaker, name = row["id"].split("/")
        return f"audio/native/{speaker}/{name}.flac"
    return f"audio/{row['audio']}"


def flac_bytes(path: Path) -> bytes:
    audio, rate = sf.read(path, dtype="int16")
    buffer = io.BytesIO()
    sf.write(buffer, to_model_input(audio, rate), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def main() -> None:
    locked = set(load_speakers("test", final=True))    # only to make sure they stay out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(OUT, "w") as z:
        for part in PARTS:
            rows = [json.loads(line) for line in (MANIFESTS / f"{part}.jsonl").read_text().splitlines()]
            if locked & {r["speaker"] for r in rows}:
                raise RuntimeError(f"{part} contains locked test speakers")
            lines = []
            for row in rows:
                source, target = DATA / row["audio"], packed_path(row)
                data = source.read_bytes() if source.suffix == ".flac" else flac_bytes(source)
                z.writestr(target, data, compress_type=zipfile.ZIP_STORED)   # FLAC is already compressed
                lines.append(json.dumps({**row, "audio": target}, ensure_ascii=False))
            z.writestr(f"manifests/{part}.jsonl", "\n".join(lines) + "\n", compress_type=zipfile.ZIP_DEFLATED)
            counts[part] = len(rows)
        for path in sorted((ROOT / "pronunciation").glob("*.py")):
            z.write(path, f"pronunciation/{path.name}", compress_type=zipfile.ZIP_DEFLATED)
        z.write(ROOT / "scripts" / "finetune_recognizer.py", "scripts/finetune_recognizer.py",
                compress_type=zipfile.ZIP_DEFLATED)
        z.write(DATA / "l2arctic" / "LICENSE", "licenses/L2-ARCTIC_LICENSE", compress_type=zipfile.ZIP_DEFLATED)
        z.write(DATA / "cmu_arctic" / "COPYING", "licenses/CMU_ARCTIC_COPYING", compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("README.txt", README, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("requirements-finetune.txt", "".join(f"{p}=={version(p)}\n" for p in PINNED),
                   compress_type=zipfile.ZIP_DEFLATED)
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.0f} MB, sentences {counts}")


if __name__ == "__main__":
    main()
