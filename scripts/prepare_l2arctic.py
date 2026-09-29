"""Extract the hand-annotated L2-ARCTIC sentences: 16 kHz FLAC + annotation, one speaker at a time.

The release is one zip (l2arctic_release_v5.0.zip, 7.5 GB) with a zip per speaker inside. The
release zip is only read, never changed (it may live on a synced cloud drive). Each speaker's zip
is copied to a temporary folder, only the ~150 annotated sentences are converted (44.1 kHz WAV ->
16 kHz FLAC) into data/l2arctic/<speaker>/, and the temporary copy is deleted right away.
Speakers that are already done are skipped, so the script can be re-run after an interruption.

L2-ARCTIC is CC BY-NC 4.0 (Zhao et al., Interspeech 2018). No audio goes into git.

Usage:
    python scripts/prepare_l2arctic.py --release "/path/to/l2arctic_release_v5.0.zip"
    python scripts/prepare_l2arctic.py --release ... --speakers ABA        # try one speaker first
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pronunciation.audio import SAMPLE_RATE, to_model_input  # noqa: E402
from pronunciation.l2arctic import SPEAKERS  # noqa: E402

OUT_DIR = ROOT / "data" / "l2arctic"


def prepare_speaker(release: zipfile.ZipFile, speaker: str, tmp_dir: Path) -> int:
    """Convert one speaker's annotated sentences; returns how many were written."""
    out = OUT_DIR / speaker
    copy = tmp_dir / f"{speaker}.zip"
    try:
        with release.open(f"{speaker}.zip") as src, open(copy, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        inner = zipfile.ZipFile(copy)
        annotations = sorted(n for n in inner.namelist() if "/annotation/" in n and n.endswith(".TextGrid"))
        (out / "annotation").mkdir(parents=True, exist_ok=True)
        for name in annotations:
            utt = Path(name).stem
            audio, rate = sf.read(io.BytesIO(inner.read(f"{speaker}/wav/{utt}.wav")), dtype="int16")
            sf.write(out / f"{utt}.flac", to_model_input(audio, rate), SAMPLE_RATE, subtype="PCM_16")
            (out / "annotation" / f"{utt}.TextGrid").write_bytes(inner.read(name))
            transcript = f"{speaker}/transcript/{utt}.txt"
            if transcript in inner.namelist():
                (out / f"{utt}.txt").write_bytes(inner.read(transcript))
        inner.close()
        (out / "DONE").write_text(f"{len(annotations)}\n")
        return len(annotations)
    finally:
        copy.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, required=True, help="l2arctic_release_v5.0.zip (only read)")
    parser.add_argument("--speakers", nargs="+", default=sorted(SPEAKERS))
    parser.add_argument("--tmp", type=Path, help="where the per-speaker zip is copied (default: system temp)")
    args = parser.parse_args()

    release = zipfile.ZipFile(args.release)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "README.md", "PROMPTS"):
        if not (OUT_DIR / name).exists():
            (OUT_DIR / name).write_bytes(release.read(name))
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        for speaker in args.speakers:
            if (OUT_DIR / speaker / "DONE").exists():
                print(f"{speaker}: already done")
                continue
            start = time.time()
            try:
                n = prepare_speaker(release, speaker, Path(tmp))
            except OSError as error:   # e.g. the cloud drive refuses the download (quota)
                print(f"{speaker}: stopped, could not read the release: {error}")
                return 1
            size = sum(f.stat().st_size for f in (OUT_DIR / speaker).rglob("*")) / 1e6
            print(f"{speaker}: {n} annotated sentences, {size:.0f} MB, {time.time() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
