"""Download Turkish and US English speakers from the Speech Accent Archive (George Mason University).

Every speaker reads the same paragraph ("Please call Stella..."), and many recordings come with
a narrow IPA transcription made by trained phoneticians. The archive is on OSF under
CC BY-NC-SA 4.0 (https://accent.gmu.edu/download): non-commercial use with attribution.
Recordings go to data/saa/, which is ignored by git: never commit them.

Downloaded:
  - every native Turkish speaker (audio, and the transcription when there is a text one;
    some only have a transcription image, which we cannot use)
  - 20 US native English speakers with a text transcription, adults, 10 female and 10 male,
    picked with a fixed seed, for comparison

Usage:
    python scripts/download_saa.py
"""

from __future__ import annotations

import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "saa"
OSF_NODE = "yh23d"
SPEAKER_INFO_URL = "https://osf.io/download/9tupa/"
FOLDERS = {"mp3": "698a8f0a9a43a780c6c72b76", "txt": "6a7dfe089137a29d1100226a"}
N_ENGLISH = 20
SEED = 762


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def osf_files(folder: str, name_filter: str) -> dict[str, str]:
    """File name -> download URL for the files in an OSF folder whose name contains `name_filter`."""
    query = urllib.parse.urlencode({"filter[name]": name_filter, "page[size]": 100})
    url = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/{folder}/?{query}"
    files = {}
    while url:
        page = json.loads(fetch(url))
        files.update({f["attributes"]["name"]: f["links"]["download"] for f in page["data"]})
        url = page["links"].get("next")
    return files


def pick_english(info: pd.DataFrame, transcribed: set[str]) -> pd.DataFrame:
    english = info[(info.native_language == "english") & (info.country == "usa") & (info.age >= 18)]
    english = english[english.speech_sample.str.replace(".mp3", ".txt").isin(transcribed)]
    rng = random.Random(SEED)
    picked = []
    for gender in ("female", "male"):
        names = sorted(english[english.gender == gender].speech_sample)
        picked += rng.sample(names, N_ENGLISH // 2)
    return english[english.speech_sample.isin(picked)]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info_path = OUT_DIR / "speaker_information.xlsx"
    if not info_path.exists():
        info_path.write_bytes(fetch(SPEAKER_INFO_URL))
    info = pd.read_excel(info_path)
    for column in ("native_language", "country", "gender"):
        info[column] = info[column].str.strip().str.lower()

    txt_files = osf_files(FOLDERS["txt"], "turkish") | osf_files(FOLDERS["txt"], "english")
    turkish = info[info.native_language == "turkish"]
    english = pick_english(info, set(txt_files))
    speakers = pd.concat([turkish.assign(group="turkish"), english.assign(group="english")])
    speakers["speaker"] = speakers.speech_sample.str.replace(".mp3", "", regex=False)
    speakers["has_transcription"] = (speakers.speaker + ".txt").isin(txt_files)

    mp3_files = osf_files(FOLDERS["mp3"], "turkish") | osf_files(FOLDERS["mp3"], "english")
    for k, speaker in enumerate(speakers.speaker, 1):
        for ext, files in (("mp3", mp3_files), ("txt", txt_files)):
            name = f"{speaker}.{ext}"
            path = OUT_DIR / name
            if name in files and not path.exists():
                path.write_bytes(fetch(files[name]))
        print(f"{k}/{len(speakers)} {speaker}")

    columns = ["speaker", "group", "has_transcription", "age", "gender", "country", "onset_age",
               "english_residence", "length_of_residence"]
    speakers[columns].to_csv(OUT_DIR / "speakers.csv", index=False)
    counts = speakers.groupby("group").has_transcription.agg(["size", "sum"])
    print(f"saved to {OUT_DIR.relative_to(ROOT)}/  (speakers, with a text transcription)")
    print(counts.rename(columns={"size": "speakers", "sum": "transcribed"}).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
