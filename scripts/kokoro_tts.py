"""Synthesize carrier sentences with Kokoro-82M (Apache-2.0) for scripts/synthetic_errors.py.

Runs in its own environment (.venv-tts): Kokoro's G2P, misaki[en], installs phonemizer-fork,
which would replace the phonemizer package of the main project.

    .venv-tts/bin/python scripts/kokoro_tts.py data/synthetic/kokoro_jobs.json

Each job says "Please say <word> again."; for an error version the word's misaki phonemes are
edited with the job's (regex, replacement) rules and passed as an override: [think](/tˈɪŋk/).
The phonemes used are written back into the jobs file.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24_000


def main() -> None:
    jobs_file = Path(sys.argv[1])
    jobs = json.loads(jobs_file.read_text())
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    for k, job in enumerate(jobs):
        phonemes, _ = pipeline.g2p(job["word"])
        for pattern, replacement in job["rules"]:
            phonemes = re.sub(pattern, replacement, phonemes)
        job["phonemes"] = phonemes
        if job["skip"]:
            continue
        text = f"{job['prefix']} [{job['word']}](/{phonemes}/) {job['suffix']}"
        audio = np.concatenate([r.audio.numpy() for r in pipeline(text, voice=job["voice"], speed=job["speed"])])
        Path(job["path"]).parent.mkdir(parents=True, exist_ok=True)
        sf.write(job["path"], audio, SAMPLE_RATE)
        if (k + 1) % 100 == 0:
            print(f"kokoro {k + 1}/{len(jobs)}")
    jobs_file.write_text(json.dumps(jobs))


if __name__ == "__main__":
    main()
