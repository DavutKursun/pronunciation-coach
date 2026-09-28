"""Cache the recognizer's output (log-probabilities) per model and data set.

Running wav2vec2 is the slow part; everything after it (decoding, alignment, feedback, a learned
detector) takes milliseconds. With the model output cached, a new decision rule can be tried on
thousands of recordings in seconds. Each model gets its own folder, so a fine-tuned recognizer
never mixes with the original one:

    data/cache/<model>/<dataset>.npz    log-probabilities, one array per item
    data/cache/<model>/<dataset>.json   vocabulary, blank id, audio duration per item
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import SAMPLE_RATE
from .recognizer import Decoder

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def model_slug(model_id: str) -> str:
    """A folder name for a model id or path: "facebook/wav2vec2-..." -> "facebook_wav2vec2-..."."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def cached_log_probs(model_id: str, dataset: str, audio: dict[str, Callable[[], np.ndarray]],
                     cache_dir: Path = CACHE_DIR, recognizer_factory=None):
    """Decoder, log-probabilities and durations for every item of `audio` (key -> audio loader).

    Only items missing from the cache are loaded and run through the model.
    """
    folder = Path(cache_dir) / model_slug(model_id)
    npz_path, meta_path = folder / f"{dataset}.npz", folder / f"{dataset}.json"
    arrays = dict(np.load(npz_path)) if npz_path.exists() else {}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"seconds": {}}

    missing = [key for key in audio if key not in arrays]
    if missing:
        if recognizer_factory is None:
            from .recognizer import PhonemeRecognizer as recognizer_factory
        print(f"Running {model_id} on {len(missing)} {dataset} items (cached afterwards)...")
        recognizer = recognizer_factory(model_id)
        for key in missing:
            samples = audio[key]()
            arrays[key] = recognizer.log_probs(samples)
            meta["seconds"][key] = len(samples) / SAMPLE_RATE
        meta.update(vocab=recognizer.token_to_id, blank_id=recognizer.blank_id,
                    special_ids=sorted(recognizer.decoder.special_ids))
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        meta_path.write_text(json.dumps(meta))

    decoder = Decoder(meta["vocab"], meta["blank_id"], set(meta["special_ids"]))
    return decoder, {k: arrays[k] for k in audio}, {k: meta["seconds"][k] for k in audio}
