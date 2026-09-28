"""Text -> the phonemes a (US English) speaker is expected to say, word by word.

Uses phonemizer with the eSpeak NG backend, the same phoneme style the recognizer
was trained on. Each word is phonemized on its own so that phonemes map 1:1 to words
(in sentence mode eSpeak merges words like "on the" into one group).
"""

from __future__ import annotations

import os
import platform
import re
from functools import lru_cache

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")

# Homebrew puts the library here; phonemizer does not always find it on macOS.
MAC_LIBRARY_PATHS = ("/opt/homebrew/lib/libespeak-ng.dylib", "/usr/local/lib/libespeak-ng.dylib")


def tokenize(text: str) -> list[str]:
    """Split a sentence into words, dropping punctuation."""
    return [w.replace("’", "'") for w in WORD_RE.findall(text)]


def _configure_macos_library() -> None:
    if platform.system() != "Darwin" or os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    for path in MAC_LIBRARY_PATHS:
        if os.path.exists(path):
            EspeakWrapper.set_library(path)
            return


@lru_cache(maxsize=1)
def _backend(language: str):
    _configure_macos_library()
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(language, preserve_punctuation=False, with_stress=False)


@lru_cache(maxsize=4096)
def _phonemize_word(word: str, language: str) -> tuple[str, ...]:
    from phonemizer.separator import Separator

    separator = Separator(phone=" ", word="|", syllable="")
    out = _backend(language).phonemize([word], separator=separator, strip=True)
    return tuple(out[0].replace("|", " ").split())


def phonemize_words(words: list[str], language: str = "en-us") -> list[list[str]]:
    """Expected phonemes for each word, e.g. ["think"] -> [["θ", "ɪ", "ŋ", "k"]]."""
    return [list(_phonemize_word(w.lower(), language)) for w in words]
