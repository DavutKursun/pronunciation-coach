"""Phoneme inventory, what counts as an acceptable variant, and error patterns of Turkish speakers.

Phonemes are IPA symbols in the eSpeak NG style, which is what both the G2P step
(phonemizer + eSpeak) and the recognizer (wav2vec2-lv-60-espeak-cv-ft) produce.
"""

from __future__ import annotations

VOWEL_CHARS = set("aeiouyæɑɒɐɔəɚɛɜɪʊʌøœɤɯɨʉᵻ")

# Multi-symbol phonemes are split so that "car" (k ɑːɹ) and a recognizer output of
# "k ɑː ɹ" compare equal. The split forms are only used for comparison.
# The vowel before /r/ in "here", "sure", "there" becomes a centring diphthong (iə, ʊə, eə):
# English has no ship/sheep or pull/pool contrast before /r/, so ACCEPT is lenient on them.
# "iə" (zero, really) stays one vowel: speakers say it as one sound, not "i" + "ə".
SPLITS = {
    "ɑːɹ": ["ɑː", "ɹ"],
    "ɔːɹ": ["oː", "ɹ"],     # store, for: no ɔ/o contrast before r
    "oːɹ": ["oː", "ɹ"],
    "ɛɹ": ["eə", "ɹ"],
    "ɪɹ": ["iə", "ɹ"],
    "ʊɹ": ["ʊə", "ɹ"],
    "aɪɚ": ["aɪ", "ɚ"],
    "aɪə": ["aɪ", "ə"],
    "əl": ["ə", "l"],
    "n̩": ["ə", "n"],
    "l̩": ["ə", "l"],
}

# Different Unicode spellings of the same symbol.
ALIASES = {"g": "ɡ", "r̩": "ɹ", "ɝ": "ɜː", "ɫ": "l", "ɻ": "ɹ"}

# ACCEPT[expected] = realizations that are NOT errors (accent/transcription variants).
# Deliberately directional: expected /i/ (the final vowel of "happy") may be heard as /ɪ/,
# but expected /ɪ/ (ship) heard as /iː/ (sheep) is a real error.
ACCEPT: dict[str, set[str]] = {
    "t": {"ɾ", "ʔ"},        # American flap: "better", glottal stop: "button"
    "d": {"ɾ"},
    "ɾ": {"t", "d"},
    "ʔ": {"t"},
    "i": {"iː", "ɪ"},
    "iː": {"i"},
    "ə": {"ɐ", "ᵻ", "ʌ", "ɪ"},
    "ɐ": {"ə", "ʌ"},
    "ᵻ": {"ɪ", "ə", "i"},
    "ʌ": {"ə", "ɐ"},
    "ɚ": {"ɜː", "ə", "ɐ"},
    "ɜː": {"ɚ"},
    "ɑː": {"ɔː", "ɑ", "ɒ", "ɔ", "a"},    # American English has no /a/ vs /ɑ/ contrast
    "ɔː": {"ɑː", "ɔ", "ɒ", "oː", "a"},
    "ɔ": {"ɔː", "ɑː", "ɒ"},
    "oʊ": {"o", "oː", "əʊ"},
    "oː": {"oʊ", "ɔː", "o", "ɔ"},
    "eɪ": {"e", "eː"},
    "ɛ": {"e"},
    "uː": {"u"},
    "u": {"uː"},
    "iə": {"iː", "ɪ", "i", "ɪə"},                 # here, zero
    "ʊə": {"ʊ", "uː", "u", "ɔː", "oː", "o"},      # sure, tour
    "eə": {"ɛ", "e", "eɪ", "ɛː", "æ"},            # there, care
}

REDUCED_VOWELS = {"ə", "ɪ", "ᵻ", "ɐ", "ɚ"}

# In connected speech these words are usually said with a reduced vowel
# ("for" -> /fɚ/, "to" -> /tə/), so any reduced vowel is accepted in them.
FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "and", "for", "from", "at", "as", "but", "or", "nor",
    "can", "could", "would", "should", "will", "shall", "must", "was", "were", "are",
    "am", "is", "be", "been", "have", "has", "had", "do", "does", "you", "your", "he",
    "she", "we", "they", "them", "him", "his", "her", "our", "their", "that", "than",
    "then", "there", "some", "just", "into", "onto", "with", "by", "in", "on", "it",
}

# Weak forms: how English speakers say these function words in connected speech (checked on the
# US English speakers of the Speech Accent Archive dev half). For each expected sound, the extra
# realizations that are fine in this word; None means the sound may be left out.
_R_DROP = {"ɹ": {None}}      # "for" -> fɚ / fə: the r is part of the r-coloured vowel or dropped
_H_DROP = {"h": {None}}      # "ask her" -> "ask 'er"
_BACK_VOWEL = {"ʌ": {"ɔ", "ɑː", "ɒ"}}
WEAK_FORMS: dict[str, dict[str, set[str | None]]] = {
    "and": {"æ": {"ɛ"}, "d": {None}},          # æn, ɛn, ən, n̩
    "with": {"ð": {"θ"}},                      # wɪθ is as common as wɪð
    "of": _BACK_VOWEL, "from": _BACK_VOWEL,    # ɔv, fɹɑm
    **{w: _R_DROP for w in ("for", "or", "nor", "are", "were", "your", "our", "their", "there")},
    **{w: _H_DROP for w in ("her", "him", "his", "he", "have", "has", "had")},
}

# voiced -> voiceless pairs (Turkish devoices consonants at the end of words: kitap/kitabı)
DEVOICED = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s", "ð": "θ", "ʒ": "ʃ", "dʒ": "tʃ"}

TIPS: dict[str, dict[str, str]] = {
    "th_voiceless": {
        "title": "'th' as in think /θ/",
        "en": "Put the tip of your tongue lightly between your teeth and blow air out. Don't stop the air like 't' or hiss like 's'.",
        "tr": "Dilinin ucunu hafifçe dişlerinin arasına koy ve hava üfle. 't' gibi havayı kesme, 's' gibi tıslama.",
    },
    "th_voiced": {
        "title": "'th' as in this /ð/",
        "en": "Same tongue position as in 'think', but with voice: your throat should vibrate. It is not 'd' or 'z'.",
        "tr": "Dil 'think'teki gibi dişlerin arasında, ama ses tellerin titreşsin. 'd' ya da 'z' değil.",
    },
    "w": {
        "title": "'w' as in west /w/",
        "en": "Round your lips like a quick 'u' and keep your lower lip away from your teeth, otherwise it becomes 'v'.",
        "tr": "Dudaklarını hızlı bir 'u' der gibi yuvarla; alt dudağın dişlerine değmesin, yoksa 'v' olur.",
    },
    "v": {
        "title": "'v' as in very /v/",
        "en": "Touch your upper teeth to your lower lip and add voice.",
        "tr": "Üst dişlerini alt dudağına değdir ve ses ver.",
    },
    "long_ee": {
        "title": "long 'ee' as in sheep /iː/",
        "en": "Make it long and tense with slightly spread lips: 'sheep', not 'ship'.",
        "tr": "Sesi uzat ve gergin tut, dudakların hafif yana gerilsin: 'ship' değil 'sheep'.",
    },
    "short_i": {
        "title": "short 'i' as in ship /ɪ/",
        "en": "Keep it short and relaxed, between Turkish 'i' and 'ı': 'ship', not 'sheep'.",
        "tr": "Kısa ve gevşek söyle, Türkçedeki 'i' ile 'ı' arası bir ses: 'sheep' değil 'ship'.",
    },
    "ae": {
        "title": "'a' as in cat /æ/",
        "en": "Open your mouth wider than for 'e'; it sits between Turkish 'a' and 'e': 'bad', not 'bed'.",
        "tr": "Ağzını 'e' derken olduğundan daha çok aç; Türkçedeki 'a' ile 'e' arası bir ses: 'bed' değil 'bad'.",
    },
    "uh": {
        "title": "'u' as in cup /ʌ/",
        "en": "A short, relaxed 'a' with the mouth only slightly open.",
        "tr": "Kısa ve gevşek bir 'a'; ağzını çok açma.",
    },
    "short_u": {
        "title": "short 'oo' as in pull /ʊ/",
        "en": "Short and relaxed, lips only a little rounded: 'pull', not 'pool'.",
        "tr": "Kısa ve gevşek, dudakların az yuvarlak: 'pool' değil 'pull'.",
    },
    "long_oo": {
        "title": "long 'oo' as in pool /uː/",
        "en": "Long, with well-rounded lips: 'pool', not 'pull'.",
        "tr": "Uzun ve dudakların iyice yuvarlak: 'pull' değil 'pool'.",
    },
    "er": {
        "title": "'er' as in bird /ɜː/",
        "en": "Keep your tongue in the middle of your mouth with a slight 'r' colour; it is not Turkish 'ö' or 'e'.",
        "tr": "Dil ağzın ortasında, hafif bir 'r' tınısıyla; Türkçedeki 'ö' ya da 'e' değil.",
    },
    "ng": {
        "title": "'ng' as in sing /ŋ/",
        "en": "Close with the back of your tongue and don't add a 'g' after it: 'singing', not 'sing-ging'. It is not 'n' either.",
        "tr": "Dilinin arka kısmıyla kapat ve arkasına 'g' ekleme: 'sing-ging' değil 'singing'. 'n' de değil.",
    },
    "r": {
        "title": "English 'r' /ɹ/",
        "en": "Don't tap or roll your tongue. Curl it slightly back without touching the roof of your mouth.",
        "tr": "Dilini damağa vurma ve titretme. Hafifçe geriye kıvır, damağına değdirme.",
    },
    "final_voicing": {
        "title": "voiced sound at the end of a word",
        "en": "Keep the voice on the final sound: 'bed' should not sound like 'bet'. Making the vowel before it a bit longer helps.",
        "tr": "Kelimenin sonundaki sesi sertleştirme: 'bed', 'bet' gibi duyulmasın. Önündeki sesliyi biraz uzatmak yardımcı olur.",
    },
    "epenthesis": {
        "title": "extra vowel in a consonant group",
        "en": "Don't add a vowel before or inside groups like 'sp', 'st', 'sk', 'str': say 'school', not 'is-chool' or 'si-chool'.",
        "tr": "'sp', 'st', 'sk', 'str' gibi ünsüz gruplarının başına ya da arasına sesli ekleme: 'ıskul' ya da 'sıkul' değil, 'school'.",
    },
}

# (expected, heard) -> tip id, for substitutions that are typical of Turkish speakers
SUBSTITUTION_TIPS: dict[tuple[str, str], str] = {}
for heard in ("t", "s", "f", "d", "t̪"):
    SUBSTITUTION_TIPS[("θ", heard)] = "th_voiceless"
for heard in ("d", "z", "v", "θ", "d̪"):
    SUBSTITUTION_TIPS[("ð", heard)] = "th_voiced"
for heard in ("v", "β", "ʋ", "ɹ"):  # Turkish /v/ is often [ʋ], which the recognizer tends to hear as ɹ
    SUBSTITUTION_TIPS[("w", heard)] = "w"
for heard in ("w", "β", "ʋ"):
    SUBSTITUTION_TIPS[("v", heard)] = "v"
SUBSTITUTION_TIPS[("iː", "ɪ")] = "long_ee"
for heard in ("i", "iː"):
    SUBSTITUTION_TIPS[("ɪ", heard)] = "short_i"
for heard in ("e", "ɛ", "a", "ɑ", "aː"):
    SUBSTITUTION_TIPS[("æ", heard)] = "ae"
for heard in ("a", "ɑː", "ɑ", "aː"):
    SUBSTITUTION_TIPS[("ʌ", heard)] = "uh"
for heard in ("uː", "u"):
    SUBSTITUTION_TIPS[("ʊ", heard)] = "short_u"
SUBSTITUTION_TIPS[("uː", "ʊ")] = "long_oo"
for heard in ("ø", "øː", "œ", "ɛ", "e"):
    SUBSTITUTION_TIPS[("ɜː", heard)] = "er"
SUBSTITUTION_TIPS[("ŋ", "n")] = "ng"
for heard in ("r", "ɾ", "ʁ"):
    SUBSTITUTION_TIPS[("ɹ", heard)] = "r"

# Example words so feedback can say "/θ/ as in 'think'" instead of bare IPA.
EXAMPLES = {
    "θ": "think", "ð": "this", "ʃ": "she", "ʒ": "measure", "tʃ": "church", "dʒ": "judge",
    "ŋ": "sing", "ɹ": "red", "j": "yes", "w": "we", "h": "hat", "p": "pen", "b": "bad",
    "t": "tea", "d": "dog", "k": "cat", "ɡ": "go", "f": "fish", "v": "very", "s": "see",
    "z": "zoo", "m": "man", "n": "no", "l": "leg", "æ": "cat", "ɛ": "bed", "ɪ": "sit",
    "iː": "see", "i": "happy", "ʊ": "book", "uː": "food", "ʌ": "cup", "ə": "about",
    "ɜː": "bird", "ɚ": "butter", "ɑː": "father", "ɔː": "law", "eɪ": "day", "aɪ": "my",
    "ɔɪ": "boy", "aʊ": "now", "oʊ": "go", "oː": "more", "iə": "here", "ʊə": "tour",
    "eə": "there",
}

# Everything eSpeak NG en-us says (found by phonemizing 30,000 dictionary words).
ENGLISH_PHONES = set(
    "ɪ n s ə t ɹ l k m d p æ ɛ ɑː ɚ b oʊ f i aɪ eɪ ʌ ɾ ɡ iː əl uː z ᵻ v ʃ ɐ ŋ dʒ h w j ɜː θ ɔː "
    "iə ɔːɹ tʃ ɑːɹ ʊ aʊ ɔ ɔɪ ʊɹ ð ʒ ɪɹ ɛɹ aɪɚ aɪə ʔ n̩ r x ɬ ɔ̃ ɑ̃".split()
)

# Sounds Turkish speakers often use instead of English ones: Turkish vowels (e, a, o, ö, ı, ü),
# long vowels, tapped or rolled r, dental t/d, "w" made with the teeth, palatal k/g, soft g.
LEARNER_PHONES = set("e a o ø œ ɯ y ɨ eː aː ɛː øː r ɾ t̪ d̪ β ʋ c ɟ ɣ ç ɫ".split())

# The recognizer is multilingual: it can output Mandarin tones ("ə1"), aspirated or palatalized
# consonants ("tʰ", "tʲ") and more. Decoding is limited to these phonemes, so it only "hears"
# sounds that English or a Turkish speaker of English would actually use.
ALLOWED_PHONES = (
    ENGLISH_PHONES | LEARNER_PHONES | set(ALIASES)
    | set(SPLITS) | {p for parts in SPLITS.values() for p in parts}
    | set(ACCEPT) | {p for variants in ACCEPT.values() for p in variants}
    | {p for pair in SUBSTITUTION_TIPS for p in pair} | set(DEVOICED.values())
)


def is_vowel(phone: str) -> bool:
    return bool(phone) and phone[0] in VOWEL_CHARS


def normalize(phones: list[str]) -> list[str]:
    """Split multi-symbol phonemes and unify spellings, for comparison."""
    out: list[str] = []
    for p in phones:
        p = ALIASES.get(p, p)
        out.extend(SPLITS.get(p, [p]))
    return out


def merge_repeats(phones: list[str]) -> list[str]:
    """Merge a phoneme repeated inside one word: eSpeak writes "during" as d ʊɹ ɹ ɪ ŋ.

    Only for the expected phonemes of a single word. The heard phonemes of a sentence are
    left alone: "went to" really has two /t/ sounds, one per word.
    """
    return [p for k, p in enumerate(phones) if k == 0 or phones[k - 1] != p]


def is_acceptable(expected: str, heard: str, function_word: bool = False, extra: set = frozenset()) -> bool:
    if expected == heard or heard in ACCEPT.get(expected, ()) or heard in extra:
        return True
    return function_word and is_vowel(expected) and heard in REDUCED_VOWELS


def describe(phone: str) -> str:
    example = EXAMPLES.get(phone)
    return f"/{phone}/ (as in '{example}')" if example else f"/{phone}/"
