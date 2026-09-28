import pytest

from pronunciation.saa import PARAGRAPH, align_words, expert_labels, narrow_to_broad, parse_transcription


@pytest.mark.parametrize("narrow, broad", [
    ("pʰliz", ["p", "l", "i", "z"]),                 # aspiration is not a different phoneme
    ("d̪ɪs", ["d", "ɪ", "s"]),                        # dental d (Turkish "these") is still a d
    ("slæb̥z̥", ["s", "l", "æ", "b", "z"]),           # partial devoicing: natives do it too
    ("slæps", ["s", "l", "æ", "p", "s"]),            # full devoicing is written as p
    ("bɹɪ̃ŋ", ["b", "ɹ", "ɪ", "ŋ"]),                  # nasalized vowel
    ("kʰɑlˠ", ["k", "ɑː", "l"]),                     # velarized l; bare ɑ is our long ɑː
    ("snæk˺", ["s", "n", "æ", "k"]),                 # unreleased stop, two ways of writing it
    ("bɪɡ̚", ["b", "ɪ", "ɡ"]),
    ("ʧiz", ["tʃ", "i", "z"]),                       # the "tesh" ligature
    ("ðiːz", ["ð", "i", "z"]),                       # length: long and short i compare the same way
    ("ðiiːz", ["ð", "i", "z"]),                      # a doubled vowel letter is one vowel
    ("meɪbi", ["m", "eɪ", "b", "i"]),                # diphthongs stay one phoneme
    ("snoʊ", ["s", "n", "oʊ"]),
    ("hɝ", ["h", "ɜː"]),                             # r-coloured vowels
    ("hɜɹ", ["h", "ɜː"]),
    ("bɹʌðəɹ", ["b", "ɹ", "ʌ", "ð", "ɚ"]),
    ("fɔ˞", ["f", "ɔ", "ɹ"]),
    ("stoəɹ", ["s", "t", "o", "ɹ"]),                 # schwa glide before r
    ("r̆ɛd", ["r", "ɛ", "d"]),                        # Turkish tapped/trilled r stays r
    ("βi", ["β", "i"]),
    ("blʉ", ["b", "l", "u"]),                        # fronted American u
    ("pə̆liːz", ["p", "ə", "l", "i", "z"]),           # an extra-short vowel is still a vowel
    ("n̩", ["ə", "n"]),                               # syllabic n, as in our SPLITS
])
def test_narrow_to_broad(narrow, broad):
    assert narrow_to_broad(narrow) == broad


def test_parse_transcription_skips_the_header():
    text = "Turkish speaker 1\n\n[pliz kol stɛla]\n"
    assert parse_transcription(text) == ["pliz", "kol", "stɛla"]


def test_paragraph_has_69_words():
    assert len(PARAGRAPH.split()) == 69


def test_align_words_handles_repeats_and_skips():
    # the speaker repeats "call" and skips "ask"
    expected = [["p", "l", "iː", "z"], ["k", "ɔː", "l"], ["s", "t", "ɛ", "l", "ə"], ["æ", "s", "k"], ["h", "ɜː"]]
    heard = [["p", "l", "i", "z"], ["k", "o", "l"], ["k", "o", "l"], ["s", "t", "ɛ", "l", "a"], ["h", "ɜː"]]
    matched = align_words(expected, heard, [False, False, False, False, True])
    assert matched == [heard[0], heard[1], heard[3], None, heard[4]]


def test_expert_labels_find_the_turkish_patterns():
    words = ["these", "things", "we"]
    expected = [["ð", "iː", "z"], ["θ", "ɪ", "ŋ", "z"], ["w", "iː"]]
    expert = [["d", "i", "s"], ["t", "ɪ", "ŋ", "s"], ["w", "i"]]
    labels = expert_labels(words, expected, expert)
    assert [label.wrong for label in labels] == [True, True, False]
    assert "th_voiced" in labels[0].tips and "th_voiceless" in labels[1].tips
    assert labels[2].tips == set()


def test_accepted_variants_are_not_expert_errors():
    labels = expert_labels(["to", "better"], [["t", "uː"], ["b", "ɛ", "t", "ɚ"]], [["ɾ", "ə"], ["b", "ɛ", "ɾ", "ɚ"]])
    assert not any(label.wrong for label in labels)


def test_expert_labels_keep_every_error():
    [label] = expert_labels(["things"], [["θ", "ɪ", "ŋ", "z"]], [["t", "i", "ŋ", "s"]])
    assert [(e.expected, e.heard) for e in label.errors] == [("θ", "t"), ("ɪ", "i"), ("z", "s")]
    assert label.pairs == ["θ → t", "ɪ → i", "z → s"]
