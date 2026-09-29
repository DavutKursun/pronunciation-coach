import pytest

from pronunciation.l2arctic import (PhoneLabel, build_target, load_speakers, parse_phone_label, parse_textgrid,
                                    perceived_to_ipa, word_annotations)

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.2
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.2
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.6
            text = "these"
        intervals [3]:
            xmin = 0.6
            xmax = 1.2
            text = "things"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.2
        intervals: size = 9
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = 0.1
            xmax = 0.25
            text = "DH,D,s "
        intervals [3]:
            xmin = 0.25
            xmax = 0.4
            text = "IY1"
        intervals [4]:
            xmin = 0.4
            xmax = 0.6
            text = "Z, S, s"
        intervals [5]:
            xmin = 0.6
            xmax = 0.7
            text = "TH"
        intervals [6]:
            xmin = 0.7
            xmax = 0.8
            text = "IH1,IY,s"
        intervals [7]:
            xmin = 0.8
            xmax = 0.9
            text = "NG"
        intervals [8]:
            xmin = 0.9
            xmax = 1.0
            text = "sil,G,a"
        intervals [9]:
            xmin = 1.0
            xmax = 1.2
            text = "Z,sil,d"
'''


@pytest.mark.parametrize("label, parsed", [
    ("IY1", PhoneLabel("IY1", "IY1", "correct")),
    ("DH,D,s ", PhoneLabel("DH", "D", "sub")),               # trailing space in the corpus
    ("Z, S, s", PhoneLabel("Z", "S", "sub")),                # spaces after commas
    ("P,P*,s", PhoneLabel("P", "P", "correct", accented=True)),   # "*" = an accented P: still a P
    ("AH0,AO*,s", PhoneLabel("AH0", "AO", "sub", accented=True)),
    ("R,R*,s", PhoneLabel("R", "R*", "sub", accented=True)),      # an accented R is a tapped/trilled r
    ("sil,R*,a", PhoneLabel(None, "R*", "ins", accented=True)),
    ("Z,sil,d", PhoneLabel("Z", None, "del")),
    ("sil,G,a", PhoneLabel(None, "G", "ins")),
    ("R,err,s", PhoneLabel("R", None, "sub", unsure=True)),
    ("sil,err,a", PhoneLabel(None, None, "ins", unsure=True)),
])
def test_parse_phone_label(label, parsed):
    assert parse_phone_label(label) == parsed


@pytest.mark.parametrize("label", ["sil", "sp", "", "spn"])
def test_silence_is_not_a_phone(label):
    assert parse_phone_label(label) is None


@pytest.mark.parametrize("perceived, canonical, ipa", [
    ("AH", "AH0", "ə"), ("AH", "AH1", "ʌ"), ("AH", "IH0", "ə"), ("AH", "EH1", "ʌ"),
    ("AH", None, "ə"),                                        # an added vowel is a schwa
    ("ER", "ER0", "ɚ"), ("ER", "ER1", "ɜː"), ("ER", None, "ɚ"),
    ("AX", "IH0", "ə"), ("IY", "IH1", "iː"), ("S", "Z", "s"),
    ("R*", "R", "ɾ"), ("R*", None, "ɾ"),                     # like the SAA experts write Turkish r
])
def test_perceived_vowels_take_the_stress_of_the_expected_vowel(perceived, canonical, ipa):
    assert perceived_to_ipa(perceived, canonical) == ipa


def test_parse_textgrid_and_group_phones_by_word():
    tiers = parse_textgrid(TEXTGRID)
    assert [w[2] for w in tiers["words"]] == ["", "these", "things"]
    words = word_annotations(tiers["words"], tiers["phones"])
    assert [w.text for w in words] == ["these", "things"]
    assert [p.kind for p in words[0].phones] == ["sub", "correct", "sub"]
    assert [(p.canonical, p.perceived, p.kind) for p in words[1].phones] == [
        ("TH", "TH", "correct"), ("IH1", "IY", "sub"), ("NG", "NG", "correct"), (None, "G", "ins"), ("Z", None, "del")]


def test_build_target_substitution_deletion_insertion():
    words = word_annotations(*parse_textgrid(TEXTGRID).values())
    espeak = [["ð", "iː", "z"], ["θ", "ɪ", "ŋ", "z"]]
    target = build_target(words, espeak)
    assert target.ok
    assert target.tokens == ["d", "iː", "s", "θ", "iː", "ŋ", "ɡ"]           # ð→d, z→s; ɪ→iː, +g, z deleted
    assert target.canonical == ["ð", "iː", "z", "θ", "ɪ", "ŋ", "z"]
    assert [(e.kind, e.expected, e.heard) for e in target.errors] == [
        ("sub", "ð", "d"), ("sub", "z", "s"), ("sub", "ɪ", "iː"), ("ins", None, "ɡ"), ("del", "z", None)]


def test_build_target_keeps_espeak_tokens_where_nothing_changed():
    # "store": eSpeak writes the r-coloured vowel as one token; CMU has S T AO1 R
    words = word_annotations([(0, 1, "store")], [(0, .2, "S"), (.2, .4, "T"), (.4, .7, "AO1"), (.7, 1, "R")])
    target = build_target(words, [["s", "t", "ɔːɹ"]])
    assert target.tokens == ["s", "t", "ɔːɹ"] and target.errors == []
    # the r dropped: the token is split and only its vowel part stays
    words = word_annotations([(0, 1, "store")], [(0, .2, "S"), (.2, .4, "T"), (.4, .7, "AO1"), (.7, 1, "R,sil,d")])
    target = build_target(words, [["s", "t", "ɔːɹ"]])
    assert target.tokens == ["s", "t", "oː"]
    assert target.canonical == ["s", "t", "ɔːɹ"]


def test_build_target_accented_sound_is_not_an_error():
    words = word_annotations([(0, 1, "pen")], [(0, .3, "P,P*,s"), (.3, .6, "EH1"), (.6, 1, "N")])
    target = build_target(words, [["p", "ɛ", "n"]])
    assert target.tokens == ["p", "ɛ", "n"] and target.errors == [] and target.accented == 1


def test_build_target_accented_r_is_a_tapped_r():
    # otherwise the model would learn to hear a Turkish r as English ɹ (the v2-2 trap, for r)
    words = word_annotations([(0, 1, "red")], [(0, .3, "R,R*,s"), (.3, .6, "EH1"), (.6, 1, "D")])
    target = build_target(words, [["ɹ", "ɛ", "d"]])
    assert target.tokens == ["ɾ", "ɛ", "d"]
    assert [(e.kind, e.expected, e.heard, e.tip) for e in target.errors] == [("sub", "ɹ", "ɾ", "r")]


def test_build_target_rejects_unsure_or_unreliable_sentences():
    words = word_annotations([(0, 1, "red")], [(0, .3, "R,err,s"), (.3, .6, "EH1"), (.6, 1, "D")])
    target = build_target(words, [["ɹ", "ɛ", "d"]])
    assert not target.ok and target.reason == "err label"
    # a consonant error that lands on a vowel of ours cannot be placed reliably
    words = word_annotations([(0, 1, "a")], [(0, 1, "D,T,s")])
    target = build_target(words, [["ɐ"]])
    assert not target.ok and target.reason == "unreliable alignment"
    # the word lists do not line up
    target = build_target(words, [["ɐ"], ["b"]])
    assert not target.ok and target.reason == "word mismatch"


def test_locked_test_speakers():
    split = {"train": ["ABA"], "dev": ["BWC"], "test": ["NJS"]}
    assert load_speakers("dev", split) == ["BWC"]
    with pytest.raises(PermissionError):
        load_speakers("test", split)
    assert load_speakers("test", split, final=True) == ["NJS"]
