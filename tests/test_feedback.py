import pytest

from pronunciation.assess import compare
from pronunciation.feedback import top_tips


def run(words, raw_phones, heard):
    return compare(words, raw_phones, heard)


def tips(results):
    return [issue.tip for r in results for issue in r.issues]


def test_perfect_word_has_score_one_and_no_issues():
    [r] = run(["think"], [["θ", "ɪ", "ŋ", "k"]], ["θ", "ɪ", "ŋ", "k"])
    assert r.score == 1.0 and r.issues == []


def test_th_said_as_t():
    [r] = run(["think"], [["θ", "ɪ", "ŋ", "k"]], ["t", "ɪ", "ŋ", "k"])
    assert tips([r]) == ["th_voiceless"]
    assert r.score == 0.75


def test_final_devoicing_bed_as_bet():
    [r] = run(["bed"], [["b", "ɛ", "d"]], ["b", "ɛ", "t"])
    assert tips([r]) == ["final_voicing"]


def test_devoicing_in_the_middle_is_not_final_voicing():
    [r] = run(["idea"], [["aɪ", "d", "iː", "ə"]], ["aɪ", "t", "iː", "ə"])
    assert tips([r]) == [None]


def test_extra_vowel_before_s_cluster():
    [r] = run(["school"], [["s", "k", "uː", "l"]], ["ɪ", "s", "k", "uː", "l"])
    assert tips([r]) == ["epenthesis"]


def test_extra_vowel_inside_cluster():
    [r] = run(["street"], [["s", "t", "ɹ", "iː", "t"]], ["s", "ɪ", "t", "ɹ", "iː", "t"])
    assert tips([r]) == ["epenthesis"]


def test_extra_vowel_between_words_goes_to_the_next_word():
    results = run(["the", "school"], [["ð", "ə"], ["s", "k", "uː", "l"]],
                  ["ð", "ə", "ɪ", "s", "k", "uː", "l"])
    assert results[0].issues == []
    assert tips(results[1:]) == ["epenthesis"]


def test_ng_followed_by_g():
    [r] = run(["sing"], [["s", "ɪ", "ŋ"]], ["s", "ɪ", "ŋ", "ɡ"])
    assert tips([r]) == ["ng"]


def test_rolled_r_and_missing_sound():
    results = run(["red", "car"], [["ɹ", "ɛ", "d"], ["k", "ɑːɹ"]], ["r", "ɛ", "d", "k", "ɑː"])
    assert tips(results[:1]) == ["r"]
    assert [i.kind for i in results[1].issues] == ["del"]


def test_top_tips_are_sorted_by_frequency():
    results = run(["think", "thing", "west"],
                  [["θ", "ɪ", "ŋ", "k"], ["θ", "ɪ", "ŋ"], ["w", "ɛ", "s", "t"]],
                  ["t", "ɪ", "ŋ", "k", "t", "ɪ", "ŋ", "v", "ɛ", "s", "t"])
    assert top_tips(results) == ["th_voiceless", "w"]


# Before /r/ English has no short/long contrast for these vowels (ship/sheep, pull/pool),
# so the phonemes below are the ones eSpeak gives and none of these readings is an error.
@pytest.mark.parametrize("word, raw, heard", [
    ("zero", ["z", "iə", "ɹ", "oʊ"], ["z", "iː", "ɹ", "oʊ"]),
    ("here", ["h", "ɪɹ"], ["h", "iː", "ɹ"]),
    ("sure", ["ʃ", "ʊɹ"], ["ʃ", "uː", "ɹ"]),
])
def test_vowel_before_r_is_not_an_error(word, raw, heard):
    [r] = run([word], [raw], heard)
    assert r.issues == []


def test_really_said_with_one_vowel():
    # eSpeak's "iə" is one vowel for most speakers; speechocean762 speakers say "ɹ ɪ l i"
    [r] = run(["really"], [["ɹ", "iə", "l", "i"]], ["ɹ", "ɪ", "l", "i"])
    assert r.issues == []


@pytest.mark.parametrize("word, raw, heard, tip", [
    ("ship", ["ʃ", "ɪ", "p"], ["ʃ", "iː", "p"], "short_i"),
    ("pull", ["p", "ʊ", "l"], ["p", "uː", "l"], "short_u"),
])
def test_short_vowel_errors_are_still_caught(word, raw, heard, tip):
    [r] = run([word], [raw], heard)
    assert tips([r]) == [tip]


def test_during_has_one_r():
    [r] = run(["during"], [["d", "ʊɹ", "ɹ", "ɪ", "ŋ"]], ["d", "ʊ", "ɹ", "ɪ", "ŋ"])
    assert r.issues == []


# Weak forms of function words: how English speakers say them in connected speech.
@pytest.mark.parametrize("word, raw, heard", [
    ("and", ["æ", "n", "d"], ["æ", "n"]),
    ("and", ["æ", "n", "d"], ["ɛ", "n"]),
    ("her", ["h", "ɜː"], ["ɚ"]),
    ("for", ["f", "ɔːɹ"], ["f", "ɚ"]),
    ("for", ["f", "ɔːɹ"], ["f", "ə"]),
    ("with", ["w", "ɪ", "ð"], ["w", "ɪ", "θ"]),
    ("of", ["ʌ", "v"], ["ɔ", "v"]),
    ("from", ["f", "ɹ", "ʌ", "m"], ["f", "ɹ", "ɑː", "m"]),
])
def test_weak_forms_are_not_errors(word, raw, heard):
    [r] = compare([word], [raw], heard)
    assert r.issues == []


@pytest.mark.parametrize("word, raw, heard", [
    ("hat", ["h", "æ", "t"], ["æ", "t"]),          # only function words may drop h
    ("band", ["b", "æ", "n", "d"], ["b", "æ", "n"]),
    ("with", ["w", "ɪ", "ð"], ["w", "ɪ", "d"]),     # th said as d is still an error
    ("of", ["ʌ", "v"], ["ʌ", "f"]),                 # final devoicing is still an error
])
def test_weak_forms_do_not_hide_real_errors(word, raw, heard):
    [r] = compare([word], [raw], heard)
    assert r.issues


# American English has no /a/ vs /ɑ/ contrast, and no /ɔ/ vs /o/ contrast before r.
@pytest.mark.parametrize("word, raw, heard", [
    ("bob", ["b", "ɑː", "b"], ["b", "a", "b"]),
    ("call", ["k", "ɔː", "l"], ["k", "a", "l"]),
    ("store", ["s", "t", "ɔːɹ"], ["s", "t", "o", "ɹ"]),
    ("store", ["s", "t", "ɔːɹ"], ["s", "t", "ɔ", "ɹ"]),
])
def test_back_vowel_variants_are_not_errors(word, raw, heard):
    [r] = run([word], [raw], heard)
    assert r.issues == []


def test_o_for_the_vowel_of_call_is_still_an_error():
    [r] = run(["call"], [["k", "ɔː", "l"]], ["k", "o", "l"])
    assert r.issues


def test_w_heard_as_r_gets_the_w_tip():
    # Turkish /v/ is often [ʋ], which the recognizer tends to hear as ɹ ("Wednesday" -> "ɹɛnzdeɪ")
    [r] = run(["west"], [["w", "ɛ", "s", "t"]], ["ɹ", "ɛ", "s", "t"])
    assert tips([r]) == ["w"]


# The fine-tuned recognizer (v2-3) outputs ɾ for a tapped r. It must stay two different things:
@pytest.mark.parametrize("word, raw, heard", [
    ("water", ["w", "ɔː", "t", "ɚ"], ["w", "ɔː", "ɾ", "ɚ"]),     # American flap for t: fine
    ("ladder", ["l", "æ", "d", "ɚ"], ["l", "æ", "ɾ", "ɚ"]),       # and for d
])
def test_tap_for_t_or_d_is_the_american_flap(word, raw, heard):
    [r] = run([word], [raw], heard)
    assert r.issues == []


@pytest.mark.parametrize("word, raw, heard", [
    ("red", ["ɹ", "ɛ", "d"], ["ɾ", "ɛ", "d"]),
    ("car", ["k", "ɑːɹ"], ["k", "ɑː", "ɾ"]),                      # r inside eSpeak's r-coloured vowel
])
def test_tap_for_r_is_the_turkish_r_pattern(word, raw, heard):
    [r] = run([word], [raw], heard)
    assert tips([r]) == ["r"]


def test_the_recognizer_may_output_a_tap():
    from pronunciation.phonemes import ALLOWED_PHONES

    assert {"ɾ", "r"} <= ALLOWED_PHONES


def test_is_confirmed_is_the_gop_check_of_one_difference():
    from pronunciation.feedback import Issue, is_confirmed

    th = Issue(0, "sub", "θ", "t", "th_voiceless", "")        # a typical Turkish error
    other = Issue(0, "sub", "m", "n", None, "")               # any other difference
    extra_vowel = Issue(0, "ins", None, "ɪ", "epenthesis", "")
    assert is_confirmed(th, -1.5, -2.5, -1.0) and not is_confirmed(other, -1.5, -2.5, -1.0)
    assert is_confirmed(other, -3.0, -2.5, -1.0)
    assert not is_confirmed(th, -0.5, -2.5, -1.0)
    assert is_confirmed(extra_vowel, 0.0, -2.5, -1.0)          # GOP cannot judge an added sound
    assert is_confirmed(other, None, -2.5, -1.0)               # no GOP for the word: keep it
    assert is_confirmed(other, 0.0, None)                      # no GOP check at all
    assert not is_confirmed(th, -1.5, -2.5)                    # without a pattern threshold: -2.5 for all
