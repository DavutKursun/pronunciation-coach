from pronunciation.align import align, edit_cost
from pronunciation.phonemes import normalize


def kinds(ops):
    return [op.kind for op in ops]


def test_identical_sequences_are_all_matches():
    ops = align(list("kæt"), list("kæt"))
    assert kinds(ops) == ["match"] * 3
    assert edit_cost(ops) == 0


def test_substitution_w_to_v():
    ops = align(["w", "ɛ", "s", "t"], ["v", "ɛ", "s", "t"])
    assert kinds(ops) == ["sub", "match", "match", "match"]
    assert (ops[0].expected, ops[0].heard) == ("w", "v")


def test_deletion_and_insertion():
    assert kinds(align(["s", "k", "uː", "l"], ["s", "k", "uː"])) == ["match"] * 3 + ["del"]
    ops = align(["s", "k", "uː", "l"], ["ɪ", "s", "k", "uː", "l"])
    assert kinds(ops) == ["ins"] + ["match"] * 4
    assert ops[0].heard == "ɪ"


def test_vowels_pair_with_vowels():
    # "bad" said as "bet": two substitutions, not a deletion plus an insertion
    ops = align(["b", "æ", "d"], ["b", "ɛ", "t"])
    assert kinds(ops) == ["match", "sub", "sub"]


def test_accent_variants_are_not_errors():
    # American flap: "better" with a plain t is fine
    assert kinds(align(["b", "ɛ", "ɾ", "ɚ"], ["b", "ɛ", "t", "ɚ"])) == ["match"] * 4


def test_acceptance_is_directional():
    # the final vowel of "happy" may be /ɪ/ ...
    assert kinds(align(["h", "æ", "p", "i"], ["h", "æ", "p", "ɪ"]))[-1] == "match"
    # ... but "ship" said with a long /iː/ is the ship/sheep mistake
    assert kinds(align(["ʃ", "ɪ", "p"], ["ʃ", "iː", "p"]))[1] == "sub"


def test_function_words_accept_reduced_vowels():
    assert kinds(align(["f", "ɔː"], ["f", "ə"], [True, True])) == ["match", "match"]
    assert kinds(align(["f", "ɔː"], ["f", "ə"], [False, False])) == ["match", "sub"]


def test_normalize_splits_rhotic_vowels():
    assert normalize(["k", "ɑːɹ"]) == ["k", "ɑː", "ɹ"]
    assert normalize(["l", "ɪ", "ɾ", "əl"]) == ["l", "ɪ", "ɾ", "ə", "l"]
    assert normalize(["g"]) == ["ɡ"]
