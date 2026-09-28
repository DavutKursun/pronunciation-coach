import numpy as np
from conftest import BLANK, TOKEN_TO_ID, fake_recognition, needs_espeak

from pronunciation import PronunciationCoach
from pronunciation.g2p import phonemize_words, tokenize


class FakeRecognizer:
    token_to_id, blank_id = TOKEN_TO_ID, BLANK

    def __init__(self, heard):
        self.heard = heard

    def recognize(self, audio):
        return fake_recognition(self.heard)


@needs_espeak
def test_feedback_page_lists_every_detected_pattern():
    import app

    text = "We will walk to school."
    heard = []
    for word, phones in zip(tokenize(text), phonemize_words(tokenize(text))):
        phones = ["v" if p == "w" else p for p in phones]
        heard += (["ɪ"] + phones) if word == "school" else phones
    result = PronunciationCoach(FakeRecognizer(heard), scorer_path=None).assess(np.zeros(16000, np.float32), text)
    page = app.render_result(result)
    assert "as in west" in page and "extra vowel" in page
    assert "Dudaklarını" in page  # the Turkish explanation is shown too
    assert app.render_details(result).count("\n") == len(result.words) + 1


def test_pick_sentence_strips_the_focus_label():
    import app

    assert app.pick_sentence(app.CHOICES[0]) == app.SENTENCES[0]["text"]


def think(reported: bool):
    from pronunciation.align import Op
    from pronunciation.feedback import Issue, WordResult

    ops = [Op("sub", "θ", "t", 0, 0)] + [Op("match", p, p, k, k) for k, p in enumerate(["ɪ", "ŋ", "k"], 1)]
    issue = Issue(0, "sub", "θ", "t", "th_voiceless", "/θ/ sounded like /t/")
    if reported:
        return WordResult("think", ["θ", "ɪ", "ŋ", "k"], ops, issues=[issue], gop=-6.0)
    return WordResult("think", ["θ", "ɪ", "ŋ", "k"], ops, dismissed=[issue], gop=-0.1)


def test_word_color_follows_the_reported_issues():
    import app

    assert app.word_color(think(reported=False)) == app.GREEN   # GOP did not confirm the difference
    assert app.word_color(think(reported=True)) == app.AMBER    # 3 of 4 sounds right


def test_details_show_differences_that_gop_did_not_confirm():
    import app
    from pronunciation.assess import Assessment

    result = Assessment("think", [think(reported=False)], ["t", "ɪ", "ŋ", "k"], 0.75, {})
    details = app.render_details(result)
    assert "not confirmed" in details and "-0.1" in details


def test_headline_counts_only_reported_errors():
    import app
    from pronunciation.assess import Assessment

    # alignment says 3 of 4 sounds matched in both cases (phone_accuracy 0.75)
    unsure = Assessment("think", [think(reported=False)], ["t", "ɪ", "ŋ", "k"], 0.75, {})
    wrong = Assessment("think", [think(reported=True)], ["t", "ɪ", "ŋ", "k"], 0.75, {})
    assert app.reported_accuracy(unsure) == 1.0
    assert app.reported_accuracy(wrong) == 0.75
    assert "Sounds said correctly: <b>100%</b>" in app.render_result(unsure)


def test_detector_levels_set_the_word_colors_and_the_legend():
    import app
    from pronunciation.assess import Assessment

    red = think(reported=True)
    red.level, red.error_prob = "red", 0.95
    yellow = think(reported=True)
    yellow.level, yellow.error_prob = "yellow", 0.6
    green = think(reported=False)
    green.level, green.error_prob = None, 0.1
    assert [app.word_color(w) for w in (red, yellow, green)] == [app.RED, app.AMBER, app.GREEN]
    page = app.render_result(Assessment("think", [red, yellow, green], [], 0.0, {}))
    assert "sure error" in page and "possible error" in page
