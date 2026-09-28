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
