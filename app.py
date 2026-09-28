"""Gradio demo: pick a sentence, record yourself, get word-by-word pronunciation feedback."""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import gradio as gr

from pronunciation import PronunciationCoach
from pronunciation.audio import SAMPLE_RATE, to_model_input
from pronunciation.phonemes import TIPS
from pronunciation.recognizer import DEFAULT_MODEL, PhonemeRecognizer

ROOT = Path(__file__).parent
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL)
SENTENCES = json.loads((ROOT / "data" / "sentences.json").read_text())
CHOICES = [f"{s['text']}  ·  {s['focus']}" for s in SENTENCES]

_coach: PronunciationCoach | None = None


def get_coach() -> PronunciationCoach:
    global _coach
    if _coach is None:
        _coach = PronunciationCoach(PhonemeRecognizer(MODEL_ID), scorer_path=ROOT / "models" / "scorer.joblib")
    return _coach


def pick_sentence(choice: str) -> str:
    return choice.split("  ·  ")[0] if choice else ""


def speak(text: str):
    """A reference recording made with the eSpeak NG synthesizer (robotic, but free and offline)."""
    if not text.strip():
        raise gr.Error("Type or pick a sentence first.")
    if shutil.which("espeak-ng") is None:
        raise gr.Error("eSpeak NG is not installed on this machine.")
    path = Path(tempfile.mkdtemp()) / "reference.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "140", "-w", str(path), text], check=True)
    return str(path)


GREEN, AMBER, RED = "22, 163, 74", "217, 119, 6", "220, 38, 38"


def word_color(word) -> str:
    """Green unless an error was reported (differences GOP did not confirm don't count)."""
    if not word.issues:
        return GREEN
    return AMBER if word.score >= 0.6 else RED


def render_result(result) -> str:
    if result.score is not None:
        headline = f"Score: <b>{result.score:.1f} / 10</b>"
    else:
        headline = f"Sounds said correctly: <b>{result.phone_accuracy:.0%}</b>"

    words = []
    for w in result.words:
        rgb = word_color(w)
        tooltip = html.escape("; ".join(i.message for i in w.issues) or "correct")
        words.append(
            f'<span title="{tooltip}" style="padding:2px 6px;margin:2px;border-radius:6px;display:inline-block;'
            f'background:rgba({rgb},0.15);border-bottom:3px solid rgb({rgb});font-size:1.25em">'
            f"{html.escape(w.text)}</span>")

    tips = []
    for tip_id in result.tips:
        tip = TIPS[tip_id]
        examples = sorted({result.words[i.word_index].text.lower() for w in result.words for i in w.issues if i.tip == tip_id})
        tips.append(
            f"<div style='margin:10px 0;padding:10px 12px;border-radius:8px;border:1px solid rgba(128,128,128,0.35)'>"
            f"<b>{html.escape(tip['title'])}</b> <span style='opacity:0.75'>in: {html.escape(', '.join(examples))}</span>"
            f"<div>{html.escape(tip['en'])}</div><div style='opacity:0.8'><i>{html.escape(tip['tr'])}</i></div></div>")

    other = [f"<li><b>{html.escape(result.words[i.word_index].text)}</b>: {html.escape(i.message)}</li>"
             for w in result.words for i in w.issues if not i.tip]
    other_html = f"<details><summary>Other differences ({len(other)})</summary><ul>{''.join(other)}</ul></details>" if other else ""

    legend = ("<div style='opacity:0.75;font-size:0.9em'>green = correct · amber = small problem · red = needs work "
              "(hover a word for details). Differences the recognizer was unsure about are not counted.</div>")
    focus = "<h4>What to practise</h4>" + "".join(tips) if tips else "<p>No typical Turkish-speaker errors found. Well done!</p>"
    return f"<div style='font-size:1.1em'>{headline}</div><div style='margin:12px 0'>{''.join(words)}</div>{legend}{focus}{other_html}"


def render_details(result) -> str:
    lines = ["| Word | Expected sounds | Heard | Sounds matched | GOP | Feedback |",
             "| --- | --- | --- | --- | --- | --- |"]
    for w in result.words:
        gop = f"{w.gop:.1f}" if w.gop is not None else "–"
        if w.issues:
            feedback = f"{len(w.issues)} issue(s)"
        elif w.dismissed:
            feedback = "ok (difference not confirmed by GOP)"
        else:
            feedback = "ok"
        lines.append(f"| {w.text} | /{' '.join(w.expected)}/ | /{' '.join(w.heard)}/ | {w.score:.0%} | {gop} | {feedback} |")
    return "\n".join(lines)


def check(text: str, recording):
    if not text or not text.strip():
        raise gr.Error("Type or pick a sentence first.")
    if recording is None:
        raise gr.Error("Record yourself reading the sentence first.")
    sample_rate, data = recording
    audio = to_model_input(data, sample_rate)
    if len(audio) < 0.3 * SAMPLE_RATE:
        raise gr.Error("The recording is too short. Please try again.")
    result = get_coach().assess(audio, text)
    if not result.heard_phones:
        return "<p>No speech was detected. Please record again, closer to the microphone.</p>", ""
    return render_result(result), render_details(result)


with gr.Blocks(title="Pronunciation Coach") as demo:
    gr.Markdown(
        "# Pronunciation Coach\n"
        "Read an English sentence aloud and get word-by-word feedback, with tips for the mistakes "
        "Turkish speakers make most often (th, w/v, ship/sheep, final consonants…).")
    with gr.Row():
        with gr.Column():
            choice = gr.Dropdown(CHOICES, value=CHOICES[0], label="Practice sentence")
            text = gr.Textbox(value=SENTENCES[0]["text"], label="Sentence (you can also type your own)")
            with gr.Row():
                listen = gr.Button("Hear it")
                reference = gr.Audio(label="Reference (synthetic voice)", type="filepath", autoplay=True)
            recording = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your recording")
            run = gr.Button("Check my pronunciation", variant="primary")
        with gr.Column():
            result_html = gr.HTML("<p style='opacity:0.7'>Your feedback will appear here: pick a sentence, "
                                  "press <b>Record</b>, read it aloud, then press <b>Check my pronunciation</b>.</p>")
            with gr.Accordion("Sound-by-sound details (IPA)", open=False):
                details = gr.Markdown()

    choice.change(pick_sentence, choice, text)
    listen.click(speak, text, reference)
    run.click(check, [text, recording], [result_html, details])

if __name__ == "__main__":
    get_coach()  # load the model before the first visitor arrives
    demo.launch()
