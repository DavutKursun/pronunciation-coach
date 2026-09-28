# Pronunciation Coach: guide for Claude

## Working with the author

- This is the author's portfolio project (new Computer Engineering graduate, applying for AI/ML and Java roles; previously interned at EnglishCentral, a language-learning company). They must understand every part and be able to explain it in interviews.
- Talk to the author in **Turkish**. Write code, comments, commit messages and the README in **English**.
- After each change, explain in a few sentences what you did and why. When a concept is new (CTC, Viterbi, GOP, edit distance, Pearson correlation, cross-validation…), explain it briefly with an example from this project.
- Prefer simple, readable code. Keep changes small and focused.
- Ask before: pushing to GitHub or Hugging Face, deleting files, installing system packages.

## What the project does

A learner reads an English sentence aloud. A wav2vec2 CTC model recognizes the phonemes they said; eSpeak NG gives the phonemes they should have said; a weighted edit-distance alignment finds the differences; known Turkish-speaker error patterns turn them into tips (English + Turkish). CTC forced alignment gives Goodness of Pronunciation (GOP) features, and a scikit-learn model trained on speechocean762 predicts a 0-10 score. The demo is a Gradio app for Hugging Face Spaces. Everything is free: no paid APIs.

## Layout

- `pronunciation/`: the library (see README "Project structure"). `assess.py` ties it together; `analyze()` is model-free so it can be unit-tested.
- `tests/`: pytest; `conftest.py` has `fake_recognition()` to simulate what the model "heard" without downloading it.
- `scripts/extract_features.py` → `data/features/*.csv`; `scripts/train_scorer.py` → `models/scorer.joblib` + `results/metrics.json`.
- Word-level checks (model output cached, rule changes re-run in seconds): `scripts/evaluate_words.py` (speechocean762 train), `scripts/evaluate_saa.py` (Speech Accent Archive dev half; `download_saa.py` + `split_saa.py` first), `scripts/synthetic_errors.py --kokoro` (synthetic error set; Kokoro runs in `.venv-tts`). Run all three after every rule change.
- `app.py`: Gradio demo. `space/README.md`: Space config. `scripts/deploy_space.py`: publishing.
- `notebooks/colab.ipynb`: feature extraction + training on Colab.

## Commands (macOS)

```bash
source .venv/bin/activate      # always first
pytest                          # must stay green
python app.py                   # demo at http://127.0.0.1:7860 (downloads ~1.2 GB model once)
python scripts/deploy_space.py USER/pronunciation-coach --dry-run
```

System tools: `espeak-ng` (brew install espeak-ng). Feature extraction on all 5,000 utterances and training run on Colab (GPU); a quick local check is `python scripts/extract_features.py --limit 20`.

## Rules

1. **Tests first.** Every change to `align.py`, `feedback.py`, `ctc.py` or `phonemes.py` comes with a test. Run `pytest` before every commit.
2. **Honest evaluation.** Choose models only with cross-validation on the training split (GroupKFold by speaker). Use the test split once, for the final numbers. Never tune on it. The same holds for the Speech Accent Archive: tune rules on the dev half only; the test half (`evaluate_saa.py --final`) is used once, in roadmap step 8b.
3. **Honest results.** Report metrics exactly as the scripts print them; never edit numbers by hand.
4. **Feature order.** `FEATURE_NAMES` in `assess.py` must match the trained scorer. Changing features means re-running extraction and training.
5. **scikit-learn version.** The version pinned in `requirements.txt` must equal the one that trained `models/scorer.joblib`.
6. **Privacy.** Never commit recordings of other people. Recordings of friends only with their permission, and don't publish them.
7. **Secrets.** Log in with `hf auth login`; never write tokens into files or commits.
8. **Tips.** Every entry in `TIPS` has `title`, `en` and `tr`. Keep explanations short and practical.
9. Commit in small steps with clear English messages.

## Step workflow

The author gives the work one step at a time. At the end of every step:

1. Summarize what you did, briefly, in Turkish.
2. If code changed, run `pytest`. If any test fails, the step is not done.
3. Commit the changes with a clear English message (skip this until git is set up).
4. Tick the step in the roadmap below.
5. Stop and wait for the next step. Never start the next step on your own.

## Roadmap (tick the boxes as we go)

- [x] 1. Environment setup
- [x] 2. Fix tokenizer loading on macOS
- [x] 3. Git and GitHub
- [x] 4. First run with the real model
- [x] 5. Sanity check on speechocean762
- [x] 5b. Fix false alarms on vowels before r
- [x] 5c. Reduce false alarms with data
- [x] 6. Synthetic error test set
- [x] 7. Speech Accent Archive: dev set analysis and fixes
- [x] 8. Word-level detection metrics
- [x] 8b. Speech Accent Archive: final evaluation on held-out speakers
- [ ] 9. GitHub Actions CI
- [ ] 10. Train on Colab
- [ ] 11. Add results to the project
- [ ] 12. Publish on Hugging Face Spaces
- [ ] 13. Demo GIF and README polish
- [ ] 14. Interview notes and CV
