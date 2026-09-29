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
- Word-level checks (model output cached per model in `data/cache/<model>/`, rule changes re-run in seconds): `scripts/evaluate_words.py` (speechocean762 val speakers), `scripts/evaluate_saa.py` (Speech Accent Archive dev half; `download_saa.py` + `split_saa.py` first), `scripts/synthetic_errors.py --kokoro` (synthetic error set; Kokoro runs in `.venv-tts`). Run all three after every rule change.
- Recognizer fine-tuning (v2-3): `scripts/prepare_l2arctic.py` (annotated L2-ARCTIC sentences → `data/l2arctic/`), `scripts/build_finetune_data.py` (targets → `data/finetune/*.jsonl`), `scripts/evaluate_l2arctic.py` (PER + MDD per model), `scripts/package_kaggle.py` (zip for a private Kaggle dataset), `scripts/finetune_recognizer.py` (training; `--smoke` for a quick end-to-end check) run by `notebooks/finetune_kaggle.ipynb`.
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
5. **scikit-learn version.** The version pinned in `requirements.txt` must equal the one that trained `models/scorer.joblib` and `models/detector.joblib` (the detector stores its version, recognizer id and feature list; retrain it with `python scripts/train_detector.py` whenever the recognizer or the features change).
6. **Privacy.** Never commit recordings of other people. Recordings of friends only with their permission, and don't publish them.
7. **Secrets.** Log in with `hf auth login`; never write tokens into files or commits.
8. **Tips.** Every entry in `TIPS` has `title`, `en` and `tr`. Keep explanations short and practical.
9. Commit in small steps with clear English messages.

## v2: goals and data rules

v1 (git tag `v1.0`) is rule-based; v2 aims at higher recall with a learned error detector and a
recognizer fine-tuned on non-native speech. Every v2 system is run with `scripts/run_experiment.py`
and compared with v1 on the same speakers (`scripts/compare_experiments.py`).

**Targets** (final evaluation: SAA test half, Turkish speakers; aim a bit higher on dev, v1 lost
about 9 points from dev to test):
- precision at least 70% (kept), recall from 34% to at least 50% (stretch: precision 75%, recall 55%)
- false alarms for native English speakers at most 2.5% on dev (10 speakers × 69 words ≈ 690 words: 0.5 points
  is 3–4 words, within the noise; v1 had 1.7% on the SAA test half)
- error-level catch: final devoicing (z → s) at least 60%, ð at least 40%, θ at least 75% (kept)

**Data rules:**
- The SAA test half and the speechocean762 test split stay locked until v2-4.
- speechocean762 train is split by speaker (`data/speechocean_split.json`): "fit" (80%) is for
  training, "val" (20%) for model selection and evaluation during development.
- The SAA dev half may be used to choose thresholds and for early stopping, never as training data.
- Synthetic recordings may be used as a regression check, not as training data for the detector.

**Fine-tuning the recognizer (v2-3):**
- L2-ARCTIC (CC BY-NC 4.0; 24 speakers, 6 L1s, ~150 hand-annotated sentences each) is split by speaker
  (`data/l2arctic_split.json`): test = NJS, TLV, TNI, TXHC, YKWK, ZHAA (the usual MDD test set, e.g.
  Peng et al., Interspeech 2021), dev = 6 (one per L1, 3 female / 3 male), train = 12. The test speakers
  are locked like the SAA test half until v2-4 and only open with `--final`.
- CMU ARCTIC US speakers bdl, slt, clb, rms (permissive license, attribution) keep the model from
  hearing native speech as wrong. They read the same sentences, so the sentences annotated for the
  L2-ARCTIC dev and test speakers are removed from their training data.
- The Speech Accent Archive is never used for training (everyone reads the same paragraph), and
  speechocean762 is not used to train the recognizer (its labels miss the Turkish errors, see v2-2).
- No audio goes into git. A recognizer fine-tuned on L2-ARCTIC is CC BY-NC 4.0; the original model
  stays Apache-2.0 and remains the default (`MODEL_ID` selects the recognizer in the app).

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
- [x] v2-1. Experiment setup
- [x] v2-2. Learned error detector
- [x] v2-3a. Fine-tuning data and training setup
- [ ] v2-3b. Evaluate the fine-tuned recognizer
- [ ] v2-4. Final v1 vs v2 comparison
- [ ] 10. Train on Colab
- [ ] 11. Add results to the project
- [ ] 12. Publish on Hugging Face Spaces
- [ ] 13. Demo GIF and README polish
- [ ] 14. Interview notes and CV
