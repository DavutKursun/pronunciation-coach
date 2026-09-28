# Pronunciation Coach

**Phoneme-level English pronunciation feedback for Turkish speakers.** Read a sentence aloud, see which words and sounds went wrong, and get targeted tips (in English and Turkish) for the mistakes Turkish speakers make most often.

🤗 **Live demo:** [huggingface.co/spaces/DavutKursun/pronunciation-coach](https://huggingface.co/spaces/DavutKursun/pronunciation-coach)

<!-- Add a short GIF of the demo here: record yourself using it. It is the first thing people look at. -->

## What it does

- Colors every word by how well it was pronounced, with sound-by-sound details in IPA
- Detects error patterns typical of Turkish speakers and explains how to fix them:

| Pattern | Example | How it is detected |
| --- | --- | --- |
| "th" said as t/d/s/z | *think* → "tink" | substitution of /θ/ or /ð/ |
| w and v mixed up | *west* → "vest" | substitution /w/ ↔ /v/ |
| short vs long vowels | *ship* ↔ *sheep*, *pull* ↔ *pool* | /ɪ/ ↔ /iː/, /ʊ/ ↔ /uː/ |
| /æ/ said as e or a | *bad* → "bed" | substitution of /æ/ |
| final consonant devoicing | *bed* → "bet" | voiced → voiceless in the word's final consonant group |
| extra vowel in consonant groups | *school* → "is-chool" | vowel inserted before or inside a cluster |
| /ŋ/ followed by g, or said as n | *singing* → "sing-ging" | insertion of /ɡ/ after /ŋ/ |
| tapped or rolled r | *red* with Turkish r | /ɹ/ heard as /r/ or /ɾ/ |

- Gives an overall 0–10 score from a model trained on expert ratings (speechocean762)
- Plays a synthetic reference pronunciation (eSpeak NG)

## How it works

```mermaid
flowchart LR
    A[Recording] --> B[wav2vec2 CTC<br/>phoneme recognizer]
    T[Target sentence] --> G[eSpeak NG<br/>expected phonemes]
    B --> H[Heard phonemes]
    G --> AL[Weighted edit-distance<br/>alignment]
    H --> AL
    AL --> F[Per-word errors +<br/>Turkish error patterns]
    B --> FA[CTC forced alignment<br/>Viterbi]
    G --> FA
    FA --> GOP[Goodness of<br/>Pronunciation]
    GOP -- confirms errors --> F
    AL --> S[Scoring model]
    GOP --> S
    F --> UI[Feedback + tips]
    S --> UI
```

1. **Phoneme recognition.** [`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) turns audio into IPA phonemes (one prediction every 20 ms, decoded with greedy CTC). The model is multilingual, so decoding is limited to English phonemes and sounds Turkish speakers typically use (tapped r, Turkish vowels…); it cannot "hear" a Mandarin tone or a Russian soft consonant.
2. **Expected phonemes.** `phonemizer` + eSpeak NG convert each word of the target sentence into the phonemes a US English speaker would say, in the same phoneme style the recognizer was trained on.
3. **Alignment** (`pronunciation/align.py`). A weighted edit distance solved with dynamic programming, plus a backtrace, finds which sounds were matched, substituted, skipped or added. Costs are phonetically motivated: accepted accent variants (the American flap in *better*, reduced vowels in *to/for/the*) are free, vowel↔vowel swaps are cheaper than vowel↔consonant ones, and acceptance is directional (*happy* may end in /ɪ/, but *ship* said with /iː/ is an error). Before /r/, where English has no *ship/sheep* contrast, short and long vowels are both accepted (*here*, *sure*, *zero*).
4. **Error patterns** (`pronunciation/feedback.py`). Each difference is attributed to a word and matched against the patterns above, using its position (for example, only the final consonant group counts for devoicing). A difference is only reported when GOP confirms it (step 5).
5. **Goodness of Pronunciation** (`pronunciation/ctc.py`). The expected phonemes are force-aligned to the audio with the Viterbi algorithm over the CTC state graph, written from scratch in NumPy. For each phoneme, the log-posterior ratio against the best competing phoneme shows how confidently it was produced. GOP also double-checks the feedback: greedy decoding can turn a near tie into a heard error, so a word's differences are reported only if its lowest GOP is below −2.5, meaning some expected sound was at least ~12 times less likely than the model's favourite. Otherwise they are treated as a likely mishearing. Differences that match a typical Turkish-speaker pattern need less evidence (GOP below −1.0), because learners are likely to make them. Added sounds of a known pattern (*is-chool*, *sing-ging*) are always reported, because GOP only scores the expected sounds. The −2.5 threshold was chosen on the speechocean762 training split (`scripts/evaluate_words.py --sweep`), the −1.0 one on the dev half of the Speech Accent Archive Turkish speakers (`scripts/evaluate_saa.py`).
6. **Scoring model** (`scripts/train_scorer.py`). 17 features (alignment error rates, word scores, GOP statistics, speaking rate) feed a regression model trained to predict expert sentence scores. Models are selected with speaker-grouped cross-validation on the training set and evaluated once on the official test set.

## Results

The results below belong to **v1**, the rule-based version tagged [`v1.0`](https://github.com/DavutKursun/pronunciation-coach/tree/v1.0) in git. v2 work (a learned error detector and a recognizer fine-tuned on non-native speech) is compared with it using `scripts/run_experiment.py` and `scripts/compare_experiments.py`.

Evaluated on the speechocean762 test set (2,500 utterances, speakers not seen in training). PCC = Pearson correlation with the expert scores.

| Model | CV PCC (train speakers) | Test PCC | Test MSE |
| --- | --- | --- | --- |
| Baseline: phone accuracy only | – | – | – |
| Ridge regression | – | – | – |
| Gradient boosting | – | – | – |

**Word-level error detection** on the same test set: does the feedback flag the words the experts did not score as perfect (word accuracy below 10)? No threshold or rule was tuned on it.

| Words | Expert: wrong | False alarm | Recall | Precision | F1 | Word score PCC |
| --- | --- | --- | --- | --- | --- | --- |
| – | – | – | – | – | – | – |

<!-- Fill in from the tables printed by scripts/train_scorer.py (also saved in results/metrics.json). -->

False alarm = correct words we flagged; recall = wrong words we flagged; precision = flagged words that were really wrong. `scripts/evaluate_words.py` (speechocean762, val speakers of the train split) and `scripts/evaluate_saa.py` (Turkish speakers of the Speech Accent Archive, with an error-level catch rate) report the same metrics during development.

### Real Turkish speakers: Speech Accent Archive

In the [Speech Accent Archive](https://accent.gmu.edu/) every speaker reads the same paragraph ("Please call Stella…") and trained phoneticians transcribe it in IPA. 24 native Turkish speakers and 20 US English speakers with a text transcription were split by speaker into a dev half, used to choose rules and thresholds, and a test half, evaluated once at the end. A word is wrong when the expert's transcription differs from the expected pronunciation; brackets are 95% confidence intervals from resampling speakers (bootstrap).

**Main labels** (a word is wrong when the expert transcription differs from the expected phonemes outside our accepted variants (flap, reduced vowels, weak forms of function words...))

| | Dev (10 English / 12 Turkish speakers) | Test (10 English / 12 Turkish speakers) |
| --- | --- | --- |
| English speakers: false alarm | 2.1% [0.8%–3.8%] | 1.7% [0.3%–3.2%] |
| Turkish speakers: false alarm | 7.1% [3.7%–12.0%] | 11.4% [6.5%–17.7%] |
| Turkish speakers: recall | 42.4% [32.7%–50.1%] | 33.5% [24.7%–42.9%] |
| Turkish speakers: precision | 82.3% [77.8%–86.5%] | 70.1% [61.0%–79.4%] |
| Turkish speakers: F1 | 0.559 [0.469–0.624] | 0.454 [0.362–0.529] |
| Turkish speakers: error-level catch | 39.7% [30.4%–47.1%] | 29.1% [21.0%–37.2%] |

**Raw labels** (a word is wrong when the expert transcription differs from the expected phonemes at all (only narrow-IPA detail and vowel length are ignored))

| | Dev (10 English / 12 Turkish speakers) | Test (10 English / 12 Turkish speakers) |
| --- | --- | --- |
| English speakers: false alarm | 2.0% [0.2%–4.2%] | 1.8% [0.5%–3.3%] |
| Turkish speakers: false alarm | 8.4% [3.1%–15.1%] | 10.6% [6.1%–16.6%] |
| Turkish speakers: recall | 30.2% [21.5%–38.7%] | 27.3% [19.4%–35.1%] |
| Turkish speakers: precision | 87.1% [81.9%–93.2%] | 81.6% [76.5%–87.0%] |
| Turkish speakers: F1 | 0.449 [0.345–0.533] | 0.409 [0.313–0.490] |

**Per pattern, Turkish speakers** (errors the expert heard → share we found on the same sound)

| Pattern | Dev | Test |
| --- | --- | --- |
| 'th' as in think /θ/ | 89% of 35 | 79% of 28 |
| 'th' as in this /ð/ | 29% of 49 | 17% of 47 |
| 'w' as in west /w/ | 64% of 33 | 58% of 33 |
| long 'ee' as in sheep /iː/ | 50% of 6 | 83% of 6 |
| short 'i' as in ship /ɪ/ | 18% of 40 | 4% of 26 |
| 'a' as in cat /æ/ | 48% of 27 | 23% of 22 |
| short 'oo' as in pull /ʊ/ | 0% of 7 | 0% of 8 |
| long 'oo' as in pool /uː/ | 0% of 3 | – |
| 'er' as in bird /ɜː/ | 0% of 1 | 0% of 1 |
| 'ng' as in sing /ŋ/ | 0% of 3 | 33% of 3 |
| English 'r' /ɹ/ | 24% of 29 | 0% of 24 |
| voiced sound at the end of a word | 38% of 146 | 36% of 126 |
| extra vowel in a consonant group | 45% of 11 | 75% of 4 |

The main labels leave out differences we accept as correct English (the American flap in *better*, *æn* for *and*…), while the raw labels count every deviation the expert wrote. Both give the same picture (about 2% false alarms for native speakers; for Turkish speakers 70–87% of the flagged words are really wrong, but most wrong words are missed), so the result does not hinge on our list of accepted variants.

The test half is clearly worse than the dev half for Turkish speakers (recall 42% → 34%, precision 82% → 70%, F1 0.56 → 0.45), and the gap is smaller with the raw labels (F1 0.45 → 0.41), which do not depend on the variants we accepted while looking at dev. So part of the dev result is overfitting to the 12 dev speakers; the rest is speaker-to-speaker variation, which is large (per-speaker recall ranges from 0% to 65%). The test numbers are the ones to quote.

### v2 experiments (development sets only)

**v2-2: a learned error detector.** A gradient-boosting model (`pronunciation/detector.py`, `scripts/train_detector.py`) predicts an error probability for every expected sound from GOP scores, the probability of the typical Turkish replacement, the alignment, the sound class, position and timing. It is trained on the speechocean762 phone scores of the "fit" speakers; its red/yellow thresholds are chosen on the Speech Accent Archive dev half. Compared with v1 on the same speakers (paired bootstrap over speakers; "real" = the 95% interval excludes zero):

| Metric | v1 | v2-2 | Difference | 95% CI | Real? |
| --- | --- | --- | --- | --- | --- |
| SAA dev, Turkish: recall | 42.4% | 11.4% | -31.0% | [-37.1%, -23.7%] | yes |
| SAA dev, Turkish: precision | 82.3% | 71.9% | -10.3% | [-23.9%, +0.5%] | no |
| SAA dev, Turkish: F1 | 55.9% | 19.6% | -36.3% | [-42.5%, -29.0%] | yes |
| SAA dev, Turkish: false alarm | 7.1% | 3.5% | -3.7% | [-7.2%, -1.1%] | yes |
| SAA dev, Turkish: error-level catch | 39.7% | 7.8% | -31.9% | [-38.3%, -24.1%] | yes |
| SAA dev, Turkish: final devoicing catch | 38.4% | 2.7% | -35.6% | [-48.1%, -21.6%] | yes |
| SAA dev, Turkish: z → s catch | 38.9% | 2.2% | -36.7% | [-56.8%, -19.4%] | yes |
| SAA dev, Turkish: ð catch | 28.6% | 0.0% | -28.6% | [-41.7%, -16.4%] | yes |
| SAA dev, Turkish: θ catch | 88.6% | 2.9% | -85.7% | [-96.9%, -70.6%] | yes |
| SAA dev, English: false alarm | 2.1% | 2.0% | -0.2% | [-1.0%, +0.6%] | no |
| speechocean val: recall | 83.5% | 81.8% | -1.7% | [-5.8%, +0.3%] | no |
| speechocean val: precision | 31.8% | 39.6% | +7.7% | [+5.5%, +10.1%] | yes |
| speechocean val: false alarm | 26.8% | 18.8% | -8.1% | [-10.3%, -6.2%] | yes |
| speechocean val: F1 | 46.1% | 53.4% | +7.3% | [+5.3%, +9.2%] | yes |

The detector is better on speechocean762 (the kind of speakers it was trained on) but much worse on Turkish speakers: speechocean's Mandarin-speaking experts rarely mark the typical Turkish errors (θ, ð, final devoicing) as wrong, so the model learns to ignore the very signals that matter here, even though they are in the recognizer's output (for example, the probability of *s* in the frames of a final *z* separates the experts' z → s errors with AUC 0.70 on dev). It is therefore not shipped with the demo, which keeps using v1; `python scripts/train_detector.py` rebuilds it, and it will be retrained on the output of a fine-tuned recognizer in v2-3.

## Project structure

```
pronunciation/
  phonemes.py     phoneme classes, accepted variants, Turkish error patterns and tips
  g2p.py          text -> expected phonemes (phonemizer + eSpeak NG)
  align.py        weighted edit-distance alignment (dynamic programming)
  feedback.py     per-word results and error-pattern detection
  ctc.py          greedy CTC decoding, Viterbi forced alignment, GOP scores
  recognizer.py   wav2vec2 phoneme recognizer
  assess.py       the full pipeline and the scoring features
  audio.py        audio loading and resampling to 16 kHz
  saa.py          Speech Accent Archive: narrow IPA -> our phonemes, expert labels
  metrics.py      word- and error-level detection metrics, bootstrap intervals, system comparison
  cache.py        recognizer output cached per model and data set
  systems.py      a "system": recognizer + decision mechanism + settings (for experiments)
  speechocean.py  speechocean762 phone labels (ARPAbet) moved onto our expected phonemes
  detector.py     v2 learned error detector: features per expected sound, red/yellow decision
scripts/
  extract_features.py   run the pipeline on speechocean762
  evaluate_words.py     word-level false alarm / catch rates on speechocean762 (val speakers)
  split_speechocean.py  speaker-level fit/val split of speechocean762 train (data/speechocean_split.json)
  download_saa.py       download Turkish and US English speakers from the Speech Accent Archive
  split_saa.py          speaker-level dev/test split (data/saa_split.json)
  evaluate_saa.py       word-level evaluation against expert IPA transcriptions
  smoke_test_real_model.py  quick check of the real model on synthesized sentences
  synthetic_errors.py   synthetic test set of Turkish-speaker errors (eSpeak NG, Kokoro)
  kokoro_tts.py         Kokoro-82M synthesis, run in its own environment (.venv-tts)
  run_experiment.py     run a system on every development set -> results/experiments/<name>.json
  compare_experiments.py  compare two systems on the same speakers (paired bootstrap)
  train_detector.py     train the v2 error detector and choose its thresholds (one command)
experiments/      system definitions (v1.json, ...)
  train_scorer.py       train and evaluate the scoring model
  deploy_space.py       publish the demo to Hugging Face Spaces
tests/            unit tests (pytest), no model download needed
app.py            Gradio demo
notebooks/colab.ipynb   feature extraction and training on a free Colab GPU
```

## Quickstart (macOS)

```bash
brew install espeak-ng python@3.12
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                     # unit tests, about 5 seconds
python app.py              # first run downloads the recognizer (about 1.2 GB)
```

On Linux: `sudo apt-get install espeak-ng`. On Apple Silicon the recognizer runs on the GPU (MPS) automatically; if you get an MPS error, start it with `PC_DEVICE=cpu python app.py`.

## Train the scoring model

Open `notebooks/colab.ipynb` in Google Colab with a free T4 GPU. It extracts features for all 5,000 speechocean762 utterances, trains the scorer and prints the results table. Then put `models/scorer.joblib` and `results/metrics.json` into the repository.

## Deploy the demo

```bash
hf auth login
python scripts/deploy_space.py DavutKursun/pronunciation-coach
```

The free CPU hardware of Hugging Face Spaces is enough.

## Limitations

- The reference accent is US English (eSpeak `en-us`). Some British pronunciations are accepted as variants, but not all.
- eSpeak gives one pronunciation per word. Words with several correct pronunciations can cause false alarms.
- The recognizer can mishear, especially with background noise. Treat the feedback as a guide, not a verdict.
- The recognizer often does not hear some typical Turkish errors. A final *z* said as *s* and *ð* said as *d* are the most common errors that the experts heard and we missed (45 and 36 words in the Speech Accent Archive test half): the model outputs the expected sound, so no rule on its output can catch them. This would need a recognizer fine-tuned on accented speech.
- On real Turkish speakers the feedback is precise but misses many errors: on the Speech Accent Archive test half it finds about a third of the words the experts marked (see Results), from only 12 test speakers.
- The scoring model is trained on speechocean762, whose speakers are Mandarin native speakers (half of them children). The tips target Turkish speakers, but no Turkish-speaker data was used for training.
- Word stress and intonation are not assessed.

## Credits and licenses

- Recognizer: [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) (Apache-2.0)
- Data: [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762) ([OpenSLR 101](https://www.openslr.org/101/), CC BY 4.0)
- Evaluation data: [Speech Accent Archive](https://accent.gmu.edu/) (Weinberger, S. H. & Kelley, M. C., George Mason University), recordings and IPA transcriptions [on OSF](https://accent.gmu.edu/download), CC BY-NC-SA 4.0. Used for non-commercial evaluation only; the recordings are downloaded by `scripts/download_saa.py` and are not part of this repository.
- Synthetic test voices: [eSpeak NG](https://github.com/espeak-ng/espeak-ng) and [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0)
- G2P and speech synthesis: [phonemizer](https://github.com/bootphon/phonemizer) and [eSpeak NG](https://github.com/espeak-ng/espeak-ng) (GPL-3.0)
- Synthetic test voices (not part of the app): [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0)
- Code: MIT (see LICENSE)
