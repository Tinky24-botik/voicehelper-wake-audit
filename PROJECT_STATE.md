# VoiceHelper Wake-Word Investigation вЂ” Current State

## Goal

Wake word:
"Р»С‘РЅСЏ"

Desired architecture:

microphone
в†’ light wake-word detector
в†’ "Р»С‘РЅСЏ"
в†’ Vosk/Groq
в†’ CommandProcessor

The experimental wake-word work must remain isolated until validated.

---

# 1. Existing VoiceHelper runtime

The production application contains:

- main.py
- core/
- plugins/
- Vosk models
- CommandProcessor
- browser/system/telegram/youtube functionality

The current wake-word experiments have NOT been integrated into main.py.

---

# 2. Existing openWakeWord integration

core/wakeword.py is a runtime wrapper around a supplied ONNX model.

It:
- loads an ONNX wake-word model through the local openWakeWord fork;
- receives int16 audio;
- calls model.predict();
- applies a threshold;
- applies cooldown;
- invokes a callback.

It does NOT:
- train models;
- load personal_wake.joblib;
- directly modify training data.

---

# 3. Existing DTW implementation

core/wake_dtw.py contains an experimental DTW detector.

Current parameters:

sample_rate = 16000
n_mfcc = 13
n_mels = 32
hop_length = 160
n_fft = 400

MFCC features are extracted with librosa.

The implementation:
- mean-centers audio;
- peak-normalizes audio;
- extracts MFCC;
- z-normalizes feature channels;
- uses subsequence DTW;
- normalizes the DTW distance by template length.

Silence is detected through RMS.

---

# 4. Custom openWakeWord DNN experiment

A custom "lenya" ONNX model was trained.

The training pipeline contained several discovered problems, including:

- feature transform setup;
- fixed feature dimensions;
- 34 vs 35 timestep mismatch;
- argparse defaults;
- false-positive validation methodology;
- synthetic speech generation;
- training/validation composition.

A later model achieved:

Accuracy в‰€ 0.968
Recall в‰€ 0.964
Reported FP/hour в‰€ 16968

This model is NOT considered production-ready.

The very high FP result is treated as evidence that the training/validation/model combination is broken, but the exact causal bug is not considered proven.

---

# 5. Synthetic speech issue

A separate script generated Russian speech with:

ru_RU-dmitri-medium.onnx

and target text:

"Р»С‘РЅСЏ"

However, the standard training pipeline calling generate_samples() did not explicitly pass
the Russian voice/model and could fall back to the default:

en-us-libritts-high.pt

Some synthetic positive data therefore may have been generated using the wrong/default
English pipeline.

Do not assume that every synthetic positive file was Russian.

This distinction is important.

---

# 6. Custom model A/B test

The custom lenya ONNX model was compared against the stock hey_jarvis model.

The custom model produced high scores on many negative synthetic files.

The stock hey_jarvis model stayed near zero on the same negative material.

This suggests that the custom model learned an unwanted acoustic/data property rather
than robustly detecting the intended word.

---

# 7. Personal Logistic Regression experiment

A classifier was trained using 384-dimensional vectors made from:

mean
std
min
max

over 96-dimensional speech embeddings.

Strict group holdout:

TRAIN positive:
- lenya_01
- lenya_02
- lenya_03
- 300 synthetic ru_synth clips

TEST positive:
- lenya_04 ... lenya_10

REAL_AUG variants were excluded from the strict positive test.

Negative:
- 4000 training negatives
- 1000 holdout negatives
- 1000 external negatives

Offline result:

threshold 0.90:
positive hit = 100%
negative FP = 0%
accuracy = 1.0
ROC-AUC = 1.0

This looked excellent offline.

---

# 8. Personal Logistic Regression realtime test

The saved personal_wake.joblib was tested against live microphone audio.

It produced high scores (~0.97вЂ“0.999) even during silence.

The score changed according to window composition and acoustic context.

Therefore the model was not considered a valid wake-word detector.

The likely methodological problem is that mean/std/min/max aggregation destroyed temporal information.

This experiment has been archived.

---

# 9. Current DTW offline experiment

Templates:

lenya_01.wav
lenya_02.wav
lenya_03.wav

Independent positives:

lenya_04.wav ... lenya_10.wav

Real negatives:

negative_check/*.wav

Old false positives:

false_positive_samples/*.wav

Results:

Independent positive:
min    2.405370
median 2.832143
mean   2.749040
max    2.900862

Real negative:
min    2.979042
median 3.276672
mean   3.226680
max    3.351743

Old false positives:
min    2.920790
median 3.137421
mean   3.142744
max    3.415939

At threshold 2.900862:

positive hit = 7/7
negative FP = 0/30

At threshold 2.920790:

positive hit = 7/7
negative FP = 1/30

The margin is small and the sample size is small.

This is NOT considered production-ready.

---

# 10. Augmented DTW experiment

A follow-up experiment was prepared using audio augmentations:

gain:
0.65вЂ“1.35

noise:
0.002вЂ“0.015

speed:
0.93вЂ“1.07

random silence:
up to 0.15 sec at both sides

The goal is to test robustness without immediately starting another large DNN training run.

---

# 11. Current architectural hypothesis

Potential direction:

microphone
в†’ VAD / speech gate
в†’ openWakeWord embedding_model.onnx
в†’ temporal sequence of embeddings
в†’ sequence similarity / DTW / small temporal classifier
в†’ "Р»С‘РЅСЏ"
в†’ Vosk/Groq

The important distinction is:

DO NOT aggregate the complete temporal sequence into only:

mean/std/min/max

because that loses word timing and phonetic progression.

---

# 12. Archived experiments

Old personal model:

models/old/personal_wake.joblib

Old personal wake scripts:

tools/old_wake_tests/

These are historical experiments, not production components.

---

# 13. Important review request

Before writing new code, independently audit:

1. data generation;
2. synthetic positive generation;
3. negative generation;
4. train/test leakage;
5. feature extraction;
6. openWakeWord embedding usage;
7. DTW implementation;
8. realtime windowing;
9. VAD placement;
10. threshold calibration;
11. evaluation methodology;
12. why offline models can look excellent while failing on live microphone audio.

Do not immediately recommend a large DNN training run.

First identify the cheapest experiment that can falsify the current hypotheses.
