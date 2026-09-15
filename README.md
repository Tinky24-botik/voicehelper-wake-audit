# VoiceHelper Wake-Word Audit Package

This repository is a reduced technical snapshot of the VoiceHelper wake-word
investigation.

It is NOT a complete backup of the VoiceHelper project.

The purpose is independent code/data/experiment review.

Important:
- Do not modify main.py based on this package.
- Do not assume archived experiments are production code.
- Do not immediately recommend another large DNN training run.
- Pay particular attention to data leakage, synthetic-data generation,
  feature extraction, temporal information, validation methodology,
  and realtime microphone behavior.

The production VoiceHelper remains outside this audit package.
