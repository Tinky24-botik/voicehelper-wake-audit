# Intentionally excluded from Claude audit package

The following are NOT copied:

- .env
- API keys / secrets
- venv/
- __pycache__/
- *.pyc
- *.npy
- *.onnx
- *.tflite
- *.joblib
- Vosk model directories
- model/
- model_small/
- large synthetic WAV datasets
- large feature datasets
- old_bad_positive/
- old_bad_positive_test/
- full openWakeWord documentation
- full notebooks
- full piper-sample-generator tree
- full piper voice models

Reason:
These files are either secrets, generated/binary artifacts, or too large and
not necessary for the first independent code/methodology audit.

Specific model binaries can be supplied later if Claude identifies a concrete
need for them.
