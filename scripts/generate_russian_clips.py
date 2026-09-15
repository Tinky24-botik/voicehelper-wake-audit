import os
import glob
import wave
import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

VOICE_MODEL_PATH = "piper_voices/ru_RU-dmitri-medium.onnx"
OUTPUT_DIR = "wakeword_training/lenya/lenya/positive_train"
TARGET_TEXT = "лёня"

NUM_CLIPS = 300


def find_voice_model() -> str:
    if os.path.exists(VOICE_MODEL_PATH):
        return VOICE_MODEL_PATH

    matches = glob.glob("piper_voices/**/*.onnx", recursive=True)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Не нашёл .onnx голос в piper_voices. "
        "Проверь путь VOICE_MODEL_PATH в этом скрипте."
    )


def save_wav(path: str, audio_int16: np.ndarray, sample_rate: int):
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def synthesize_one(voice: PiperVoice, length_scale: float, noise_scale: float, noise_w_scale: float):
    syn_config = SynthesisConfig(
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )

    chunks = list(voice.synthesize(TARGET_TEXT, syn_config=syn_config))

    audio_parts = [chunk.audio_float_array for chunk in chunks]
    audio = np.concatenate(audio_parts)

    audio_int16 = (audio * 32767).astype(np.int16)
    sample_rate = chunks[0].sample_rate

    return audio_int16, sample_rate


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = find_voice_model()
    print(f"Использую голос: {model_path}")

    voice = PiperVoice.load(model_path)

    print(f"Генерирую {NUM_CLIPS} вариаций «{TARGET_TEXT}»...")

    rng = np.random.default_rng(42)

    for i in range(NUM_CLIPS):
        length_scale = rng.uniform(0.8, 1.3)
        noise_scale = rng.uniform(0.5, 0.8)
        noise_w_scale = rng.uniform(0.6, 0.9)

        audio_int16, sample_rate = synthesize_one(
            voice, length_scale, noise_scale, noise_w_scale
        )

        path = os.path.join(OUTPUT_DIR, f"ru_synth_{i:04d}.wav")
        save_wav(path, audio_int16, sample_rate)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{NUM_CLIPS}")

    print("\nГотово.")


if __name__ == "__main__":
    main()