from pathlib import Path
import argparse
import shutil
import os
import subprocess
import time

import numpy as np
import soundfile as sf
import openwakeword
from openwakeword.model import Model


# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = Path(r"D:\VoiceHelper")

MODEL_PATH = BASE_DIR / "wakeword_training" / "lenya" / "lenya.onnx"
NEGATIVE_DIR = BASE_DIR / "wakeword_training" / "lenya" / "lenya" / "negative_test"

OUTPUT_DIR = BASE_DIR / "false_positive_samples"
REPORT_PATH = BASE_DIR / "false_positive_report.txt"

TOP_N = 20
CHUNK_SECONDS = 1.0
SAMPLE_RATE = 16000


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def load_model():
    print("=" * 70)
    print("Загрузка модели...")
    print(MODEL_PATH)
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена:\n{MODEL_PATH}"
        )

    model = Model(
        wakeword_models=[str(MODEL_PATH)],
        inference_framework="onnx",
    )

    print("Модель загружена.")
    print()

    return model


def normalize_audio(audio):
    audio = np.asarray(audio)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # Если WAV сохранён как int16
    if np.max(np.abs(audio)) > 1.5:
        audio /= 32768.0

    audio = np.clip(audio, -1.0, 1.0)

    return audio


def resample_audio(audio, original_sr):
    if original_sr == SAMPLE_RATE:
        return audio

    duration = len(audio) / original_sr

    new_length = int(duration * SAMPLE_RATE)

    if new_length <= 0:
        return np.array([], dtype=np.float32)

    old_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    new_x = np.linspace(0.0, 1.0, new_length, endpoint=False)

    return np.interp(new_x, old_x, audio).astype(np.float32)


def score_wav(model, path):
    try:
        audio, sr = sf.read(path, dtype="float32")

        audio = normalize_audio(audio)

        if sr != SAMPLE_RATE:
            audio = resample_audio(audio, sr)

        if len(audio) == 0:
            return 0.0, 0.0

        chunk_size = int(SAMPLE_RATE * CHUNK_SECONDS)

        scores = []

        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]

            if len(chunk) < int(SAMPLE_RATE * 0.25):
                continue

            # Дополняем короткий последний кусок нулями
            if len(chunk) < chunk_size:
                padded = np.zeros(chunk_size, dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunk = padded

            pcm16 = (chunk * 32767.0).astype(np.int16)

            prediction = model.predict(pcm16)

            if isinstance(prediction, dict):
                values = list(prediction.values())

                if not values:
                    continue

                score = float(np.max(values))
            else:
                score = float(prediction)

            scores.append(score)

        if not scores:
            return 0.0, len(audio) / SAMPLE_RATE

        return max(scores), len(audio) / SAMPLE_RATE

    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        return 0.0, 0.0


def open_wav(path):
    """
    Открывает WAV стандартным проигрывателем Windows.
    """

    try:
        os.startfile(str(path))
        return True
    except Exception as e:
        print(f"Не удалось открыть {path.name}: {e}")
        return False


# ============================================================
# ОСНОВНАЯ ПРОВЕРКА
# ============================================================

def main(play=False):
    print()
    print("=" * 70)
    print("FALSE POSITIVE INSPECTOR")
    print("=" * 70)
    print()

    if not NEGATIVE_DIR.exists():
        raise FileNotFoundError(
            f"Папка negative_test не найдена:\n{NEGATIVE_DIR}"
        )

    wav_files = sorted(NEGATIVE_DIR.glob("*.wav"))

    if not wav_files:
        raise RuntimeError(
            f"В папке нет WAV:\n{NEGATIVE_DIR}"
        )

    print(f"Найдено WAV: {len(wav_files)}")
    print()

    model = load_model()

    results = []

    print("=" * 70)
    print("Проверка negative_test")
    print("=" * 70)
    print()

    for index, wav_path in enumerate(wav_files, start=1):

        score, duration = score_wav(model, wav_path)

        results.append({
            "path": wav_path,
            "score": score,
            "duration": duration,
        })

        print(
            f"[{index:4d}/{len(wav_files)}] "
            f"{wav_path.name:<25} "
            f"score={score:.6f} "
            f"duration={duration:.2f}s"
        )

    # ========================================================
    # Сортировка
    # ========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print()
    print("=" * 70)
    print(f"TOP-{TOP_N} FALSE POSITIVES")
    print("=" * 70)
    print()

    for i, item in enumerate(results[:TOP_N], start=1):
        print(
            f"{i:2d}. "
            f"{item['score']:.6f}  "
            f"{item['duration']:6.2f}s  "
            f"{item['path'].name}"
        )

    # ========================================================
    # Статистика
    # ========================================================

    scores = np.array(
        [item["score"] for item in results],
        dtype=np.float32
    )

    print()
    print("=" * 70)
    print("СТАТИСТИКА")
    print("=" * 70)
    print()

    print(f"Всего файлов:       {len(scores)}")
    print(f"Средний score:      {np.mean(scores):.6f}")
    print(f"Медиана:            {np.median(scores):.6f}")
    print(f"Максимум:           {np.max(scores):.6f}")
    print(f"Минимум:            {np.min(scores):.6f}")

    for threshold in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
        count = int(np.sum(scores >= threshold))

        print(
            f"score >= {threshold:.1f}: "
            f"{count}/{len(scores)}"
        )

    # ========================================================
    # Сохранение TOP файлов
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("Копирование TOP файлов")
    print("=" * 70)
    print()

    top_results = results[:TOP_N]

    copied_files = []

    for i, item in enumerate(top_results, start=1):

        source = item["path"]

        destination = (
            OUTPUT_DIR /
            f"{i:02d}_score_{item['score']:.6f}_{source.name}"
        )

        try:
            shutil.copy2(source, destination)

            copied_files.append({
                "source": source,
                "destination": destination,
                "score": item["score"],
                "duration": item["duration"],
            })

            print(
                f"{i:2d}. "
                f"{destination.name}"
            )

        except Exception as e:
            print(
                f"[ERROR] Не удалось скопировать "
                f"{source.name}: {e}"
            )

    # ========================================================
    # REPORT
    # ========================================================

    with REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write("FALSE POSITIVE REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Negative directory: {NEGATIVE_DIR}\n")
        f.write(f"Total files: {len(results)}\n\n")

        f.write("STATISTICS\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Average: {np.mean(scores):.6f}\n"
        )

        f.write(
            f"Median: {np.median(scores):.6f}\n"
        )

        f.write(
            f"Maximum: {np.max(scores):.6f}\n"
        )

        f.write(
            f"Minimum: {np.min(scores):.6f}\n"
        )

        for threshold in [0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:

            count = int(np.sum(scores >= threshold))

            f.write(
                f"Score >= {threshold:.1f}: "
                f"{count}/{len(scores)}\n"
            )

        f.write("\n")
        f.write(f"TOP-{TOP_N}\n")
        f.write("-" * 70 + "\n")

        for i, item in enumerate(top_results, start=1):

            f.write(
                f"{i:2d}. "
                f"score={item['score']:.6f} "
                f"duration={item['duration']:.2f}s "
                f"file={item['path'].name}\n"
            )

        f.write("\n")
        f.write("ALL FILES SORTED BY SCORE\n")
        f.write("-" * 70 + "\n")

        for i, item in enumerate(results, start=1):

            f.write(
                f"{i:4d}. "
                f"score={item['score']:.6f} "
                f"duration={item['duration']:.2f}s "
                f"file={item['path'].name}\n"
            )

    print()
    print(f"Отчёт сохранён:")
    print(REPORT_PATH)

    print()
    print(f"TOP-{TOP_N} WAV скопированы сюда:")
    print(OUTPUT_DIR)

    # ========================================================
    # ПОСЛЕДОВАТЕЛЬНОЕ ПРОСЛУШИВАНИЕ
    # ========================================================

    if play and copied_files:

        print()
        print("=" * 70)
        print("ПРОСЛУШИВАНИЕ FALSE POSITIVES")
        print("=" * 70)
        print()
        print(
            "Каждый файл будет открыт стандартным "
            "проигрывателем Windows."
        )
        print(
            "После прослушивания нажми Enter "
            "для перехода к следующему."
        )
        print()
        print(
            "Если не хочешь слушать какой-либо файл, "
            "просто нажми Enter."
        )

        for i, item in enumerate(copied_files, start=1):

            print()
            print("-" * 70)
            print(
                f"[{i}/{len(copied_files)}] "
                f"score={item['score']:.6f}"
            )
            print(
                f"Файл: {item['destination'].name}"
            )
            print("-" * 70)

            opened = open_wav(item["destination"])

            if opened:
                print("Файл открыт в проигрывателе Windows.")

            input("Enter -> следующий файл... ")

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inspect false positives of custom wake-word model."
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help="Открывать TOP WAV по одному для прослушивания."
    )

    args = parser.parse_args()

    main(play=args.play)