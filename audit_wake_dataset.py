from pathlib import Path
from collections import Counter
import wave
import numpy as np


BASE_DIR = Path(r"D:\VoiceHelper")
DATA_DIR = BASE_DIR / "wakeword_training" / "lenya" / "lenya"

SETS = {
    "positive_train": DATA_DIR / "positive_train",
    "positive_test": DATA_DIR / "positive_test",
    "negative_train": DATA_DIR / "negative_train",
    "negative_test": DATA_DIR / "negative_test",
}


def wav_info(path):
    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sample_width = w.getsampwidth()
            sample_rate = w.getframerate()
            frames = w.getnframes()

        duration = frames / sample_rate if sample_rate else 0

        return {
            "channels": channels,
            "sample_width": sample_width,
            "sample_rate": sample_rate,
            "frames": frames,
            "duration": duration,
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def print_stats(name, folder):
    print()
    print("=" * 80)
    print(name)
    print(folder)
    print("=" * 80)

    if not folder.exists():
        print("ПАПКА НЕ НАЙДЕНА")
        return

    files = sorted(folder.glob("*.wav"))

    print(f"Количество WAV: {len(files)}")

    if not files:
        return

    sample_rates = Counter()
    channels = Counter()
    sample_widths = Counter()

    durations = []

    errors = []

    names = []

    for path in files:
        names.append(path.name)

        info = wav_info(path)

        if "error" in info:
            errors.append((path.name, info["error"]))
            continue

        sample_rates[info["sample_rate"]] += 1
        channels[info["channels"]] += 1
        sample_widths[info["sample_width"]] += 1
        durations.append(info["duration"])

    print()
    print("Sample rate:")
    for value, count in sample_rates.items():
        print(f"  {value} Hz: {count}")

    print()
    print("Channels:")
    for value, count in channels.items():
        print(f"  {value}: {count}")

    print()
    print("Sample width:")
    for value, count in sample_widths.items():
        print(f"  {value} bytes: {count}")

    if durations:
        durations = np.array(durations)

        print()
        print("Длительность:")
        print(f"  min:    {durations.min():.3f}s")
        print(f"  max:    {durations.max():.3f}s")
        print(f"  mean:   {durations.mean():.3f}s")
        print(f"  median: {np.median(durations):.3f}s")

    if errors:
        print()
        print("ОШИБКИ:")
        for filename, error in errors[:20]:
            print(f"  {filename}: {error}")

    print()
    print("Первые 15 файлов:")

    for filename in names[:15]:
        print(f"  {filename}")


def compare_sets():
    print()
    print("=" * 80)
    print("ПРОВЕРКА ПЕРЕСЕЧЕНИЙ")
    print("=" * 80)

    sets = {}

    for name, folder in SETS.items():

        if not folder.exists():
            continue

        sets[name] = {
            p.name
            for p in folder.glob("*.wav")
        }

    names = list(sets.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a = names[i]
            b = names[j]

            overlap = sets[a] & sets[b]

            print()
            print(f"{a} <-> {b}: {len(overlap)} совпадений")

            if overlap:
                for filename in sorted(overlap)[:10]:
                    print(f"  {filename}")


def inspect_special_folders():
    print()
    print("=" * 80)
    print("ДОПОЛНИТЕЛЬНЫЕ ИСТОЧНИКИ")
    print("=" * 80)

    candidates = [
        BASE_DIR / "wake_samples",
        BASE_DIR / "wakeword_training",
        BASE_DIR / "wakeword_training" / "lenya",
        BASE_DIR / "wakeword_training" / "lenya" / "lenya",
    ]

    for folder in candidates:

        if not folder.exists():
            continue

        print()
        print(f"{folder}")

        try:
            children = list(folder.iterdir())

            for child in children[:30]:
                kind = "DIR " if child.is_dir() else "FILE"
                print(f"  [{kind}] {child.name}")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():

    print()
    print("=" * 80)
    print("VOICEHELPER WAKE-WORD DATASET AUDIT")
    print("=" * 80)

    print(f"Base directory: {BASE_DIR}")
    print(f"Dataset directory: {DATA_DIR}")

    for name, folder in SETS.items():
        print_stats(name, folder)

    compare_sets()

    inspect_special_folders()

    print()
    print("=" * 80)
    print("АУДИТ ЗАВЕРШЁН")
    print("=" * 80)


if __name__ == "__main__":
    main()