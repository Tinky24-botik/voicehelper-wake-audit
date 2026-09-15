from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"D:\VoiceHelper")
DEFAULT_LENYA = PROJECT_ROOT / "wakeword_training" / "lenya" / "lenya.onnx"

JARVIS_CANDIDATES = [
    PROJECT_ROOT / "openWakeWord" / "openwakeword" / "resources" / "models" / "hey_jarvis_v0.1.onnx",
    PROJECT_ROOT / "openWakeWord" / "openwakeword" / "resources" / "models" / "hey_jarvis_v0.1.tflite",
    PROJECT_ROOT / "openWakeWord" / "resources" / "models" / "hey_jarvis_v0.1.onnx",
    PROJECT_ROOT / "openWakeWord" / "resources" / "models" / "hey_jarvis_v0.1.tflite",
]


def find_jarvis() -> Path | None:
    for path in JARVIS_CANDIDATES:
        if path.exists():
            return path

    root = PROJECT_ROOT / "openWakeWord"
    if root.exists():
        matches = list(root.rglob("hey_jarvis_v0.1.onnx"))
        if matches:
            return matches[0]

    return None


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if rate != 16000:
        raise ValueError(f"{path.name}: sample rate {rate}, нужен 16000 Hz")
    if channels != 1:
        raise ValueError(f"{path.name}: channels={channels}, нужен mono")
    if width != 2:
        raise ValueError(f"{path.name}: sample width={width} bytes, нужен int16 PCM")

    return np.frombuffer(raw, dtype=np.int16)


def score_wav(model, audio: np.ndarray) -> float:
    chunk_size = 16000
    scores = []

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) == 0:
            continue

        prediction = model.predict(chunk)

        for value in prediction.values():
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                pass

    return max(scores) if scores else 0.0


def collect_files(folder: Path, limit: int) -> list[Path]:
    if not folder.exists():
        return []

    files = sorted(folder.glob("*.wav"))
    return files[:limit] if limit > 0 else files


def run_group(model, name: str, files: list[Path]):
    print(f"\n--- {name} ({len(files)} WAV) ---")

    results = []
    for path in files:
        try:
            audio = read_wav(path)
            score = score_wav(model, audio)
            results.append((path.name, score))
            print(f"{path.name:35s} {score:.6f}")
        except Exception as exc:
            print(f"{path.name:35s} ERROR: {exc}")

    if results:
        values = np.array([x[1] for x in results], dtype=np.float64)
        print(
            f"AVG={values.mean():.6f}  "
            f"MEDIAN={np.median(values):.6f}  "
            f"MAX={values.max():.6f}  "
            f"MIN={values.min():.6f}"
        )

    return results


def print_comparison(group_name, lenya_results, jarvis_results):
    lenya = dict(lenya_results)
    jarvis = dict(jarvis_results)
    names = sorted(set(lenya) | set(jarvis))

    print(f"\n=== A/B: {group_name} ===")
    print(f"{'WAV':35s} {'LENYA':>12s} {'JARVIS':>12s}")
    print("-" * 63)

    for name in names:
        l = lenya.get(name)
        j = jarvis.get(name)
        ls = f"{l:.6f}" if l is not None else "-"
        js = f"{j:.6f}" if j is not None else "-"
        print(f"{name:35s} {ls:>12s} {js:>12s}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Контрольный A/B тест lenya.onnx против hey_jarvis_v0.1 на одинаковых WAV."
    )

    parser.add_argument("--lenya", type=Path, default=DEFAULT_LENYA)
    parser.add_argument("--jarvis", type=Path, default=None)
    parser.add_argument(
        "--real",
        type=Path,
        default=PROJECT_ROOT / "wake_samples",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=PROJECT_ROOT / "wakeword_training" / "lenya" / "lenya" / "positive_test",
    )
    parser.add_argument(
        "--negative",
        type=Path,
        default=PROJECT_ROOT / "wakeword_training" / "lenya" / "lenya" / "negative_test",
    )
    parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if not args.lenya.exists():
        print(f"ОШИБКА: не найден lenya: {args.lenya}")
        return 1

    jarvis_path = args.jarvis or find_jarvis()

    if jarvis_path is None or not jarvis_path.exists():
        print("ОШИБКА: не найден hey_jarvis_v0.1.onnx/.tflite.")
        print("Укажи путь вручную через --jarvis.")
        return 1

    sys.path.insert(0, str(PROJECT_ROOT / "openWakeWord"))

    try:
        from openwakeword.model import Model
    except Exception as exc:
        print("ОШИБКА импорта openwakeword:")
        print(exc)
        print("\nЗапускай из D:\\VoiceHelper с тем же venv, что используется проектом.")
        return 1

    print("=" * 75)
    print("VoiceHelper — контрольный A/B тест wake-word")
    print("=" * 75)
    print(f"LENYA : {args.lenya}")
    print(f"JARVIS: {jarvis_path}")
    print(f"REAL  : {args.real}")
    print(f"SYNTH : {args.synthetic}")
    print(f"NEG   : {args.negative}")

    try:
        lenya_model = Model(
            wakeword_models=[str(args.lenya)],
            inference_framework="onnx",
        )
    except Exception as exc:
        print(f"\nОШИБКА загрузки lenya: {exc}")
        return 1

    try:
        jarvis_model = Model(
            wakeword_models=[str(jarvis_path)],
            inference_framework="onnx",
        )
    except Exception as exc:
        print(f"\nОШИБКА загрузки Jarvis: {exc}")
        return 1

    groups = [
        ("REAL", collect_files(args.real, args.limit)),
        ("SYNTHETIC POSITIVE", collect_files(args.synthetic, args.limit)),
        ("NEGATIVE", collect_files(args.negative, args.limit)),
    ]

    all_lenya = {}
    all_jarvis = {}

    for group_name, files in groups:
        if not files:
            print(f"\n--- {group_name}: WAV не найдены ---")
            continue

        lenya_results = run_group(lenya_model, group_name + " / LENYA", files)
        jarvis_results = run_group(jarvis_model, group_name + " / JARVIS", files)

        print_comparison(group_name, lenya_results, jarvis_results)

        all_lenya[group_name] = lenya_results
        all_jarvis[group_name] = jarvis_results

    print("\n" + "=" * 75)
    print("ИТОГ")
    print("=" * 75)

    for group_name, _files in groups:
        lr = all_lenya.get(group_name, [])
        jr = all_jarvis.get(group_name, [])

        if not lr or not jr:
            continue

        lv = np.array([x[1] for x in lr], dtype=np.float64)
        jv = np.array([x[1] for x in jr], dtype=np.float64)

        print(
            f"{group_name:20s} | "
            f"LENYA avg={lv.mean():.6f} max={lv.max():.6f} | "
            f"JARVIS avg={jv.mean():.6f} max={jv.max():.6f}"
        )

    print("\nГотово.")
    print("Не переобучаем модель после этого теста — сначала смотрим результаты.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
