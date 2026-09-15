from pathlib import Path
import re

import librosa
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]

WAKE_SAMPLES = PROJECT_DIR / "wake_samples"
WAKE_VALIDATION = PROJECT_DIR / "wake_samples_validation"
NEGATIVE_CHECK = PROJECT_DIR / "negative_check"
FALSE_POSITIVES = PROJECT_DIR / "false_positive_samples"


SAMPLE_RATE = 16000

N_MFCC = 13
N_MELS = 32
HOP_LENGTH = 160
N_FFT = 400

# Чем меньше distance, тем сильнее совпадение.
#
# Пока это НЕ рабочий порог.
# Ниже программа сначала покажет реальные расстояния
# и предложит диапазон порогов.
#
# Порог намеренно оставляем None, чтобы не подгонять
# результат заранее.
THRESHOLD = None


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError("Пустой WAV")

    # Убираем DC offset.
    audio = audio - np.mean(audio)

    # Нормализуем громкость.
    peak = np.max(np.abs(audio))

    if peak > 1e-6:
        audio = audio / peak

    return audio


def extract_features(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    mean = np.mean(
        mfcc,
        axis=1,
        keepdims=True,
    )

    std = np.std(
        mfcc,
        axis=1,
        keepdims=True,
    )

    std = np.maximum(
        std,
        1e-6,
    )

    mfcc = (
        mfcc - mean
    ) / std

    return mfcc.astype(
        np.float32
    )


def make_template(path: Path) -> np.ndarray:
    audio = load_audio(path)

    features = extract_features(
        audio
    )

    if features.shape[1] < 3:
        raise ValueError(
            "Слишком короткая запись"
        )

    return features


def dtw_distance(
    template: np.ndarray,
    audio_features: np.ndarray,
) -> float:

    distance, _ = librosa.sequence.dtw(
        X=template,
        Y=audio_features,
        metric="euclidean",
        subseq=True,
        backtrack=True,
    )

    if distance.size == 0:
        return float("inf")

    # Стоимость лучшего subsequence match.
    #
    # Делим на длину шаблона, чтобы результат
    # был более сопоставим между разными записями.
    value = float(
        np.min(distance[-1])
        / max(
            template.shape[1],
            1,
        )
    )

    return value


def score_file(
    templates: list[np.ndarray],
    path: Path,
) -> float:

    try:
        audio = load_audio(path)
        features = extract_features(audio)

        if features.shape[1] < 3:
            return float("inf")

        best = float("inf")

        for template in templates:

            try:
                value = dtw_distance(
                    template,
                    features,
                )

            except Exception:
                continue

            if value < best:
                best = value

        return best

    except Exception as error:

        print(
            f"[ERROR] {path.name}: {error}"
        )

        return float("inf")


def print_stats(
    title: str,
    results: list[tuple[str, float]],
):
    values = np.array(
        [
            score
            for _, score in results
            if np.isfinite(score)
        ],
        dtype=np.float64,
    )

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if values.size == 0:
        print("Нет валидных результатов.")
        return

    print(
        f"count   : {len(values)}"
    )

    print(
        f"min     : {np.min(values):.6f}"
    )

    print(
        f"median  : {np.median(values):.6f}"
    )

    print(
        f"mean    : {np.mean(values):.6f}"
    )

    print(
        f"max     : {np.max(values):.6f}"
    )

    print()

    sorted_results = sorted(
        results,
        key=lambda item: item[1],
    )

    print("Лучшие совпадения:")

    for name, score in sorted_results[:10]:

        print(
            f"  {score:.6f}  {name}"
        )


def print_threshold_table(
    positive: list[tuple[str, float]],
    negative: list[tuple[str, float]],
):
    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    positive_values = np.array(
        [
            score
            for _, score in positive
            if np.isfinite(score)
        ],
        dtype=np.float64,
    )

    negative_values = np.array(
        [
            score
            for _, score in negative
            if np.isfinite(score)
        ],
        dtype=np.float64,
    )

    if (
        positive_values.size == 0
        or negative_values.size == 0
    ):
        print(
            "Недостаточно данных."
        )
        return

    # Порог выбирается по расстояниям,
    # а не заранее по желанию.
    candidates = sorted(
        set(
            np.concatenate(
                [
                    positive_values,
                    negative_values,
                ]
            ).tolist()
        )
    )

    # Берём характерные точки распределения,
    # чтобы вывод не был огромным.
    selected = set()

    for values in (
        positive_values,
        negative_values,
    ):
        for percentile in (
            0,
            10,
            25,
            50,
            75,
            90,
            95,
            99,
            100,
        ):
            selected.add(
                float(
                    np.percentile(
                        values,
                        percentile,
                    )
                )
            )

    selected = sorted(selected)

    print(
        f"{'THRESHOLD':>12} "
        f"{'POS HIT':>10} "
        f"{'NEG FP':>10}"
    )

    print("-" * 38)

    for threshold in selected:

        pos_hit = (
            np.mean(
                positive_values
                <= threshold
            )
            * 100
        )

        neg_fp = (
            np.mean(
                negative_values
                <= threshold
            )
            * 100
        )

        print(
            f"{threshold:12.6f} "
            f"{pos_hit:9.2f}% "
            f"{neg_fp:9.2f}%"
        )


def load_files(
    directory: Path,
    pattern: str = "*.wav",
) -> list[Path]:

    if not directory.exists():
        return []

    return sorted(
        directory.glob(pattern)
    )


def main():

    print()
    print("=" * 70)
    print("VOICEHELPER — DTW WAKE WORD TEST")
    print("=" * 70)

    print(
        f"Project: {PROJECT_DIR}"
    )

    print()

    # --------------------------------------------------
    # TRAIN TEMPLATES
    # --------------------------------------------------

    template_paths = [
        WAKE_SAMPLES / "lenya_01.wav",
        WAKE_SAMPLES / "lenya_02.wav",
        WAKE_SAMPLES / "lenya_03.wav",
    ]

    template_paths = [
        path
        for path in template_paths
        if path.exists()
    ]

    if not template_paths:
        raise RuntimeError(
            "Не найдены DTW-шаблоны."
        )

    print(
        f"Templates: {len(template_paths)}"
    )

    templates = []

    for path in template_paths:

        print(
            f"  loading {path.name}"
        )

        templates.append(
            make_template(path)
        )

    # --------------------------------------------------
    # INDEPENDENT POSITIVE TEST
    # --------------------------------------------------

    positive_paths = [
        WAKE_SAMPLES / f"lenya_{index:02d}.wav"
        for index in range(4, 11)
    ]

    positive_paths = [
        path
        for path in positive_paths
        if path.exists()
    ]

    positive_results = []

    print()
    print(
        "Testing independent positive recordings..."
    )

    for path in positive_paths:

        score = score_file(
            templates,
            path,
        )

        positive_results.append(
            (
                path.name,
                score,
            )
        )

    # --------------------------------------------------
    # REAL NEGATIVES
    # --------------------------------------------------

    negative_paths = load_files(
        NEGATIVE_CHECK
    )

    negative_results = []

    print()
    print(
        "Testing real negative recordings..."
    )

    for path in negative_paths:

        score = score_file(
            templates,
            path,
        )

        negative_results.append(
            (
                path.name,
                score,
            )
        )

    # --------------------------------------------------
    # FALSE POSITIVE SAMPLES
    # --------------------------------------------------

    fp_paths = load_files(
        FALSE_POSITIVES
    )

    fp_results = []

    print()
    print(
        "Testing old false-positive recordings..."
    )

    for path in fp_paths:

        score = score_file(
            templates,
            path,
        )

        fp_results.append(
            (
                path.name,
                score,
            )
        )

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print_stats(
        "INDEPENDENT POSITIVE — lenya_04..10",
        positive_results,
    )

    print_stats(
        "REAL NEGATIVE — negative_check",
        negative_results,
    )

    print_stats(
        "OLD FALSE POSITIVES",
        fp_results,
    )

    # --------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------

    combined_negative = (
        negative_results
        + fp_results
    )

    print_threshold_table(
        positive_results,
        combined_negative,
    )

    # --------------------------------------------------
    # EXPLICIT BEST/WORST
    # --------------------------------------------------

    if positive_results:

        best_positive = min(
            positive_results,
            key=lambda item: item[1],
        )

        worst_positive = max(
            positive_results,
            key=lambda item: item[1],
        )

        print()
        print(
            "=" * 70
        )
        print(
            "POSITIVE RANGE"
        )
        print(
            "=" * 70
        )

        print(
            "BEST  : "
            f"{best_positive[0]} "
            f"{best_positive[1]:.6f}"
        )

        print(
            "WORST : "
            f"{worst_positive[0]} "
            f"{worst_positive[1]:.6f}"
        )

    if combined_negative:

        valid_negative = [
            item
            for item in combined_negative
            if np.isfinite(item[1])
        ]

        if valid_negative:

            best_negative = min(
                valid_negative,
                key=lambda item: item[1],
            )

            worst_negative = max(
                valid_negative,
                key=lambda item: item[1],
            )

            print()
            print(
                "=" * 70
            )
            print(
                "NEGATIVE RANGE"
            )
            print(
                "=" * 70
            )

            print(
                "MOST DANGEROUS FP : "
                f"{best_negative[0]} "
                f"{best_negative[1]:.6f}"
            )

            print(
                "WORST MATCH       : "
                f"{worst_negative[0]} "
                f"{worst_negative[1]:.6f}"
            )

    print()
    print(
        "=" * 70
    )
    print(
        "TEST FINISHED"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()