from pathlib import Path
import random

import librosa
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]

WAKE_SAMPLES = PROJECT_DIR / "wake_samples"
NEGATIVE_CHECK = PROJECT_DIR / "negative_check"
FALSE_POSITIVES = PROJECT_DIR / "false_positive_samples"

SAMPLE_RATE = 16000

N_MFCC = 13
N_MELS = 32
HOP_LENGTH = 160
N_FFT = 400

# Количество случайных вариантов каждого файла.
AUGMENTATIONS_PER_FILE = 20

# Фиксированный seed, чтобы результаты можно было повторить.
RANDOM_SEED = 42

# Небольшие изменения.
GAIN_MIN = 0.65
GAIN_MAX = 1.35

NOISE_LEVEL_MIN = 0.002
NOISE_LEVEL_MAX = 0.015

SPEED_MIN = 0.93
SPEED_MAX = 1.07

# Максимальная случайная добавленная тишина.
SILENCE_MAX_SECONDS = 0.15


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# AUDIO
# ============================================================

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
        raise ValueError(
            f"Пустой WAV: {path}"
        )

    # Убираем DC offset.
    audio = (
        audio
        - np.mean(audio)
    )

    return audio


def normalize_peak(
    audio: np.ndarray,
) -> np.ndarray:

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak > 1e-6:
        audio = (
            audio / peak
        )

    return audio.astype(
        np.float32
    )


def augment_audio(
    audio: np.ndarray,
) -> np.ndarray:

    result = np.asarray(
        audio,
        dtype=np.float32,
    ).copy()

    # --------------------------------------------------------
    # GAIN
    # --------------------------------------------------------

    gain = random.uniform(
        GAIN_MIN,
        GAIN_MAX,
    )

    result *= gain

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    speed = random.uniform(
        SPEED_MIN,
        SPEED_MAX,
    )

    try:
        result = librosa.effects.time_stretch(
            result,
            rate=speed,
        )
    except Exception:
        pass

    # --------------------------------------------------------
    # NOISE
    # --------------------------------------------------------

    noise_level = random.uniform(
        NOISE_LEVEL_MIN,
        NOISE_LEVEL_MAX,
    )

    noise = (
        np.random.randn(
            result.size
        ).astype(
            np.float32
        )
        * noise_level
    )

    result += noise

    # --------------------------------------------------------
    # RANDOM SILENCE
    # --------------------------------------------------------

    left_silence = random.randint(
        0,
        int(
            SILENCE_MAX_SECONDS
            * SAMPLE_RATE
        ),
    )

    right_silence = random.randint(
        0,
        int(
            SILENCE_MAX_SECONDS
            * SAMPLE_RATE
        ),
    )

    if left_silence > 0:

        result = np.concatenate(
            [
                np.zeros(
                    left_silence,
                    dtype=np.float32,
                ),
                result,
            ]
        )

    if right_silence > 0:

        result = np.concatenate(
            [
                result,
                np.zeros(
                    right_silence,
                    dtype=np.float32,
                ),
            ]
        )

    return normalize_peak(
        result
    )


# ============================================================
# FEATURES
# ============================================================

def extract_features(
    audio: np.ndarray,
) -> np.ndarray:

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            "Пустой audio"
        )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # ВАЖНО:
    # Здесь оставляем исходный MFCC без
    # поканальной z-нормализации.
    #
    # Для DTW это позволяет сохранить
    # дополнительную спектральную информацию.
    return mfcc.astype(
        np.float32
    )


# ============================================================
# DTW
# ============================================================

def dtw_distance(
    template: np.ndarray,
    features: np.ndarray,
) -> float:

    distance, _ = librosa.sequence.dtw(
        X=template,
        Y=features,
        metric="euclidean",
        subseq=True,
        backtrack=True,
    )

    if distance.size == 0:
        return float("inf")

    value = float(
        np.min(
            distance[-1]
        )
        / max(
            template.shape[1],
            1,
        )
    )

    return value


def score_audio(
    templates: list[np.ndarray],
    audio: np.ndarray,
) -> float:

    if audio.size == 0:
        return float("inf")

    features = extract_features(
        audio
    )

    if features.shape[1] < 3:
        return float("inf")

    best = float("inf")

    for template in templates:

        try:

            score = dtw_distance(
                template,
                features,
            )

        except Exception:

            continue

        if score < best:
            best = score

    return best


def score_file(
    templates: list[np.ndarray],
    path: Path,
) -> float:

    try:

        audio = load_audio(
            path
        )

        return score_audio(
            templates,
            audio,
        )

    except Exception as error:

        print(
            f"[ERROR] {path.name}: {error}"
        )

        return float("inf")


# ============================================================
# AUGMENTED TEST
# ============================================================

def augmented_scores(
    templates: list[np.ndarray],
    path: Path,
) -> list[float]:

    audio = load_audio(
        path
    )

    scores = []

    # Оригинал тоже проверяем.
    original_score = score_audio(
        templates,
        audio,
    )

    scores.append(
        original_score
    )

    for _ in range(
        AUGMENTATIONS_PER_FILE
    ):

        augmented = augment_audio(
            audio
        )

        score = score_audio(
            templates,
            augmented,
        )

        if np.isfinite(score):

            scores.append(
                score
            )

    return scores


# ============================================================
# STATISTICS
# ============================================================

def print_augmented_stats(
    title: str,
    per_file: list[tuple[str, list[float]]],
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    all_scores = []

    for name, scores in per_file:

        finite = [
            value
            for value in scores
            if np.isfinite(value)
        ]

        if not finite:
            continue

        all_scores.extend(
            finite
        )

        print(
            f"{name:45s} "
            f"min={min(finite):.4f} "
            f"median={np.median(finite):.4f} "
            f"max={max(finite):.4f}"
        )

    if not all_scores:
        print(
            "Нет валидных результатов."
        )
        return

    values = np.asarray(
        all_scores,
        dtype=np.float64,
    )

    print()
    print(
        f"TOTAL SAMPLES : {len(values)}"
    )

    print(
        f"MIN           : {np.min(values):.6f}"
    )

    print(
        f"MEDIAN        : {np.median(values):.6f}"
    )

    print(
        f"MEAN          : {np.mean(values):.6f}"
    )

    print(
        f"MAX           : {np.max(values):.6f}"
    )


# ============================================================
# THRESHOLD
# ============================================================

def threshold_analysis(
    positive_scores: list[float],
    negative_scores: list[float],
):

    positive = np.asarray(
        [
            value
            for value in positive_scores
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    negative = np.asarray(
        [
            value
            for value in negative_scores
            if np.isfinite(value)
        ],
        dtype=np.float64,
    )

    if (
        positive.size == 0
        or negative.size == 0
    ):
        return

    print()
    print("=" * 70)
    print("AUGMENTED THRESHOLD ANALYSIS")
    print("=" * 70)

    # Важные пороги вокруг максимального
    # положительного расстояния.
    candidates = [
        float(
            np.percentile(
                positive,
                percentile,
            )
        )
        for percentile in (
            50,
            75,
            90,
            95,
            99,
            100,
        )
    ]

    # И несколько точек между
    # положительным и отрицательным диапазоном.
    pmax = float(
        np.max(positive)
    )

    nmin = float(
        np.min(negative)
    )

    if nmin > pmax:

        candidates.extend(
            np.linspace(
                pmax,
                nmin,
                8,
            ).tolist()
        )

    candidates = sorted(
        set(
            round(
                value,
                6,
            )
            for value in candidates
        )
    )

    print(
        f"{'THRESHOLD':>12} "
        f"{'POS HIT':>10} "
        f"{'NEG FP':>10}"
    )

    print(
        "-" * 38
    )

    for threshold in candidates:

        pos_hit = (
            np.mean(
                positive
                <= threshold
            )
            * 100
        )

        neg_fp = (
            np.mean(
                negative
                <= threshold
            )
            * 100
        )

        print(
            f"{threshold:12.6f} "
            f"{pos_hit:9.2f}% "
            f"{neg_fp:9.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "VOICEHELPER — DTW AUGMENTED TEST"
    )
    print("=" * 70)

    print(
        f"Seed: {RANDOM_SEED}"
    )

    print(
        f"Augmentations/file: "
        f"{AUGMENTATIONS_PER_FILE}"
    )

    print()

    # --------------------------------------------------------
    # TEMPLATES
    # --------------------------------------------------------

    template_paths = [
        WAKE_SAMPLES / "lenya_01.wav",
        WAKE_SAMPLES / "lenya_02.wav",
        WAKE_SAMPLES / "lenya_03.wav",
    ]

    templates = []

    print(
        "Loading templates..."
    )

    for path in template_paths:

        if not path.exists():

            print(
                f"[SKIP] {path}"
            )

            continue

        print(
            f"  {path.name}"
        )

        audio = load_audio(
            path
        )

        features = extract_features(
            audio
        )

        templates.append(
            features
        )

    if not templates:

        raise RuntimeError(
            "Нет DTW-шаблонов."
        )

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    positive_paths = [
        WAKE_SAMPLES / f"lenya_{i:02d}.wav"
        for i in range(4, 11)
    ]

    positive_paths = [
        path
        for path in positive_paths
        if path.exists()
    ]

    positive_per_file = []

    print()
    print(
        "Testing positive recordings..."
    )

    for path in positive_paths:

        scores = augmented_scores(
            templates,
            path,
        )

        positive_per_file.append(
            (
                path.name,
                scores,
            )
        )

    # --------------------------------------------------------
    # NEGATIVE
    # --------------------------------------------------------

    negative_paths = sorted(
        NEGATIVE_CHECK.glob(
            "*.wav"
        )
    )

    negative_per_file = []

    print()
    print(
        "Testing real negative recordings..."
    )

    for path in negative_paths:

        scores = augmented_scores(
            templates,
            path,
        )

        negative_per_file.append(
            (
                path.name,
                scores,
            )
        )

    # --------------------------------------------------------
    # OLD FALSE POSITIVES
    # --------------------------------------------------------

    fp_paths = sorted(
        FALSE_POSITIVES.glob(
            "*.wav"
        )
    )

    fp_per_file = []

    print()
    print(
        "Testing old false positives..."
    )

    for path in fp_paths:

        scores = augmented_scores(
            templates,
            path,
        )

        fp_per_file.append(
            (
                path.name,
                scores,
            )
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print_augmented_stats(
        "POSITIVE — lenya_04..10",
        positive_per_file,
    )

    print_augmented_stats(
        "REAL NEGATIVE",
        negative_per_file,
    )

    print_augmented_stats(
        "OLD FALSE POSITIVES",
        fp_per_file,
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    positive_scores = [
        score
        for _, scores in positive_per_file
        for score in scores
    ]

    negative_scores = [
        score
        for _, scores in negative_per_file
        for score in scores
    ]

    negative_scores.extend(
        score
        for _, scores in fp_per_file
        for score in scores
    )

    threshold_analysis(
        positive_scores,
        negative_scores,
    )

    # --------------------------------------------------------
    # WORST CASES
    # --------------------------------------------------------

    if positive_scores:

        finite_positive = [
            value
            for value in positive_scores
            if np.isfinite(value)
        ]

        if finite_positive:

            print()
            print(
                "=" * 70
            )
            print(
                "WORST POSITIVE"
            )
            print(
                "=" * 70
            )

            print(
                f"Maximum positive distance: "
                f"{max(finite_positive):.6f}"
            )

    if negative_scores:

        finite_negative = [
            value
            for value in negative_scores
            if np.isfinite(value)
        ]

        if finite_negative:

            print()
            print(
                "=" * 70
            )
            print(
                "MOST DANGEROUS NEGATIVE"
            )
            print(
                "=" * 70
            )

            print(
                f"Minimum negative distance: "
                f"{min(finite_negative):.6f}"
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