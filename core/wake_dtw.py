from pathlib import Path

import librosa
import numpy as np


class WakeWordDTW:
    """
    Экспериментальный офлайн wake-word detector.

    Сравнивает текущий аудиофрагмент
    с несколькими эталонными записями
    через:

        audio
          ↓
        MFCC
          ↓
        DTW
          ↓
        normalized distance
    """

    def __init__(
        self,
        samples_dir: str | Path,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 32,
        hop_length: int = 160,
        n_fft: int = 400,
    ):
        self.samples_dir = Path(
            samples_dir
        )

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.templates: list[np.ndarray] = []

        self._load_templates()

        if not self.templates:
            raise RuntimeError(
                "Не найдено ни одного "
                "wake-word образца."
            )

    # ==================================================
    # LOAD TEMPLATES
    # ==================================================

    def _load_templates(self) -> None:
        files = sorted(
            self.samples_dir.glob(
                "lenya_*.wav"
            )
        )

        if not files:
            return

        print(
            f"[WakeDTW] Найдено образцов: "
            f"{len(files)}"
        )

        for path in files:

            try:
                audio, _ = librosa.load(
                    path,
                    sr=self.sample_rate,
                    mono=True,
                )

                features = (
                    self._extract_features(
                        audio
                    )
                )

                if features.shape[1] < 3:
                    continue

                self.templates.append(
                    features
                )

                print(
                    f"[WakeDTW] Загружен: "
                    f"{path.name}"
                )

            except Exception as error:
                print(
                    f"[WakeDTW] Ошибка "
                    f"{path.name}: {error}"
                )

    # ==================================================
    # MFCC
    # ==================================================

    def _extract_features(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        Преобразует аудио в MFCC.

        Возвращает:

            [n_mfcc, time]
        """

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return np.empty(
                (self.n_mfcc, 0),
                dtype=np.float32,
            )

        # Убираем постоянную составляющую.
        audio = (
            audio
            - np.mean(audio)
        )

        # Нормализация громкости.
        peak = np.max(
            np.abs(audio)
        )

        if peak > 1e-6:
            audio = audio / peak

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Нормализуем каждый MFCC-канал.
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

    # ==================================================
    # SCORE
    # ==================================================

    # Порог RMS-энергии (уже после DC-offset removal,
    # до деления на peak): если фрагмент тише этого
    # значения, считаем его тишиной/шумом и НЕ гоняем
    # через DTW вообще.
    #
    # Почему это обязательно: при делении на std, который
    # у тихого/шумового фрагмента близок к нулю, признаки
    # "взрываются" в вырожденный вектор, который DTW может
    # случайно посчитать похожим на что угодно. Порог по
    # энергии отсекает это до того, как дойдёт до DTW.
    SILENCE_RMS_THRESHOLD = 0.01

    def _is_silence(
        self,
        audio: np.ndarray,
    ) -> bool:
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return True

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(audio)
                )
            )
        )

        return rms < self.SILENCE_RMS_THRESHOLD

    def score(
        self,
        audio: np.ndarray,
    ) -> float:
        """
        Возвращает минимальную DTW-дистанцию
        до любого эталона.

        Меньше = больше похоже.

        Фрагменты ниже порога энергии (тишина/слабый
        шум) сразу возвращают inf, не доходя до DTW -
        там нормализация вырождается и даёт ложно
        низкий (уверенный) score.
        """

        if self._is_silence(audio):
            return float("inf")

        features = (
            self._extract_features(
                audio
            )
        )

        if features.shape[1] < 3:
            return float("inf")

        best_score = float("inf")

        for template in self.templates:

            try:
                distance, _ = (
                    librosa.sequence.dtw(
                        X=template,
                        Y=features,
                        metric="euclidean",
                        subseq=True,
                        backtrack=True,
                    )
                )

            except Exception:
                continue

            if distance.size == 0:
                continue

            # При subseq=True последняя строка
            # содержит стоимость завершения
            # совпадения шаблона внутри сигнала.
            value = float(
                np.min(
                    distance[-1]
                )
                / max(
                    template.shape[1],
                    1,
                )
            )

            if value < best_score:
                best_score = value

        return best_score

    def detect(
        self,
        audio: np.ndarray,
        threshold: float,
    ) -> bool:
        """
        Проверяет, превышено ли
        заданное качество совпадения.
        """

        return (
            self.score(audio)
            <= threshold
        )