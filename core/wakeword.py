from pathlib import Path
import sys
import threading
import time

import numpy as np


PROJECT_DIR = (
    Path(__file__).resolve().parent.parent
)

OPENWAKEWORD_DIR = (
    PROJECT_DIR / "openWakeWord"
)

if not OPENWAKEWORD_DIR.exists():
    raise FileNotFoundError(
        f"Не найдена локальная папка openWakeWord:\n"
        f"{OPENWAKEWORD_DIR}"
    )

sys.path.insert(
    0,
    str(OPENWAKEWORD_DIR)
)

from openwakeword.model import Model


class WakeWordDetector:
    """
    Лёгкий постоянный wake-word detector.

    Поток:
        audio bytes
            ↓
        numpy int16
            ↓
        openWakeWord
            ↓
        score
            ↓
        callback при обнаружении
    """

    def __init__(
        self,
        model_path: str | Path,
        wakeword_name: str = "lenya",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        cooldown: float = 1.5,
        on_detect=None,
    ):
        self.model_path = Path(
            model_path
        )

        self.wakeword_name = wakeword_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.cooldown = cooldown
        self.on_detect = on_detect

        self.model = None

        self._lock = threading.Lock()

        self._last_detection = 0.0

        self._running = False

        self._last_score = 0.0

        self._detections = 0

    # ==================================================
    # START
    # ==================================================

    def start(self):
        """
        Загружает ONNX-модель.
        """

        if self._running:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Wake-word модель не найдена:\n"
                f"{self.model_path}"
            )

        print(
            "[Wake] Загружаю wake-word модель..."
        )

        self.model = Model(
            wakeword_models=[
                str(self.model_path)
            ],
            inference_framework="onnx",
        )

        available_models = list(
            self.model.models.keys()
        )

        print(
            "[Wake] Runtime models:",
            available_models,
        )

        if (
            self.wakeword_name
            not in available_models
        ):
            raise RuntimeError(
                f"Модель '{self.wakeword_name}' "
                "не найдена.\n"
                f"Доступные модели: "
                f"{available_models}"
            )

        self._running = True

        print(
            "[Wake] Wake-word detector запущен."
        )

        print(
            f"[Wake] Порог: "
            f"{self.threshold}"
        )

    # ==================================================
    # STOP
    # ==================================================

    def stop(self):
        """
        Останавливает detector.
        """

        self._running = False

        with self._lock:
            self.model = None

    # ==================================================
    # PROCESS AUDIO
    # ==================================================

    def process_audio(
        self,
        audio: bytes | np.ndarray,
    ) -> float:
        """
        Передаёт один аудиоблок
        в openWakeWord.

        audio:
            int16 PCM, 16 kHz, mono.
        """

        if not self._running:
            return 0.0

        if self.model is None:
            return 0.0

        if isinstance(
            audio,
            bytes,
        ):
            samples = np.frombuffer(
                audio,
                dtype=np.int16,
            )
        else:
            samples = np.asarray(
                audio,
                dtype=np.int16,
            )

        if samples.size == 0:
            return 0.0

        try:
            prediction = (
                self.model.predict(
                    samples
                )
            )

        except Exception as error:
            print(
                "[Wake] Ошибка inference:",
                error,
            )
            return 0.0

        score = float(
            prediction.get(
                self.wakeword_name,
                0.0,
            )
        )

        self._last_score = score

        # --------------------------------------------------
        # Threshold
        # --------------------------------------------------

        if score < self.threshold:
            return score

        # --------------------------------------------------
        # Cooldown
        # --------------------------------------------------

        now = time.monotonic()

        if (
            now - self._last_detection
            < self.cooldown
        ):
            return score

        self._last_detection = now
        self._detections += 1

        print(
            f"[Wake] ЛЁНЯ DETECTED "
            f"(score={score:.4f})"
        )

        if self.on_detect:

            try:
                self.on_detect(
                    score
                )

            except Exception as error:

                print(
                    "[Wake] Ошибка "
                    "on_detect:",
                    error,
                )

        return score

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def last_score(self) -> float:
        return self._last_score

    @property
    def detections(self) -> int:
        return self._detections

    @property
    def running(self) -> bool:
        return self._running

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):
        """
        Сбрасывает внутреннее состояние модели.
        """

        if self.model is not None:

            try:
                self.model.reset()

            except Exception:
                pass

        self._last_score = 0.0
        self._last_detection = 0.0