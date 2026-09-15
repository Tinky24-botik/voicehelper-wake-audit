import json
import os
import queue
import threading
import difflib
import ctypes

import sounddevice as sd

os.environ["VOSK_LOG_LEVEL"] = "-1"

import vosk

from core.connectivity import has_internet
from core.groq_stt import recognize_with_groq
from core.notifier import notify

THREAD_PRIORITY_BELOW_NORMAL = -1


def _lower_current_thread_priority():
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentThread()
        kernel32.SetThreadPriority(handle, THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass


class VoiceListener:
    TRIGGER_MATCH_CUTOFF = 0.75
    WAITING_FOR_COMMAND_TIMEOUT_SECONDS = 8

    def __init__(
        self,
        processor,
        trigger_word: str = "лёня",
        small_model_path: str = "model_small",
        big_model_path: str = "model",
        stt_mode: str = "auto",
        on_state_change=None,
    ):
        self.processor = processor
        self.trigger_word = trigger_word.lower().strip()
        self.small_model_path = small_model_path
        self.big_model_path = big_model_path
        self.stt_mode = stt_mode
        self.on_state_change = on_state_change

        self.running = False
        self.thread = None
        self.audio_queue = queue.Queue()
        self.small_model = None
        self.recognizer = None
        self.samplerate = 16000
        self.big_model = None
        self.big_model_ready = False
        self.waiting_for_command = False
        self.audio_buffer = bytearray()

        self._waiting_timer = None

    # ==================================================
    # START / STOP
    # ==================================================
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        threading.Thread(target=self._load_big_model, daemon=True).start()

    def stop(self):
        self.running = False
        self._cancel_waiting_timer()

    # ==================================================
    # BIG MODEL
    # ==================================================
    def _load_big_model(self):
        _lower_current_thread_priority()

        print("\n[Voice] Гружу точную модель в фоне...")
        try:
            self.big_model = vosk.Model(self.big_model_path)
            print("[Voice] Прогреваю точную модель...")
            warmup_recognizer = vosk.KaldiRecognizer(self.big_model, self.samplerate)
            warmup_recognizer.AcceptWaveform(bytes(3200))
            warmup_recognizer.FinalResult()
            self.big_model_ready = True
            print("[Voice] Точная модель готова.")
        except Exception as error:
            print(f"[Voice] Не удалось загрузить точную модель: {error}")

    # ==================================================
    # WAITING-FOR-COMMAND TIMEOUT
    # ==================================================
    def _start_waiting_timer(self):
        self._cancel_waiting_timer()
        self._waiting_timer = threading.Timer(
            self.WAITING_FOR_COMMAND_TIMEOUT_SECONDS,
            self._on_waiting_timeout,
        )
        self._waiting_timer.daemon = True
        self._waiting_timer.start()

    def _cancel_waiting_timer(self):
        if self._waiting_timer:
            self._waiting_timer.cancel()
            self._waiting_timer = None

    def _on_waiting_timeout(self):
        self.waiting_for_command = False
        notify("Не дождался команду после триггера.")

    # ==================================================
    # MAIN VOICE LOOP
    # ==================================================
    def _run(self):
        try:
            print("\n[Voice] Гружу лёгкую модель...")
            try:
                self.small_model = vosk.Model(self.small_model_path)
            except Exception as error:
                print(f"[Voice] ОШИБКА загрузки лёгкой модели: {error}")
                self.running = False
                return

            print("[Voice] Лёгкая модель готова.")
            try:
                device = sd.query_devices(kind="input")
            except Exception as error:
                print(f"[Voice] ОШИБКА получения микрофона: {error}")
                self.running = False
                return

            samplerate = int(device["default_samplerate"])
            self.samplerate = samplerate
            print(f"[Voice] Микрофон: {device['name']}\n[Voice] Sample rate: {samplerate}")

            try:
                self.recognizer = vosk.KaldiRecognizer(self.small_model, samplerate)
            except Exception as error:
                print(f"[Voice] ОШИБКА создания распознавателя: {error}")
                self.running = False
                return

            print(f"\n[Voice] Готов. Триггер: «{self.trigger_word}»\n[Voice] Голосовой listener запущен.")

            try:
                with sd.RawInputStream(
                    samplerate=samplerate, blocksize=8000, dtype="int16",
                    channels=1, latency="high", callback=self._audio_callback
                ):
                    while self.running:
                        try:
                            data = self.audio_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        self._process_audio(data)
            except Exception as error:
                print(f"[Voice] ОШИБКА аудиопотока: {error}")
                self.running = False
                return

        except Exception as error:
            print(f"[Voice] КРИТИЧЕСКАЯ ОШИБКА: {error}")
            self.running = False

    # ==================================================
    # AUDIO CALLBACK
    # ==================================================
    def _audio_callback(self, indata, frames, time, status):
        if status: print(f"[Voice][Audio] {status}")
        if not self.running: return
        self.audio_queue.put(bytes(indata))

    # ==================================================
    # AUDIO PROCESSING
    # ==================================================
    def _process_audio(self, data: bytes):
        if self.recognizer is None: return
        self.audio_buffer.extend(data)

        try:
            accepted = self.recognizer.AcceptWaveform(data)
        except Exception as error:
            print(f"[Voice] Ошибка распознавания: {error}")
            return

        if not accepted:
            try:
                partial_res = json.loads(self.recognizer.PartialResult())
                partial_text = partial_res.get("partial", "").strip()
                if self.trigger_word in partial_text.lower():
                    if self.on_state_change:
                        self.on_state_change("processing")
            except Exception:
                pass
            return

        try:
            result = json.loads(self.recognizer.Result())
        except Exception as error:
            print(f"[Voice] Ошибка чтения результата Vosk: {error}")
            self.audio_buffer = bytearray()
            return

        quick_text = result.get("text", "").strip()
        audio_bytes = bytes(self.audio_buffer)
        self.audio_buffer = bytearray()

        if not quick_text: return
        self._handle_phrase(quick_text, audio_bytes)

    # ==================================================
    # TRIGGER MATCHING
    # ==================================================
    def _find_trigger_index(self, words: list) -> int | None:
        if self.trigger_word in words: return words.index(self.trigger_word)
        close = difflib.get_close_matches(self.trigger_word, words, n=1, cutoff=self.TRIGGER_MATCH_CUTOFF)
        if close: return words.index(close[0])
        return None

    # ==================================================
    # PHRASE HANDLING
    # ==================================================
    def _is_actionable(self, quick_text: str) -> bool:
        words = quick_text.lower().split()

        if self.processor.has_pending(): return True
        if self.waiting_for_command: return True
        if self._find_trigger_index(words) is not None: return True
        return False

    def _handle_phrase(self, quick_text: str, audio_bytes: bytes):
        print(f"[Voice] услышал (черновик): {quick_text}")
        if not self._is_actionable(quick_text): return

        if self.on_state_change: self.on_state_change("processing")

        text = self._recognize_precisely(quick_text, audio_bytes)
        print(f"[Voice] точный текст: {text}")

        self._route_text(text)

    # ==================================================
    # РАСПОЗНАВАНИЕ
    # ==================================================
    def _recognize_precisely(self, fallback_text: str, audio_bytes: bytes) -> str:
        if self.stt_mode == "offline": return self._recognize_with_big_model(fallback_text, audio_bytes)
        if self.stt_mode == "online":
            try: return recognize_with_groq(audio_bytes, self.samplerate)
            except Exception as error:
                print(f"[Voice] Groq недоступен ({error}), офлайн-резерв")
                return self._recognize_with_big_model(fallback_text, audio_bytes)
        if not has_internet(): return self._recognize_with_big_model(fallback_text, audio_bytes)
        try: return recognize_with_groq(audio_bytes, self.samplerate)
        except Exception as error:
            print(f"[Voice] Groq недоступен ({error}), офлайн-резерв")
            return self._recognize_with_big_model(fallback_text, audio_bytes)

    def _recognize_with_big_model(self, fallback_text: str, audio_bytes: bytes) -> str:
        if not self.big_model_ready:
            print("[Voice] Точная модель ещё грузится, использую черновой текст.")
            return fallback_text
        try:
            recognizer = vosk.KaldiRecognizer(self.big_model, self.samplerate)
            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "").strip()
            return text if text else fallback_text
        except Exception as error:
            print(f"[Voice] Ошибка точной модели: {error}")
            return fallback_text

    # ==================================================
    # МАРШРУТИЗАЦИЯ
    # ==================================================
    def _route_text(self, text: str):
        normalized_text = text.lower().strip()
        words = normalized_text.split()

        if self.processor.has_pending():
            if self._find_trigger_index(words) is not None:
                self.processor.clear_pending()
                self._handle_trigger(text, words)
                return
            print(f"[Voice] Ответ: {text}")
            self.processor.process(text)
            return

        if self.waiting_for_command:
            self.waiting_for_command = False
            self._cancel_waiting_timer()
            print(f"[Voice] Команда: {text}")
            self.processor.process(text)
            return

        self._handle_trigger(text, words)

    def _handle_trigger(self, text: str, words: list):
        trigger_index = self._find_trigger_index(words)
        if trigger_index is None: return
        original_words = text.split()
        command_words = original_words[trigger_index + 1:]
        command = " ".join(command_words).strip()

        if not command:
            self.waiting_for_command = True
            self._start_waiting_timer()
            print("[Voice] Триггер активирован.\n[Voice] Жду команду...")

            if self.on_state_change:
                self.on_state_change("passive")

            return

        print(f"[Voice] Команда: {command}")
        self.processor.process(command)