import json
import threading

from core.plugin_manager import PluginManager
from core.skill_registry import SkillRegistry
from core.router import CommandRouter
from core.command_parser import CommandParser
from core.command_processor import CommandProcessor
from core.stt import VoiceListener


class VoiceHelperEngine:
    """
    Оборачивает весь голосовой движок (плагины, модели,
    VoiceListener) в start()/stop(), чтобы им можно было
    управлять по требованию — из кнопки в UI или
    автоматически при автозапуске.
    """

    def __init__(self, on_state_change=None, on_show_info=None):
        self.on_state_change = on_state_change
        self.on_show_info = on_show_info

        self.running = False

        self.plugin_manager = None
        self.voice_listener = None
        self.processor = None

    def is_running(self) -> bool:
        return self.running

    def start(self):
        if self.running:
            return

        self.running = True

        def _startup():
            skill_registry = SkillRegistry()
            self.plugin_manager = PluginManager(skill_registry=skill_registry)

            print("=== VoiceHelper ===\n\nЗагрузка plugins...")
            self.plugin_manager.load_all()
            print(f"\nЗагружено plugins: {len(self.plugin_manager.get_plugin_ids())}")
            print(f"Доступно skills: {len(skill_registry.get_all())}")

            router = CommandRouter(skill_registry=skill_registry)
            parser = CommandParser()

            self.processor = CommandProcessor(
                router=router,
                parser=parser,
                on_state_change=self.on_state_change,
                on_show_info=self.on_show_info,
            )

            with open("config/settings.json", "r", encoding="utf-8") as f:
                app_settings = json.load(f)

            self.voice_listener = VoiceListener(
                processor=self.processor,
                trigger_word=app_settings.get("trigger", "лёня"),
                small_model_path=app_settings.get("small_model_path", "model_small"),
                big_model_path=app_settings.get("model_path", "model"),
                stt_mode=app_settings.get("stt_mode", "auto"),
                on_state_change=self.on_state_change,
            )
            self.voice_listener.start()

            print("\nГолосовой движок запущен.")

        threading.Thread(target=_startup, daemon=True).start()

    def stop(self):
        if not self.running:
            return

        self.running = False

        if self.voice_listener:
            self.voice_listener.stop()

            if self.voice_listener.thread:
                self.voice_listener.thread.join(timeout=3)

        if self.plugin_manager:
            self.plugin_manager.shutdown_all()

        print("\nГолосовой движок остановлен.")

    def handle_text_command(self, text: str):
        if self.processor is None:
            return

        threading.Thread(
            target=self.processor.process,
            args=(text,),
            daemon=True,
        ).start()