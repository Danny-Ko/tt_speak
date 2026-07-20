import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # STT Settings
    WHISPER_MODEL: str = "base"
    WHISPER_LANGUAGE: str = "yue"
    COMPUTE_TYPE: str = "int8"
    INITIAL_PROMPT: str = "這是繁體中文廣東話對話。"

    # TTS Settings
    DEFAULT_VOICE: str = "zh-HK-HiuMaanNeural"
    ALTERNATIVE_VOICE: str = "zh-HK-WanLungNeural"
    OUTPUT_FORMAT: str = "ogg"

    # Limits
    MAX_AUDIO_SECONDS: int = 60
    MAX_TEXT_LENGTH: int = 1000

    # Paths
    # Use local ./data directory for development, /app/data for Docker
    DATA_DIR: str = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()