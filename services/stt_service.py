import time
from typing import Optional, Tuple

from faster_whisper import WhisperModel

from core.config import get_settings

settings = get_settings()

_whisper_model: Optional[WhisperModel] = None


def load_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {settings.WHISPER_MODEL}...")
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device="cpu",
            compute_type=settings.COMPUTE_TYPE
        )
        print("Whisper model loaded.")
    return _whisper_model


def unload_whisper_model() -> None:
    global _whisper_model
    _whisper_model = None


async def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None
) -> Tuple[str, float, float]:
    start_time = time.time()
    model = load_whisper_model()

    lang = language or settings.WHISPER_LANGUAGE

    segments, info = model.transcribe(
        audio_path,
        language=lang,
        initial_prompt=settings.INITIAL_PROMPT,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    full_text = ""
    total_confidence = 0.0
    segment_count = 0

    for segment in segments:
        full_text += segment.text
        total_confidence += segment.avg_logprob
        segment_count += 1

    avg_confidence = 0.0
    if segment_count > 0:
        avg_logprob = total_confidence / segment_count
        avg_confidence = 2 ** avg_logprob

    processing_time = (time.time() - start_time) * 1000

    return full_text.strip(), avg_confidence, processing_time