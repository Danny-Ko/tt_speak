import os
import uuid
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from core.config import get_settings
from core.models import STTResponse, TTSRequest, ErrorResponse
from services.stt_service import load_whisper_model, unload_whisper_model, transcribe_audio
from services.tts_service import text_to_speech, list_available_voices
from services.audio_utils import (
    get_temp_path, cleanup_file, check_audio_length,
    ensure_data_dir
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Cantonese Voice API Service...")
    ensure_data_dir()
    print(f"Data directory: {settings.DATA_DIR}")
    print("Loading Whisper model (this may take a moment on first run)...")
    load_whisper_model()
    print("Service ready!")
    yield
    print("Shutting down...")
    unload_whisper_model()


app = FastAPI(
    title="Cantonese Voice API",
    description="Speech-to-Text and Text-to-Speech for Cantonese chat room messages. "
                "Optimized for short audio (max 60 seconds) with Traditional Chinese output.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LenientJSONMiddleware(BaseHTTPMiddleware):
    """Middleware to handle malformed JSON with raw control characters.
    
    This handles cases where users paste multi-line text into Swagger UI,
    which sends raw newline/control characters in JSON strings instead of
    properly escaped versions.
    
    Uses json.loads(strict=False) which:
    - Allows unescaped control characters (\\n, \\t, \\r)
    - Still parses otherwise valid JSON
    """
    
    async def dispatch(self, request: StarletteRequest, call_next):
        # Only process JSON requests
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return await call_next(request)
        
        # Read and buffer the body
        body = await request.body()
        
        if body:
            try:
                # Try normal strict parsing first
                json.loads(body)
                # If it works, no fix needed - restore body and continue
                request._body = body
            except json.JSONDecodeError:
                try:
                    # Strict parsing failed - try lenient parsing
                    data = json.loads(body, strict=False)
                    # Re-encode with proper escaping
                    fixed_body = json.dumps(data).encode("utf-8")
                    # Replace the body with fixed version
                    request._body = fixed_body
                except Exception:
                    # If even lenient parsing fails, let it through
                    # FastAPI will return the proper error
                    request._body = body
        else:
            request._body = body
        
        return await call_next(request)


app.add_middleware(LenientJSONMiddleware)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "cantonese-voice-api"}


@app.post(
    "/api/v1/stt",
    response_model=STTResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Speech-to-Text"]
)
async def speech_to_text_endpoint(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file (OGG/MP3/WAV, max 60 seconds)")
):
    """
    Convert Cantonese speech to Traditional Chinese text.

    - **audio**: Audio file upload (any format supported by FFmpeg)
    - Max duration: 60 seconds
    - Output: Traditional Chinese text (max 1000 characters)
    """
    temp_audio: Optional[str] = None
    temp_ogg: Optional[str] = None

    try:
        ext = os.path.splitext(audio.filename)[1] or ".ogg"
        temp_audio = get_temp_path(ext)

        content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(content)

        is_ok, duration = await check_audio_length(temp_audio)
        if not is_ok:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration:.1f}s (max {settings.MAX_AUDIO_SECONDS}s)"
            )

        text, confidence, processing_ms = await transcribe_audio(temp_audio)

        if not text:
            raise HTTPException(
                status_code=400,
                detail="No speech detected in audio"
            )

        if len(text) > settings.MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Transcribed text too long: {len(text)} chars (max {settings.MAX_TEXT_LENGTH} chars)"
            )

        return STTResponse(
            text=text,
            language="yue",
            duration_seconds=round(duration, 2),
            processing_time_ms=int(processing_ms),
            confidence=round(confidence, 3) if confidence else None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"STT processing failed: {str(e)}"
        )
    finally:
        if temp_audio:
            background_tasks.add_task(cleanup_file, temp_audio)
        if temp_ogg:
            background_tasks.add_task(cleanup_file, temp_ogg)


@app.post(
    "/api/v1/tts",
    response_class=FileResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["Text-to-Speech"]
)
async def text_to_speech_endpoint(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Convert Traditional Chinese text to Cantonese speech (OGG/Opus format).

    - **text**: Text to convert to speech (max 1000 characters)
    - **voice**: Optional voice name (default: zh-HK-HiuMaanNeural)
    - **rate**: Optional speech rate (e.g., "-10%", "+20%")
    - Returns: OGG audio file (Opus codec)
    """
    output_path: Optional[str] = None

    try:
        output_path, _ = await text_to_speech(
            text=request.text,
            voice=request.voice,
            rate=request.rate
        )

        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            path=output_path,
            media_type="application/ogg",
            filename=f"tts_{uuid.uuid4().hex[:8]}.ogg"
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        if output_path:
            background_tasks.add_task(cleanup_file, output_path)
        raise HTTPException(
            status_code=500,
            detail=f"TTS processing failed: {str(e)}"
        )


@app.get("/api/v1/voices", tags=["Text-to-Speech"])
async def get_voices():
    """List all available Cantonese (zh-HK) voices from edge-tts"""
    voices = await list_available_voices()
    return {
        "count": len(voices),
        "default_voice": settings.DEFAULT_VOICE,
        "voices": [
            {
                "name": v.get("ShortName"),
                "friendly_name": v.get("FriendlyName"),
                "gender": v.get("Gender"),
                "locale": v.get("Locale")
            }
            for v in voices
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )