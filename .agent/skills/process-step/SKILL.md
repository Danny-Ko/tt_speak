# Cantonese Voice API Service - Implementation Process

## Phase 1: Project Setup

### Step 1.1: Directory Structure

```
speak/
├── main.py                 # FastAPI app entry point
├── requirements.txt        # Dependencies
├── Dockerfile              # Container config
├── core/
│   ├── __init__.py
│   ├── config.py           # Settings (MAX_SECONDS, VOICE, etc.)
│   └── models.py           # Pydantic schemas
└── services/
    ├── __init__.py
    ├── stt_service.py      # faster-whisper wrapper
    ├── tts_service.py      # edge-tts wrapper
    └── audio_utils.py      # FFmpeg helpers + file cleanup
```

### Step 1.2: Install Dependencies

```bash
# System (Ubuntu)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Python
pip install fastapi uvicorn python-multipart faster-whisper edge-tts
```

### Step 1.3: Create requirements.txt

```txt
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
faster-whisper>=1.0.0
edge-tts>=6.1.0
pydantic>=2.0.0
```

---

## Phase 2: Core Configuration (core/)

### Step 2.1: core/config.py

```python
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # STT Settings
    WHISPER_MODEL: str = "base"        # tiny/base/small
    WHISPER_LANGUAGE: str = "yue"      # Cantonese
    COMPUTE_TYPE: str = "int8"          # CPU optimization
    INITIAL_PROMPT: str = "這是繁體中文廣東話對話。"  # Traditional Chinese bias
    
    # TTS Settings
    DEFAULT_VOICE: str = "zh-HK-HiuMaanNeural"
    ALTERNATIVE_VOICE: str = "zh-HK-WanLungNeural"
    OUTPUT_FORMAT: str = "ogg"          # Opus codec
    
    # Limits
    MAX_AUDIO_SECONDS: int = 15
    MAX_TEXT_LENGTH: int = 500
    
    # Paths
    DATA_DIR: str = "/app/data"         # For Docker volume
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
```

### Step 2.2: core/models.py

```python
from pydantic import BaseModel, field_validator
from typing import Optional

# ============ REQUEST MODELS ============

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None  # Uses default if not provided
    rate: Optional[str] = "+0%"   # e.g., "-10%", "+20%"
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v
    
    @field_validator('text')
    @classmethod
    def text_length_ok(cls, v):
        if len(v) > 500:
            raise ValueError("Text too long (max 500 chars)")
        return v

# ============ RESPONSE MODELS ============

class STTResponse(BaseModel):
    text: str
    language: str = "yue"
    duration_seconds: float
    processing_time_ms: int
    confidence: Optional[float] = None

class TTSResponse(BaseModel):
    audio_url: Optional[str] = None   # For saved file mode
    message: str
    processing_time_ms: int

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
```

---

## Phase 3: Service Implementation (services/)

### Step 3.1: services/audio_utils.py

```python
import os
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from fastapi import BackgroundTasks

from core.config import get_settings

settings = get_settings()

def ensure_data_dir():
    """Ensure /app/data exists for temp files"""
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)

def get_temp_path(suffix: str = ".ogg") -> str:
    """Generate temp file path in data dir"""
    ensure_data_dir()
    fd, path = tempfile.mkstemp(suffix=suffix, dir=settings.DATA_DIR)
    os.close(fd)
    return path

def cleanup_file(filepath: str):
    """Delete a file (for BackgroundTasks)"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass  # Best effort

async def get_audio_duration(filepath: str) -> float:
    """Get audio duration in seconds using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return 0.0
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0

async def check_audio_length(filepath: str) -> Tuple[bool, float]:
    """Check if audio is within MAX_AUDIO_SECONDS limit"""
    duration = await get_audio_duration(filepath)
    is_ok = duration <= settings.MAX_AUDIO_SECONDS
    return is_ok, duration

async def convert_to_opus(input_path: str, output_path: Optional[str] = None) -> str:
    """Convert any audio to OGG/Opus format"""
    if output_path is None:
        output_path = get_temp_path(".ogg")
    
    # Opus settings optimized for speech: 16kHz mono, low complexity
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "libopus",
        "-ar", "16000",      # 16kHz sample rate (speech)
        "-ac", "1",           # Mono
        "-b:a", "32k",        # 32kbps bitrate
        "-application", "voip",  # Optimize for voice
        "-v", "error",
        output_path
    ]
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {stderr.decode()}")
    
    return output_path
```

### Step 3.2: services/stt_service.py

```python
import time
from pathlib import Path
from typing import Optional, Tuple
from faster_whisper import WhisperModel

from core.config import get_settings

settings = get_settings()

# Singleton: load model once at app startup
_whisper_model: Optional[WhisperModel] = None

def load_whisper_model() -> WhisperModel:
    """Load Whisper model (call at startup)"""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device="cpu",
            compute_type=settings.COMPUTE_TYPE
        )
    return _whisper_model

def unload_whisper_model():
    """Cleanup (call at shutdown)"""
    global _whisper_model
    _whisper_model = None

async def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None
) -> Tuple[str, float, float]:
    """
    Transcribe audio file to Cantonese text.
    Returns: (text, avg_confidence, processing_time_ms)
    """
    start_time = time.time()
    model = load_whisper_model()
    
    # Use Cantonese by default
    lang = language or settings.WHISPER_LANGUAGE
    
    # Transcribe with Traditional Chinese bias
    segments, info = model.transcribe(
        audio_path,
        language=lang,
        initial_prompt=settings.INITIAL_PROMPT,
        beam_size=5,
        vad_filter=True,  # Voice Activity Detection
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Collect segments
    full_text = ""
    total_confidence = 0.0
    segment_count = 0
    
    for segment in segments:
        full_text += segment.text
        total_confidence += segment.avg_logprob
        segment_count += 1
    
    # Convert logprob to approximate confidence (0-1)
    avg_confidence = 0.0
    if segment_count > 0:
        avg_logprob = total_confidence / segment_count
        avg_confidence = 2 ** avg_logprob  # Log probability -> probability
    
    processing_time = (time.time() - start_time) * 1000
    
    return full_text.strip(), avg_confidence, processing_time
```

### Step 3.3: services/tts_service.py

```python
import time
import tempfile
from typing import Optional
import edge_tts

from core.config import get_settings
from services.audio_utils import convert_to_opus, get_temp_path

settings = get_settings()

async def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
    output_format: str = "ogg"
) -> tuple[str, int]:
    """
    Convert text to speech using edge-tts.
    First generates MP3/WAV, then converts to OGG/Opus.
    Returns: (output_file_path, processing_time_ms)
    """
    start_time = time.time()
    
    # Use defaults if not provided
    selected_voice = voice or settings.DEFAULT_VOICE
    selected_rate = rate or "+0%"
    
    # Create temp file for initial output (edge-tts uses MP3)
    temp_mp3 = get_temp_path(".mp3")
    
    try:
        # Generate speech with edge-tts
        communicate = edge_tts.Communicate(
            text=text,
            voice=selected_voice,
            rate=selected_rate
        )
        await communicate.save(temp_mp3)
        
        # Convert to OGG/Opus
        final_output = get_temp_path(".ogg")
        await convert_to_opus(temp_mp3, final_output)
        
        processing_time = int((time.time() - start_time) * 1000)
        return final_output, processing_time
        
    finally:
        # Clean up intermediate MP3
        import os
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

async def list_available_voices() -> list[dict]:
    """List all available Cantonese voices"""
    voices = await edge_tts.list_voices()
    cantonese_voices = [
        v for v in voices 
        if v.get("Locale", "").startswith("zh-HK")
    ]
    return cantonese_voices
```

---

## Phase 4: FastAPI Endpoints (main.py)

```python
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from core.config import get_settings
from core.models import STTResponse, TTSRequest, TTSResponse, ErrorResponse
from services.stt_service import load_whisper_model, unload_whisper_model, transcribe_audio
from services.tts_service import text_to_speech, list_available_voices
from services.audio_utils import (
    get_temp_path, cleanup_file, check_audio_length, 
    get_audio_duration, ensure_data_dir
)

settings = get_settings()

# ============ LIFESPAN (Startup/Shutdown) ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, unload at shutdown"""
    # Startup
    print("Loading Whisper model...")
    load_whisper_model()
    ensure_data_dir()
    print(f"Model loaded. Ready on port 8000.")
    yield
    # Shutdown
    print("Shutting down...")
    unload_whisper_model()

# ============ APP INITIALIZATION ============

app = FastAPI(
    title="Cantonese Voice API",
    description="Speech-to-Text and Text-to-Speech for Cantonese chat messages",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for chat room frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ HEALTH CHECK ============

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "service": "cantonese-voice-api"}

# ============ STT ENDPOINT ============

@app.post(
    "/api/v1/stt",
    response_model=STTResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def speech_to_text(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Audio file (OGG/MP3/WAV, max 15s)")
):
    """
    Convert Cantonese speech to Traditional Chinese text.
    
    - **audio**: Audio file upload (any format FFmpeg supports)
    - Limit: 15 seconds max
    """
    temp_audio: Optional[str] = None
    temp_ogg: Optional[str] = None
    
    try:
        # 1. Save uploaded file
        ext = os.path.splitext(audio.filename)[1] or ".ogg"
        temp_audio = get_temp_path(ext)
        
        content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(content)
        
        # 2. Check audio duration BEFORE processing
        is_ok, duration = await check_audio_length(temp_audio)
        if not is_ok:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long: {duration:.1f}s (max {settings.MAX_AUDIO_SECONDS}s)"
            )
        
        # 3. Convert to format Whisper likes (16kHz mono WAV via Opus)
        temp_ogg = await convert_to_opus(temp_audio)
        
        # 4. Transcribe
        text, confidence, processing_ms = await transcribe_audio(temp_ogg)
        
        if not text:
            raise HTTPException(
                status_code=400,
                detail="No speech detected in audio"
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
        # Cleanup files
        if temp_audio:
            background_tasks.add_task(cleanup_file, temp_audio)
        if temp_ogg:
            background_tasks.add_task(cleanup_file, temp_ogg)

# ============ TTS ENDPOINTS ============

@app.post(
    "/api/v1/tts",
    response_class=FileResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def text_to_speech_endpoint(
    request: TTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Convert Traditional Chinese text to Cantonese speech (OGG/Opus).
    Returns the audio file directly (application/ogg).
    
    - **text**: Text to speak (max 500 chars)
    - **voice**: Optional voice name (default: zh-HK-HiuMaanNeural)
    - **rate**: Optional speech rate (-20% to +20%)
    """
    output_path: Optional[str] = None
    
    try:
        # Generate speech
        output_path, processing_ms = await text_to_speech(
            text=request.text,
            voice=request.voice,
            rate=request.rate
        )
        
        # Schedule cleanup after response
        background_tasks.add_task(cleanup_file, output_path)
        
        return FileResponse(
            path=output_path,
            media_type="application/ogg",
            filename=f"tts_{uuid.uuid4().hex[:8]}.ogg"
        )
        
    except HTTPException:
        raise
    except ValidationError as e:
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

@app.get("/api/v1/voices")
async def get_voices():
    """List all available Cantonese (zh-HK) voices"""
    voices = await list_available_voices()
    return {
        "count": len(voices),
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

# ============ RUN ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Disable in production
    )
```

---

## Phase 5: Testing

### Step 5.1: Run the server

```bash
cd /home/dannyko/python/speak
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5.2: Test with curl

```bash
# Health check
curl http://localhost:8000/health

# STT (upload audio)
curl -X POST "http://localhost:8000/api/v1/stt" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@your_audio.ogg;type=audio/ogg"

# TTS (generate audio)
curl -X POST "http://localhost:8000/api/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，這是測試"}' \
  --output output.ogg

# List voices
curl http://localhost:8000/api/v1/voices
```

### Step 5.3: Test with FastAPI Docs

Open browser: `http://localhost:8000/docs`

---

## Phase 6: Docker Deployment

### Step 6.1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

# Install Python deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    faster-whisper \
    edge-tts \
    pydantic-settings

# App setup
WORKDIR /app
COPY . /app
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# Pre-download whisper model (optional, speeds up first request)
# RUN python -c "from faster_whisper import WhisperModel; m = WhisperModel('base', device='cpu', compute_type='int8')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6.2: Create docker-compose.yml

```yaml
version: '3.8'

services:
  voice-api:
    build: .
    container_name: cantonese-voice-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./whisper-cache:/app/.cache/huggingface  # Persist model
    environment:
      - WHISPER_MODEL=base
      - MAX_AUDIO_SECONDS=15
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Step 6.3: Deploy commands

```bash
# Build and run
cd /home/dannyko/python/speak
docker-compose up -d --build

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down
```

---

## Phase 7: Chat Room Integration Checklist

- [ ] Chat room client sends OGG audio blob to `/api/v1/stt`
- [ ] STT returns Traditional Chinese text for display
- [ ] User types text → sends to `/api/v1/tts`
- [ ] TTS returns OGG file → chat room plays via `<audio>` element
- [ ] Add loading states (STT ~3s, TTS ~2s)
- [ ] Handle errors gracefully (timeouts, no speech, etc.)
- [ ] Consider rate limiting per user
- [ ] Monitor disk usage in /app/data

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow first STT request | Model downloads on first use. Pre-download or wait. |
| No audio output | Check FFmpeg installation: `ffmpeg -version` |
| Memory usage too high | Use `tiny` model instead of `base`/`small` |
| STT accuracy low | Ensure audio is clear Cantonese, try `small` model |
| File cleanup failing | Ensure /app/data has write permissions |