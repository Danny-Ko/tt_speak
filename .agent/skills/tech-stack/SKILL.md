# Cantonese Voice API Service - Tech Stack

## Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Web Framework | FastAPI | Latest | Async API server |
| ASGI Server | uvicorn | Latest | Run FastAPI application |
| File Upload | python-multipart | Latest | Handle multipart/form-data requests |
| Speech-to-Text | faster-whisper | Latest | Local STT with CTranslate2 optimization |
| Text-to-Speech | edge-tts | Latest | Free Microsoft Edge TTS API wrapper |

## System Dependencies

| Component | Purpose |
|-----------|---------|
| FFmpeg | Audio format conversion (OGG/Opus codec support) |
| Python 3.11+ | Runtime environment with async/await support |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   main.py    │  │   core/      │  │    services/     │ │
│  │  (Endpoints) │  │ (Config/Models)│ │  (STT/TTS Logic) │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │              External Libraries                        │  │
│  │  faster-whisper (STT)  ◄───►  edge-tts (TTS)        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Technical Details

### 1. Speech-to-Text (faster-whisper)
- **Model**: Optimized CTranslate2 engine
- **Compute Type**: `int8` (CPU-efficient)
- **Language**: `yue` (Cantonese)
- **Output**: Traditional Chinese
- **Max Audio Length**: 15 seconds
- **Optimization**: 
  - Load model once at startup
  - Use `initial_prompt` for Traditional Chinese bias
  - Short audio = fast inference (< 3 seconds target)

### 2. Text-to-Speech (edge-tts)
- **Provider**: Microsoft Edge Online TTS (Free)
- **Default Voice**: `zh-HK-HiuMaanNeural` (Female)
- **Alternative Voice**: `zh-HK-WanLungNeural` (Male)
- **Output Format**: OGG (Opus codec via FFmpeg)
- **Async Support**: Full async/await for non-blocking I/O

### 3. Audio Format
- **Container**: OGG
- **Codec**: Opus
- **Processing**: FFmpeg for conversion/normalization
- **Why Opus**: Low latency, high quality for speech

### 4. Resource Management
- **Temp Files**: Use Python `tempfile` module
- **Auto-cleanup**: FastAPI `BackgroundTasks`
- **Target Dir**: `/app/data` (Docker volume)
- **Guarantee**: `try/finally` blocks + context managers

## Docker Deployment

```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    faster-whisper \
    edge-tts

# App setup
WORKDIR /app
COPY . /app
RUN mkdir -p /app/data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| STT Latency | < 3s | For 15s audio, int8 model |
| TTS Latency | < 2s | Network + synthesis |
| Memory Usage | ~500MB | Whisper small/tiny model |
| Concurrent Requests | 4-8 | CPU-bound (STT) |

## Model Selection (faster-whisper)

| Model | Speed | Accuracy | Memory | Recommendation |
|-------|-------|----------|--------|----------------|
| tiny | Fastest | Good | ~1GB RAM | Dev/Testing |
| base | Fast | Better | ~1GB RAM | Default |
| small | Medium | Best | ~2GB RAM | Production if CPU allows |

**Recommendation**: Start with `base` model, test with `tiny` if too slow.