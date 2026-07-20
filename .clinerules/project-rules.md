# Project Rules: Cantonese Voice API Service

## Tech Stack & Architecture
- **Framework**: FastAPI (Python 3.11+, Async/Await)
- **STT (Speech-to-Text)**: `faster-whisper` (Optimized CTranslate2 engine, CPU-bound, `compute_type="int8"`).
- **TTS (Text-to-Speech)**: `edge-tts` (Microsoft Edge API wrapper, zero cost, supports Cantonese).
- **Audio Format**: OGG (.ogg, Opus codec via FFmpeg).
- **Environment**: Ubuntu Server (Dockerized).

## Core Requirements & Constraints
1. **Language Focus**: 
   - STT must be optimized for **Cantonese (`yue`)** with Traditional Chinese output (use initial prompts or post-processing when necessary).
   - TTS must use Cantonese voices (default: `zh-HK-HiuMaanNeural` or `zh-HK-WanLungNeural`).
2. **Audio Length**: Designed for short chat room messages (**max 15 seconds** per request). No complex queue systems needed, but fast response time (< 3 seconds) is critical.
3. **Zero Cost**: Absolutely no paid third-party APIs (no OpenAI API, no Google Cloud Speech). Everything must run locally or via free open-source mechanisms.
4. **Resource Management**: Always clean up temporary audio files in `/app/data` or use Python's `tempfile` / `BackgroundTasks` to prevent disk bloat on the NAS.

## Coding Style & Guidelines
- Keep the codebase modular (`main.py`, `core/`, `services/`).
- Handle all exceptions gracefully and return descriptive JSON errors (e.g., status 400 for bad format, 500 for processing error).
- Ensure all FastAPI endpoints include proper type hints and Pydantic models.