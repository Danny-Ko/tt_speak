import os
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from core.config import get_settings

settings = get_settings()


def ensure_data_dir() -> None:
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_temp_path(suffix: str = ".ogg") -> str:
    ensure_data_dir()
    fd, path = tempfile.mkstemp(suffix=suffix, dir=settings.DATA_DIR)
    os.close(fd)
    return path


def cleanup_file(filepath: str) -> None:
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


async def get_audio_duration(filepath: str) -> float:
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
    duration = await get_audio_duration(filepath)
    is_ok = duration <= settings.MAX_AUDIO_SECONDS
    return is_ok, duration


async def convert_to_opus(input_path: str, output_path: Optional[str] = None) -> str:
    if output_path is None:
        output_path = get_temp_path(".ogg")

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "libopus",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",
        "-application", "voip",
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