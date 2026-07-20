import re
import time
import os
import unicodedata
from typing import Optional

import edge_tts

from core.config import get_settings
from services.audio_utils import convert_to_opus, get_temp_path


def remove_emojis(text: str) -> str:
    """Remove emojis and other special symbols from text for TTS processing.
    
    Preserves:
    - CJK characters (Chinese, Japanese, Korean)
    - Letters (a-z, A-Z)
    - Numbers
    - Common punctuation and sentence terminators
    - Whitespace
    
    Removes:
    - Emojis
    - Special symbols and pictographs
    - Decorative characters
    """
    result = []
    for char in text:
        # Get Unicode category
        cat = unicodedata.category(char)
        
        # Keep based on category:
        # - CJK Unified Ideographs are in category "Lo" (Letter, Other)
        # - Ll/Lu/Lt/Lc (letters), Nd (numbers), Po (other punctuation)
        # - Pc (connector punctuation), Pd (dash punctuation), Ps/Pe (brackets)
        # - Pi/Pf (quotes), Zs/Zl/Zp (whitespace)
        keep_categories = {
            "Ll", "Lu", "Lt", "LC", "Lm", "Lo",  # Letters
            "Nd", "Nl", "No",                      # Numbers
            "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",  # Punctuation
            "Zs", "Zl", "Zp",                      # Separators/whitespace
            "Sc",                                   # Currency symbols
        }
        
        if cat in keep_categories:
            result.append(char)
        else:
            # Check if it's a known emoji-range character
            code_point = ord(char)
            # Don't remove if it's within CJK ranges
            is_cjk = (
                0x4E00 <= code_point <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= code_point <= 0x4DBF or    # CJK Unified Ideographs Extension A
                0x20000 <= code_point <= 0x2A6DF or  # CJK Unified Ideographs Extension B
                0x2A700 <= code_point <= 0x2B73F or  # CJK Unified Ideographs Extension C
                0x2B740 <= code_point <= 0x2B81F or  # CJK Unified Ideographs Extension D
                0x2B820 <= code_point <= 0x2CEAF or  # CJK Unified Ideographs Extension E
                0x3000 <= code_point <= 0x303F or    # CJK Symbols and Punctuation
                0xFF00 <= code_point <= 0xFFEF or    # Halfwidth and Fullwidth Forms
                0x3100 <= code_point <= 0x312F or    # Bopomofo
                0x31A0 <= code_point <= 0x31BF       # Bopomofo Extended
            )
            if is_cjk:
                result.append(char)
    
    cleaned = "".join(result)
    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

settings = get_settings()


async def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    rate: Optional[str] = None,
) -> tuple[str, int]:
    start_time = time.time()

    # Remove emojis from text before TTS processing
    cleaned_text = remove_emojis(text)
    if not cleaned_text.strip():
        raise ValueError("Text is empty after removing emojis")

    selected_voice = voice or settings.DEFAULT_VOICE
    selected_rate = rate or "+0%"

    temp_mp3 = get_temp_path(".mp3")

    try:
        communicate = edge_tts.Communicate(
            text=cleaned_text,
            voice=selected_voice,
            rate=selected_rate
        )
        await communicate.save(temp_mp3)

        final_output = get_temp_path(".ogg")
        await convert_to_opus(temp_mp3, final_output)

        processing_time = int((time.time() - start_time) * 1000)
        return final_output, processing_time

    finally:
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)


async def list_available_voices() -> list[dict]:
    voices = await edge_tts.list_voices()
    cantonese_voices = [
        v for v in voices
        if v.get("Locale", "").startswith("zh-HK")
    ]
    return cantonese_voices