"""
rp_schema.py

This module defines the input schema for the runpod TTS worker.
Now the "text" field is expected to be a list of pairs where each pair contains
a speaker ID and the corresponding text to synthesize.
The "long_text" mode is always enabled.
An additional parameter "output_format" may be provided with values "wav" (default) or "mp3".
"""

from typing import Any, Dict

INPUT_SCHEMA: Dict[str, Any] = {
    "text": {"type": list, "required": True},  # e.g. [[0, "Text for speaker 0"], [1, "Text for speaker 1"]]
    "speed": {"type": float, "required": False},
    "accentize": {"type": bool, "required": False},
    "volume": {"type": float, "required": False},
    "low_pass_filter_cutoff": {"type": int, "required": False},
    "enhance_audio": {"type": bool, "required": False},
    "output_format": {"type": str, "required": False}  # "wav" (default) or "mp3"
}
