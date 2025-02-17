"""
rp_schema.py

This module defines the input schema for the runpod TTS worker.
Now the "text" field is expected to be a dictionary mapping speaker IDs to text strings,
allowing multi-speaker synthesis.
The parameter "long_text" is no longer passed externally and is used as True by default.
"""

from typing import Any, Dict

INPUT_SCHEMA: Dict[str, Any] = {
    "text": {"type": dict, "required": True},  # e.g. { "0": "Text for speaker 0", "1": "Text for speaker 1" }
    "speed": {"type": float, "required": False},
    "accentize": {"type": bool, "required": False},
    "volume": {"type": float, "required": False},
    "low_pass_filter_cutoff": {"type": int, "required": False},
    "enhance_audio": {"type": bool, "required": False}
}
