"""
Voice Input Component for GrievanceRoute.
Provides browser-based Web Speech API recognition supporting English, Hindi, and Marathi.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components


_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# Declare custom component
_voice_input_func = components.declare_component(
    "voice_input_component",
    path=str(_FRONTEND_DIR),
)


def render_voice_input(
    key: str = "voice_input_widget",
    default_lang: str = "en-IN",
) -> Optional[Dict[str, Any]]:
    """
    Renders the browser-based speech recognition microphone widget.
    
    Supports:
      - English (en-IN)
      - Hindi (hi-IN)
      - Marathi (mr-IN)

    Returns:
      dict with keys:
        - transcript: str
        - language: str (e.g. 'mr-IN')
        - language_name: str (e.g. 'मराठी (Marathi)')
        - is_clear: bool
        - timestamp: int
      or None if no recording event has occurred.
    """
    return _voice_input_func(key=key, default_lang=default_lang, default=None)
