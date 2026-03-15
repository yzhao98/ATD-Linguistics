from typing import Dict, List, Tuple


# Minimal Indo-European groupings for quick experiments
INDO_EUROPEAN_PAIRS_DEFAULT: List[Tuple[str, str]] = [
    ("en", "mr"), ("en", "pa"), ("en", "ne"),
    ("en", "hi"), ("en", "ur"), ("en", "si"),
]


# M2M100 language codes often match ISO but keep a mapping space for exceptions
M2M_CODE_MAP: Dict[str, str] = {
    # Identity mapping by default; override if needed
    "en": "en",
    "mr": "mr",
    "pa": "pa",
    "ne": "ne",
    "hi": "hi",
    "ur": "ur",
    "si": "si",
}

