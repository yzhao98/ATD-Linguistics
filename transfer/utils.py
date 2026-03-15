import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# Language family mapping for cross-lingual attention regularization
# Key: low-resource language, Value: list of high-resource sibling languages to regularize against
INDO_ARYAN_SIBLINGS = {
    "mr": ["hi", "ur", "si"],  # Marathi -> Hindi, Urdu, Sinhala
    "pa": ["hi", "ur", "si"],  # Punjabi -> Hindi, Urdu, Sinhala
    "ne": ["hi", "ur", "si"],  # Nepali -> Hindi, Urdu, Sinhala
    "hi": ["ur", "si"],        # Hindi -> Urdu, Sinhala
}


def get_sibling_languages(lang: str) -> List[str]:
    """Get list of sibling languages for cross-lingual regularization."""
    return INDO_ARYAN_SIBLINGS.get(lang, [])

