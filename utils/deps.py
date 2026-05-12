from __future__ import annotations

import importlib
from typing import List, Tuple


def check_dependencies() -> Tuple[List[str], List[str]]:
    core = ["cv2", "numpy", "PIL"]
    optional = ["mediapipe", "pygame"]
    missing_core: List[str] = []
    missing_optional: List[str] = []
    for module in core:
        try:
            importlib.import_module(module)
        except Exception:
            missing_core.append(module)
    for module in optional:
        try:
            mod = importlib.import_module(module)
            if module == "mediapipe" and not hasattr(mod, "solutions"):
                missing_optional.append("mediapipe (solutions missing)")
        except Exception:
            missing_optional.append(module)
    return missing_core, missing_optional
