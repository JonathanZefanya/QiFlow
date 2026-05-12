from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from utils.assets import asset_path


def load_spell_config() -> Dict[str, object]:
    custom_path = asset_path("config", "spells_custom.json")
    if custom_path.exists():
        return json.loads(custom_path.read_text(encoding="utf-8"))
    fallback = asset_path("config", "spells.json")
    if fallback.exists():
        return json.loads(fallback.read_text(encoding="utf-8"))
    return {}


def save_spell_config(data: Dict[str, object]) -> Path:
    custom_path = asset_path("config", "spells_custom.json")
    custom_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return custom_path
