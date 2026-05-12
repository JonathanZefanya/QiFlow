import json

from utils.assets import asset_path


def load_config() -> dict:
    config_path = asset_path("config", "config.json")
    default_path = asset_path("config", "default_config.json")
    if not config_path.exists() and default_path.exists():
        config_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}
