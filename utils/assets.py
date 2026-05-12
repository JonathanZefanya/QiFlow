from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def asset_path(*parts: str) -> Path:
    return BASE_DIR.joinpath(*parts)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
