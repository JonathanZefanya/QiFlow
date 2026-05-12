from __future__ import annotations

import math
import wave
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = BASE_DIR / "assets"


def _write_tone(path: Path, freq: float, duration: float = 0.25, volume: float = 0.4) -> None:
    sample_rate = 44100
    total_samples = int(sample_rate * duration)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(total_samples):
            value = volume * math.sin(2 * math.pi * freq * (i / sample_rate))
            wf.writeframesraw(int(value * 32767).to_bytes(2, byteorder="little", signed=True))


def generate_overlay() -> None:
    size = 256
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    color = (255, 200, 40, 180)
    cv2.circle(img, center, 110, color, 2)
    cv2.circle(img, center, 70, color, 1)
    for angle in range(0, 360, 30):
        x = int(center[0] + 95 * math.cos(math.radians(angle)))
        y = int(center[1] + 95 * math.sin(math.radians(angle)))
        cv2.line(img, center, (x, y), color, 1)
    cv2.imwrite(str(ASSET_DIR / "overlay_magic.png"), img)


def generate_sounds() -> None:
    _write_tone(ASSET_DIR / "sfx_charge.wav", 220)
    _write_tone(ASSET_DIR / "sfx_fire.wav", 330)
    _write_tone(ASSET_DIR / "sfx_shield.wav", 180)
    _write_tone(ASSET_DIR / "sfx_wind.wav", 260)
    _write_tone(ASSET_DIR / "sfx_lightning.wav", 440)
    _write_tone(ASSET_DIR / "sfx_spiral_charge.wav", 520, duration=0.45, volume=0.35)
    _write_tone(ASSET_DIR / "sfx_spiral_release.wav", 760, duration=0.18, volume=0.5)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    generate_overlay()
    generate_sounds()


if __name__ == "__main__":
    main()
