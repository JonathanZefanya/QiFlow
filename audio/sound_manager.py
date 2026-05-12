from __future__ import annotations

from typing import Dict, TYPE_CHECKING

try:
    import pygame
except Exception:  # pragma: no cover - optional dependency
    pygame = None

if TYPE_CHECKING:
    import pygame as pygame_typing

from utils.assets import asset_path


class SoundManager:
    def __init__(self, enabled: bool = True, volume: float = 0.7) -> None:
        self.enabled = enabled and pygame is not None
        self.volume = volume
        self.sounds: Dict[str, "pygame_typing.mixer.Sound"] = {}
        self.ready = False
        if self.enabled:
            try:
                pygame.mixer.init()
                self.ready = True
            except pygame.error:
                self.ready = False
        if self.ready:
            self._load_sounds()

    def _load_sounds(self) -> None:
        mapping = {
            "charge_energy": "sfx_charge.wav",
            "fire_punch": "sfx_fire.wav",
            "energy_shield": "sfx_shield.wav",
            "wind_blade": "sfx_wind.wav",
            "lightning_shot": "sfx_lightning.wav",
        }
        for key, filename in mapping.items():
            path = asset_path("assets", filename)
            if path.exists():
                sound = pygame.mixer.Sound(str(path))
                sound.set_volume(self.volume)
                self.sounds[key] = sound

    def play(self, key: str) -> None:
        if not self.enabled or not self.ready:
            return
        sound = self.sounds.get(key)
        if sound:
            sound.play()

    def toggle(self) -> None:
        self.enabled = not self.enabled

    def set_volume(self, volume: float) -> None:
        self.volume = volume
        for sound in self.sounds.values():
            sound.set_volume(self.volume)
