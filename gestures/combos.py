from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque

from utils.timer import Cooldown, now


@dataclass
class ComboRule:
    key: str
    name: str
    sequence: List[str]
    window: float
    cooldown: Cooldown


class ComboSystem:
    def __init__(self, combos: List[Dict[str, object]]) -> None:
        self.combos: Dict[str, ComboRule] = {}
        for cfg in combos:
            key = str(cfg["key"])
            self.combos[key] = ComboRule(
                key=key,
                name=str(cfg.get("name", key)).title(),
                sequence=list(cfg["sequence"]),
                window=float(cfg.get("window", 2.5)),
                cooldown=Cooldown(float(cfg.get("cooldown", 4.0))),
            )
        self.history: Deque[Tuple[str, float]] = deque(maxlen=20)

    def register_spell(self, spell_key: str) -> Optional[ComboRule]:
        t = now()
        self.history.append((spell_key, t))
        for combo in self.combos.values():
            if not combo.cooldown.ready(t):
                continue
            if self._match(combo, t):
                combo.cooldown.trigger(t)
                return combo
        return None

    def _match(self, combo: ComboRule, t: float) -> bool:
        seq = combo.sequence
        if not seq:
            return False
        items = [item for item in self.history if (t - item[1]) <= combo.window]
        if len(items) < len(seq):
            return False
        keys = [item[0] for item in items]
        idx = len(keys) - len(seq)
        return keys[idx:] == seq
