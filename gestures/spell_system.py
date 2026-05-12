from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from utils.timer import Cooldown, now
from gestures.combos import ComboSystem
from gestures.detectors import (
    detect_circle_motion,
    detect_finger_gun,
    detect_fist,
    detect_open_palm,
    hand_size,
    thresholds_from_sensitivity,
)

from gestures.hand_tracker import HandData


@dataclass
class Spell:
    key: str
    name: str
    cooldown: Cooldown


@dataclass
class SpellEvent:
    key: str
    name: str
    is_combo: bool


@dataclass
class HandSpellEvent:
    label: str
    key: str
    name: str
    is_combo: bool


class SpellSystem:
    def __init__(self, cooldowns: Dict[str, float], sensitivity: Dict[str, float], combos: List[Dict[str, object]]) -> None:
        self.spells = {
            "charge_energy": Spell("charge_energy", "Charge Energy", Cooldown(cooldowns["charge_energy"])),
            "fire_punch": Spell("fire_punch", "Fire Punch", Cooldown(cooldowns["fire_punch"])),
            "energy_shield": Spell("energy_shield", "Energy Shield", Cooldown(cooldowns["energy_shield"])),
            "wind_blade": Spell("wind_blade", "Wind Blade", Cooldown(cooldowns["wind_blade"])),
            "lightning_shot": Spell("lightning_shot", "Lightning Shot", Cooldown(cooldowns["lightning_shot"])),
        }
        self.base_cooldowns = {k: v.cooldown.duration for k, v in self.spells.items()}
        self.motion_history: Dict[str, Deque[Tuple[int, int]]] = {
            "Left": deque(maxlen=32),
            "Right": deque(maxlen=32),
        }
        self.last_spell: Optional[Spell] = None
        self.thresholds = thresholds_from_sensitivity(sensitivity)
        self.combo_system = ComboSystem(combos)
        self.hand_cooldowns: Dict[str, Dict[str, Cooldown]] = {}

    def update_thresholds(self, sensitivity: Dict[str, float]) -> None:
        self.thresholds = thresholds_from_sensitivity(sensitivity)

    def _ensure_hand(self, label: str) -> Dict[str, Cooldown]:
        if label not in self.hand_cooldowns:
            self.hand_cooldowns[label] = {
                key: Cooldown(duration) for key, duration in self.base_cooldowns.items()
            }
        return self.hand_cooldowns[label]

    def get_cooldown_progress(self, label: str, key: str) -> float:
        if label not in self.hand_cooldowns:
            return 0.0
        cd = self.hand_cooldowns[label].get(key)
        if not cd:
            return 0.0
        return min(1.0, (now() - cd.last_trigger) / cd.duration)

    def update_motion(self, hands: List[HandData]) -> None:
        active = {hand.label for hand in hands}
        for label in list(self.motion_history.keys()):
            if label not in active:
                self.motion_history[label].clear()
        for hand in hands:
            self.motion_history[hand.label].append(hand.wrist)

    def detect_spell(self, hands: List[HandData]) -> Optional[Spell]:
        if not hands:
            return None

        two_hands_open = (
            len(hands) >= 2
            and detect_open_palm(hands[0].landmarks, self.thresholds)
            and detect_open_palm(hands[1].landmarks, self.thresholds)
        )
        if two_hands_open:
            return self.spells["energy_shield"]

        for hand in hands:
            scale = hand_size(hand.landmarks) * self.thresholds.circle_radius_scale
            if detect_circle_motion(self.motion_history[hand.label], min_radius=scale):
                return self.spells["wind_blade"]

        for hand in hands:
            if detect_finger_gun(hand.landmarks, self.thresholds):
                return self.spells["lightning_shot"]

        for hand in hands:
            if detect_fist(hand.landmarks, self.thresholds):
                return self.spells["fire_punch"]

        for hand in hands:
            if detect_open_palm(hand.landmarks, self.thresholds):
                return self.spells["charge_energy"]

        return None

    def update_multi(self, hands: List[HandData]) -> List[HandSpellEvent]:
        self.update_motion(hands)
        events: List[HandSpellEvent] = []

        two_hands_open = False
        if len(hands) >= 2:
            two_hands_open = (
                detect_open_palm(hands[0].landmarks, self.thresholds)
                and detect_open_palm(hands[1].landmarks, self.thresholds)
            )
            if two_hands_open:
                label = "Both"
                cooldowns = self._ensure_hand(label)
                cd = cooldowns["energy_shield"]
                t = now()
                if cd.ready(t):
                    cd.trigger(t)
                    combo = self.combo_system.register_spell("energy_shield")
                    if combo:
                        events.append(HandSpellEvent(label, combo.key, combo.name, True))
                    events.append(HandSpellEvent(label, "energy_shield", "Energy Shield", False))

        for hand in hands:
            label = hand.label
            cooldowns = self._ensure_hand(label)
            spell_key: Optional[str] = None
            if detect_circle_motion(self.motion_history[label], min_radius=hand_size(hand.landmarks) * self.thresholds.circle_radius_scale):
                spell_key = "wind_blade"
            elif detect_finger_gun(hand.landmarks, self.thresholds):
                spell_key = "lightning_shot"
            elif detect_fist(hand.landmarks, self.thresholds):
                spell_key = "fire_punch"
            elif detect_open_palm(hand.landmarks, self.thresholds):
                if not two_hands_open:
                    spell_key = "charge_energy"

            if spell_key:
                cd = cooldowns[spell_key]
                t = now()
                if cd.ready(t):
                    cd.trigger(t)
                    combo = self.combo_system.register_spell(spell_key)
                    if combo:
                        events.append(HandSpellEvent(label, combo.key, combo.name, True))
                    events.append(HandSpellEvent(label, spell_key, self.spells[spell_key].name, False))

        return events

    def update(self, hands: List[HandData]) -> Optional[SpellEvent]:
        self.update_motion(hands)
        spell = self.detect_spell(hands)
        if not spell:
            return None
        t = now()
        if spell.cooldown.ready(t):
            spell.cooldown.trigger(t)
            self.last_spell = spell
            combo = self.combo_system.register_spell(spell.key)
            if combo:
                return SpellEvent(combo.key, combo.name, True)
            return SpellEvent(spell.key, spell.name, False)
        return None

    def trigger_manual(self, spell_key: str) -> Optional[SpellEvent]:
        spell = self.spells.get(spell_key)
        if not spell:
            return None
        t = now()
        if spell.cooldown.ready(t):
            spell.cooldown.trigger(t)
            combo = self.combo_system.register_spell(spell.key)
            if combo:
                return SpellEvent(combo.key, combo.name, True)
            return SpellEvent(spell.key, spell.name, False)
        return None

    def trigger_manual_for(self, label: str, spell_key: str) -> Optional[HandSpellEvent]:
        if spell_key not in self.spells:
            return None
        cooldowns = self._ensure_hand(label)
        cd = cooldowns[spell_key]
        t = now()
        if cd.ready(t):
            cd.trigger(t)
            combo = self.combo_system.register_spell(spell_key)
            if combo:
                return HandSpellEvent(label, combo.key, combo.name, True)
            return HandSpellEvent(label, spell_key, self.spells[spell_key].name, False)
        return None
