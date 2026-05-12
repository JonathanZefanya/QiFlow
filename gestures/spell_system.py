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
    detect_spiral_qi_activation,
    hand_size,
    palm_center,
    palm_depth,
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


@dataclass
class SpiralQiState:
    label: str
    charging: bool = False
    ready: bool = False
    released: bool = False
    start_time: float = 0.0
    charge_duration: float = 0.0
    anchor: Tuple[int, int] = (0, 0)
    previous_anchor: Optional[Tuple[int, int]] = None
    previous_size: float = 0.0
    previous_depth: float = 0.0
    stable_start: float = 0.0
    last_seen: float = 0.0


class SpellSystem:
    def __init__(
        self,
        cooldowns: Dict[str, float],
        sensitivity: Dict[str, float],
        combos: List[Dict[str, object]],
        spiral_qi_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.spiral_qi_config = {
            "enabled": True,
            "cooldown": 3.0,
            "charge_time": 0.8,
            "max_charge_time": 2.0,
            "release_speed": 25,
            **(spiral_qi_config or {}),
        }
        self.spells = {
            "charge_energy": Spell("charge_energy", "Charge Energy", Cooldown(cooldowns["charge_energy"])),
            "fire_punch": Spell("fire_punch", "Fire Punch", Cooldown(cooldowns["fire_punch"])),
            "energy_shield": Spell("energy_shield", "Energy Shield", Cooldown(cooldowns["energy_shield"])),
            "wind_blade": Spell("wind_blade", "Wind Blade", Cooldown(cooldowns["wind_blade"])),
            "lightning_shot": Spell("lightning_shot", "Lightning Shot", Cooldown(cooldowns["lightning_shot"])),
            "spiral_qi_sphere": Spell(
                "spiral_qi_sphere",
                "Spiral Qi Sphere",
                Cooldown(float(cooldowns.get("spiral_qi_sphere", self.spiral_qi_config["cooldown"]))),
            ),
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
        self.spiral_states: Dict[str, SpiralQiState] = {
            "Left": SpiralQiState("Left"),
            "Right": SpiralQiState("Right"),
        }

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

    def get_spiral_qi_states(self) -> Dict[str, Dict[str, object]]:
        data: Dict[str, Dict[str, object]] = {}
        max_charge = float(self.spiral_qi_config.get("max_charge_time", 2.0))
        for label, state in self.spiral_states.items():
            if not state.charging:
                continue
            data[label] = {
                "anchor": state.anchor,
                "charge": min(1.0, state.charge_duration / max_charge),
                "ready": state.ready,
                "duration": state.charge_duration,
            }
        return data

    def _update_spiral_qi(self, hands: List[HandData]) -> List[HandSpellEvent]:
        events: List[HandSpellEvent] = []
        if not bool(self.spiral_qi_config.get("enabled", True)):
            return events

        t = now()
        active_labels = {hand.label for hand in hands}
        charge_time = float(self.spiral_qi_config.get("charge_time", 0.8))
        max_charge = float(self.spiral_qi_config.get("max_charge_time", 2.0))
        release_speed = float(self.spiral_qi_config.get("release_speed", 25))

        for label, state in self.spiral_states.items():
            if label not in active_labels and state.charging:
                state.charging = False
                state.ready = False
                state.charge_duration = 0.0
                state.stable_start = 0.0

        for hand in hands:
            label = hand.label
            state = self.spiral_states.setdefault(label, SpiralQiState(label))
            cooldowns = self._ensure_hand(label)
            cd = cooldowns["spiral_qi_sphere"]
            anchor = palm_center(hand.landmarks)
            size = hand_size(hand.landmarks)
            depth = palm_depth(hand.landmarks)
            active = detect_spiral_qi_activation(hand.landmarks, self.thresholds)

            stable = False
            fast_motion = False
            forward_push = False
            if state.previous_anchor is not None:
                move = ((anchor[0] - state.previous_anchor[0]) ** 2 + (anchor[1] - state.previous_anchor[1]) ** 2) ** 0.5
                stable = move < max(8.0, size * 0.12)
                size_delta = size - state.previous_size
                depth_delta = depth - state.previous_depth
                forward_push = size_delta > max(6.0, size * 0.06) or depth_delta < -0.014
                fast_motion = move > release_speed

            if state.charging and state.ready and (forward_push or fast_motion) and cd.ready(t):
                cd.trigger(t)
                state.anchor = anchor
                state.charging = False
                state.ready = False
                state.released = True
                events.append(HandSpellEvent(label, "spiral_qi_sphere", "Spiral Qi Sphere", False))
                state.previous_anchor = anchor
                state.previous_size = size
                state.previous_depth = depth
                continue

            if active and cd.ready(t):
                if not state.charging:
                    state.charging = True
                    state.ready = False
                    state.released = False
                    state.start_time = t
                    state.stable_start = t
                    state.charge_duration = 0.0
                elif stable:
                    state.charge_duration = min(max_charge, t - state.stable_start)
                else:
                    state.stable_start = t
                    state.charge_duration = 0.0

                state.ready = state.charge_duration >= charge_time
                state.anchor = anchor
                state.last_seen = t

            else:
                state.charging = False
                state.ready = False
                state.charge_duration = 0.0
                state.stable_start = 0.0

            state.previous_anchor = anchor
            state.previous_size = size
            state.previous_depth = depth

        return events

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
        events.extend(self._update_spiral_qi(hands))
        released_spiral_labels = {event.label for event in events if event.key == "spiral_qi_sphere"}

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
            if label in released_spiral_labels:
                continue
            cooldowns = self._ensure_hand(label)
            spell_key: Optional[str] = None
            if detect_circle_motion(self.motion_history[label], min_radius=hand_size(hand.landmarks) * self.thresholds.circle_radius_scale):
                spell_key = "wind_blade"
            elif detect_finger_gun(hand.landmarks, self.thresholds):
                spell_key = "lightning_shot"
            elif detect_fist(hand.landmarks, self.thresholds):
                spell_key = "fire_punch"
            elif detect_open_palm(hand.landmarks, self.thresholds):
                spiral_state = self.spiral_states.get(label)
                if not two_hands_open and not (spiral_state and spiral_state.charging):
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
