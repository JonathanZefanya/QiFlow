from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from utils.timer import Cooldown, now
from gestures.combos import ComboSystem
from gestures.detectors import (
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


@dataclass
class HandGestureRuntime:
    label: str
    state: str = "IDLE"
    gesture: str = "none"
    previous_gesture: str = "none"
    gesture_since: float = 0.0
    open_since: float = 0.0
    finger_gun_since: float = 0.0
    stable_since: float = 0.0
    last_seen: float = 0.0
    last_center: Optional[Tuple[int, int]] = None
    last_depth: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    depth_velocity: float = 0.0
    stability_score: float = 0.0
    active_spell: str = "-"
    charge_progress: float = 0.0


@dataclass
class ShieldRuntime:
    state: str = "IDLE"
    started_at: float = 0.0
    active: bool = False
    center: Tuple[int, int] = (0, 0)
    progress: float = 0.0


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
        self.hand_runtime: Dict[str, HandGestureRuntime] = {
            "Left": HandGestureRuntime("Left"),
            "Right": HandGestureRuntime("Right"),
        }
        self.shield_runtime = ShieldRuntime()
        self.last_update_time = now()
        self.debug_snapshot: Dict[str, object] = {}

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

    def get_debug_snapshot(self) -> Dict[str, object]:
        return self.debug_snapshot

    def get_cooldown_remaining(self, label: str, key: str) -> float:
        cooldowns = self._ensure_hand(label)
        cd = cooldowns.get(key)
        if not cd:
            return 0.0
        return max(0.0, cd.duration - (now() - cd.last_trigger))

    def _classify_gesture(self, hand: HandData) -> str:
        if detect_spiral_qi_activation(hand.landmarks, self.thresholds):
            return "open_palm_front"
        if detect_open_palm(hand.landmarks, self.thresholds):
            return "open_palm"
        if detect_fist(hand.landmarks, self.thresholds):
            return "fist"
        if detect_finger_gun(hand.landmarks, self.thresholds):
            return "finger_gun"
        return "unknown"

    def _update_hand_runtime(self, hand: HandData, t: float, dt: float) -> HandGestureRuntime:
        runtime = self.hand_runtime.setdefault(hand.label, HandGestureRuntime(hand.label))
        center = palm_center(hand.landmarks)
        depth = palm_depth(hand.landmarks)
        gesture = self._classify_gesture(hand)
        previous_gesture = runtime.gesture
        runtime.previous_gesture = previous_gesture
        runtime.gesture = gesture

        if previous_gesture != gesture:
            runtime.gesture_since = t
        if gesture.startswith("open_palm"):
            if not runtime.previous_gesture.startswith("open_palm"):
                runtime.open_since = t
        if gesture == "finger_gun":
            if previous_gesture != "finger_gun":
                runtime.finger_gun_since = t
        else:
            runtime.finger_gun_since = 0.0

        if runtime.last_center is None or dt <= 0:
            vx = vy = 0.0
            depth_velocity = 0.0
        else:
            vx = (center[0] - runtime.last_center[0]) / dt
            vy = (center[1] - runtime.last_center[1]) / dt
            depth_velocity = (depth - runtime.last_depth) / dt

        speed = (vx * vx + vy * vy) ** 0.5
        scale = hand_size(hand.landmarks)
        stable_threshold = max(90.0, scale * 2.4)
        runtime.stability_score = max(0.0, min(1.0, 1.0 - speed / stable_threshold))
        if runtime.stability_score >= 0.62:
            if runtime.stable_since <= 0:
                runtime.stable_since = t
        else:
            runtime.stable_since = 0.0

        runtime.velocity = (vx, vy)
        runtime.depth_velocity = depth_velocity
        runtime.last_center = center
        runtime.last_depth = depth
        runtime.last_seen = t
        return runtime

    def _reset_missing_hands(self, active_labels: set[str]) -> None:
        for label, runtime in self.hand_runtime.items():
            if label in active_labels:
                continue
            if runtime.state in {"PREPARE", "CHARGING", "READY"}:
                runtime.state = "CANCELLED"
            else:
                runtime.state = "IDLE"
            runtime.gesture = "none"
            runtime.active_spell = "-"
            runtime.charge_progress = 0.0
            runtime.open_since = 0.0
            runtime.finger_gun_since = 0.0
            runtime.stable_since = 0.0

    def _trigger_event(self, label: str, spell_key: str, t: float, is_combo: bool = False) -> Optional[HandSpellEvent]:
        cooldowns = self._ensure_hand(label)
        cd = cooldowns[spell_key]
        if not cd.ready(t):
            runtime = self.hand_runtime.setdefault(label, HandGestureRuntime(label))
            runtime.state = "COOLDOWN"
            runtime.active_spell = spell_key
            return None
        cd.trigger(t)
        runtime = self.hand_runtime.setdefault(label, HandGestureRuntime(label))
        runtime.state = "CASTING"
        runtime.active_spell = spell_key
        runtime.charge_progress = 1.0
        combo = self.combo_system.register_spell(spell_key)
        if combo and not is_combo:
            return HandSpellEvent(label, combo.key, combo.name, True)
        return HandSpellEvent(label, spell_key, self.spells[spell_key].name, is_combo)

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
        t = now()
        dt = max(1 / 60, min(0.25, t - self.last_update_time))
        self.last_update_time = t
        self.update_motion(hands)

        events: List[HandSpellEvent] = []
        runtimes: Dict[str, HandGestureRuntime] = {}
        active_labels = {hand.label for hand in hands}
        self._reset_missing_hands(active_labels)
        for hand in hands:
            runtimes[hand.label] = self._update_hand_runtime(hand, t, dt)

        blocked_labels: set[str] = set()

        # 1. Energy Shield: highest priority, intentional dual-hand hold.
        if len(hands) >= 2:
            h1, h2 = hands[0], hands[1]
            r1, r2 = runtimes[h1.label], runtimes[h2.label]
            both_open = r1.gesture.startswith("open_palm") and r2.gesture.startswith("open_palm")
            distance = ((palm_center(h1.landmarks)[0] - palm_center(h2.landmarks)[0]) ** 2 + (palm_center(h1.landmarks)[1] - palm_center(h2.landmarks)[1]) ** 2) ** 0.5
            shield_valid = both_open and distance >= 150 and r1.stability_score >= 0.5 and r2.stability_score >= 0.5
            if shield_valid:
                blocked_labels.update({h1.label, h2.label})
                if self.shield_runtime.state not in {"PREPARE", "READY"}:
                    self.shield_runtime.state = "PREPARE"
                    self.shield_runtime.started_at = t
                elapsed = t - self.shield_runtime.started_at
                self.shield_runtime.progress = min(1.0, elapsed / 1.0)
                self.shield_runtime.center = (
                    int((h1.center[0] + h2.center[0]) / 2),
                    int((h1.center[1] + h2.center[1]) / 2),
                )
                if elapsed >= 1.0:
                    self.shield_runtime.state = "READY"
                    event = self._trigger_event("Both", "energy_shield", t)
                    if event:
                        events.append(event)
                        self.shield_runtime.active = True
            else:
                self.shield_runtime = ShieldRuntime(state="CANCELLED" if self.shield_runtime.state in {"PREPARE", "READY"} else "IDLE")

        # 2. Spiral Qi Sphere: stable palm charge, released only by deliberate push.
        if not blocked_labels and bool(self.spiral_qi_config.get("enabled", True)):
            for hand in hands:
                runtime = runtimes[hand.label]
                state = self.spiral_states.setdefault(hand.label, SpiralQiState(hand.label))
                anchor = palm_center(hand.landmarks)
                size = hand_size(hand.landmarks)
                depth = palm_depth(hand.landmarks)
                speed = (runtime.velocity[0] ** 2 + runtime.velocity[1] ** 2) ** 0.5
                forward_push = False
                if state.previous_anchor is not None:
                    size_delta = size - state.previous_size
                    forward_push = size_delta > max(7.0, size * 0.07) or runtime.depth_velocity < -0.12
                fast_push = speed > float(self.spiral_qi_config.get("release_speed", 25)) * 20.0

                if state.charging and state.ready and (forward_push or fast_push):
                    event = self._trigger_event(hand.label, "spiral_qi_sphere", t)
                    if event:
                        events.append(event)
                    state.charging = False
                    state.ready = False
                    state.released = True
                    blocked_labels.add(hand.label)
                    state.previous_anchor = anchor
                    state.previous_size = size
                    state.previous_depth = depth
                    continue

                if runtime.gesture == "open_palm_front" and runtime.stability_score >= 0.62 and self._ensure_hand(hand.label)["spiral_qi_sphere"].ready(t):
                    if not state.charging:
                        state.charging = True
                        state.ready = False
                        state.start_time = t
                        state.stable_start = t
                    state.charge_duration = min(float(self.spiral_qi_config.get("max_charge_time", 2.0)), t - state.stable_start)
                    state.ready = state.charge_duration >= float(self.spiral_qi_config.get("charge_time", 0.8))
                    state.anchor = anchor
                    runtime.state = "READY" if state.ready else "CHARGING"
                    runtime.active_spell = "spiral_qi_sphere"
                    runtime.charge_progress = min(1.0, state.charge_duration / float(self.spiral_qi_config.get("max_charge_time", 2.0)))
                    blocked_labels.add(hand.label)
                elif state.charging:
                    state.charging = False
                    state.ready = False
                    state.charge_duration = 0.0
                    runtime.state = "CANCELLED"
                    runtime.charge_progress = 0.0

                state.previous_anchor = anchor
                state.previous_size = size
                state.previous_depth = depth

        # 3-7. Single-hand rules by priority.
        for hand in hands:
            if hand.label in blocked_labels:
                continue
            runtime = runtimes[hand.label]
            speed_x = abs(runtime.velocity[0])
            speed_y = abs(runtime.velocity[1])
            scale = hand_size(hand.landmarks)

            # Fire Punch: open palm preparation, then a fist transition.
            if runtime.gesture == "fist" and runtime.previous_gesture.startswith("open_palm") and runtime.open_since > 0 and (t - runtime.open_since) >= 0.5:
                event = self._trigger_event(hand.label, "fire_punch", t)
                if event:
                    events.append(event)
                continue

            # Lightning Shot: finger gun lock for 0.5 seconds.
            if runtime.gesture == "finger_gun":
                lock_time = t - runtime.finger_gun_since if runtime.finger_gun_since > 0 else 0.0
                runtime.state = "READY" if lock_time >= 0.5 else "PREPARE"
                runtime.active_spell = "lightning_shot"
                runtime.charge_progress = min(1.0, lock_time / 0.5)
                if lock_time >= 0.5:
                    event = self._trigger_event(hand.label, "lightning_shot", t)
                    if event:
                        events.append(event)
                continue

            # Wind Blade: strong horizontal swipe only.
            if runtime.gesture.startswith("open_palm") and speed_x > max(650.0, scale * 9.0) and speed_x > speed_y * 1.7:
                event = self._trigger_event(hand.label, "wind_blade", t)
                if event:
                    events.append(event)
                continue

            # Charge Energy: stable open palm facing camera for 1 second.
            if runtime.gesture == "open_palm_front":
                stable_time = t - runtime.stable_since if runtime.stable_since > 0 else 0.0
                runtime.state = "CHARGING" if stable_time < 1.0 else "READY"
                runtime.active_spell = "charge_energy"
                runtime.charge_progress = min(1.0, stable_time / 1.0)
                if stable_time >= 1.0:
                    event = self._trigger_event(hand.label, "charge_energy", t)
                    if event:
                        events.append(event)
                continue

            if runtime.state not in {"COOLDOWN", "CASTING"}:
                runtime.state = "IDLE" if runtime.gesture in {"none", "unknown"} else "PREPARE"
                runtime.active_spell = "-"
                runtime.charge_progress = 0.0

        self._update_debug_snapshot(hands, runtimes)
        return events

    def _update_debug_snapshot(self, hands: List[HandData], runtimes: Dict[str, HandGestureRuntime]) -> None:
        hand_debug: Dict[str, Dict[str, object]] = {}
        for hand in hands:
            runtime = runtimes[hand.label]
            cooldowns = {
                key: round(self.get_cooldown_remaining(hand.label, key), 2)
                for key in self.spells
            }
            hand_debug[hand.label] = {
                "gesture": runtime.gesture,
                "state": runtime.state,
                "active_spell": runtime.active_spell,
                "charge_progress": round(runtime.charge_progress, 2),
                "cooldown_remaining": cooldowns,
                "velocity": (round(runtime.velocity[0], 1), round(runtime.velocity[1], 1)),
                "stability_score": round(runtime.stability_score, 2),
            }

        self.debug_snapshot = {
            "hands": hand_debug,
            "shield": {
                "state": self.shield_runtime.state,
                "progress": round(self.shield_runtime.progress, 2),
                "active": self.shield_runtime.active,
            },
            "priority": "Energy Shield > Spiral Qi Sphere > Fire Punch > Lightning Shot > Wind Blade > Charge Energy > Hand Aura",
        }

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
