from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from effects.overlays import draw_spell_text
from effects.particle import ParticleSystem


@dataclass
class Projectile:
    position: np.ndarray
    velocity: np.ndarray
    life: float
    target: Tuple[int, int]


@dataclass
class SlashEffect:
    start: Tuple[int, int]
    end: Tuple[int, int]
    life: float
    width: int


@dataclass
class WaveEffect:
    center: Tuple[int, int]
    direction: Tuple[float, float]
    color: Tuple[int, int, int]
    life: float
    max_life: float


@dataclass
class HandEffectState:
    particles: ParticleSystem
    trails: Deque[Tuple[int, int]]
    spell_timers: Dict[str, float]
    projectiles: Deque[Projectile]
    slashes: Deque[SlashEffect]
    waves: Deque[WaveEffect]
    flash_intensity: float
    hit_timer: float
    active_spell: Optional[str]
    smoothed_center: Optional[Tuple[int, int]]
    smoothed_landmarks: Optional[List[Tuple[int, int, float]]]
    last_center: Optional[Tuple[int, int]]
    hold_time: float
    emit_accumulator: float
    aura_alpha: float


class SpellEffects:
    def __init__(
        self,
        screen_flash: float = 0.5,
        particle_count: int = 120,
        effect_intensity: float = 1.0,
        glow_strength: float = 0.9,
        animation_speed: float = 1.0,
        screen_flash_enabled: bool = True,
        dual_hand_effect_enabled: bool = True,
    ) -> None:
        self.time = time.time()
        self.screen_flash = screen_flash
        self.screen_flash_enabled = screen_flash_enabled
        self.particle_count = particle_count
        self.effect_intensity = max(0.25, effect_intensity)
        self.glow_strength = max(0.0, glow_strength)
        self.animation_speed = max(0.1, animation_speed)
        self.dual_hand_effect_enabled = dual_hand_effect_enabled
        self.states: Dict[str, HandEffectState] = {}
        self.text_timer = 0.0
        self.active_spell: Optional[str] = None

        self.colors: Dict[str, Tuple[int, int, int]] = {
            "charge_energy": (255, 230, 80),
            "fire_punch": (40, 95, 255),
            "energy_shield": (255, 210, 80),
            "wind_blade": (190, 255, 170),
            "lightning_shot": (255, 120, 230),
            "storm_breaker": (255, 150, 255),
            "phoenix_chain": (60, 170, 255),
        }

    def _ensure_state(self, label: str) -> HandEffectState:
        if label not in self.states:
            self.states[label] = HandEffectState(
                particles=ParticleSystem(),
                trails=deque(maxlen=28),
                spell_timers={},
                projectiles=deque(maxlen=28),
                slashes=deque(maxlen=18),
                waves=deque(maxlen=16),
                flash_intensity=0.0,
                hit_timer=0.0,
                active_spell=None,
                smoothed_center=None,
                smoothed_landmarks=None,
                last_center=None,
                hold_time=0.0,
                emit_accumulator=0.0,
                aura_alpha=0.0,
            )
        return self.states[label]

    def trigger_spell(self, label: str, spell_key: str, position: Tuple[int, int], boost: bool = False) -> None:
        state = self._ensure_state(label)
        state.active_spell = spell_key
        state.flash_intensity = self.screen_flash if self.screen_flash_enabled else 0.0
        state.hit_timer = 0.38
        state.spell_timers[spell_key] = 1.35 + (0.45 if boost else 0.0)
        self.active_spell = spell_key
        self.text_timer = 1.2

        color = self.colors.get(spell_key, (220, 220, 220))
        count = int((self.particle_count * 0.55 + (70 if boost else 0)) * self.effect_intensity)
        state.particles.emit(position, count, color, speed=(1.6, 5.0), life=(0.35, 0.95), size=(2.0, 7.0), additive=True)

        direction = self._direction_from_state(state) or (0.0, -1.0)
        state.waves.append(WaveEffect(position, direction, color, 0.55, 0.55))

    def update(self, dt: float, hands) -> None:
        self.time = time.time() * self.animation_speed
        self.text_timer = max(0.0, self.text_timer - dt)

        active = set()
        for hand in hands:
            active.add(hand.label)
            state = self._ensure_state(hand.label)
            state.last_center = state.smoothed_center
            state.smoothed_center = self._smooth_point(state.smoothed_center, hand.center, 0.32)
            state.smoothed_landmarks = self._smooth_landmarks(state.smoothed_landmarks, hand.landmarks, 0.42)
            state.trails.append(state.smoothed_center)
            state.aura_alpha = min(1.0, state.aura_alpha + dt * 5.0)
            state.hold_time += dt if state.active_spell else 0.0

        for label, state in list(self.states.items()):
            if label not in active:
                state.aura_alpha = max(0.0, state.aura_alpha - dt * 2.6)
                if state.aura_alpha <= 0.0:
                    state.trails.clear()
                    state.smoothed_center = None
                    state.smoothed_landmarks = None
                    state.last_center = None
                state.hold_time = 0.0

            if state.active_spell and state.active_spell not in state.spell_timers:
                state.active_spell = None
                state.hold_time = 0.0

            for key in list(state.spell_timers.keys()):
                state.spell_timers[key] = max(0.0, state.spell_timers[key] - dt)
                if state.spell_timers[key] <= 0:
                    del state.spell_timers[key]

            state.flash_intensity = max(0.0, state.flash_intensity - dt * 1.9)
            state.hit_timer = max(0.0, state.hit_timer - dt * 1.8)
            self._update_motion_effects(state, dt)
            state.particles.update(dt)

    def _update_motion_effects(self, state: HandEffectState, dt: float) -> None:
        alive_projectiles: Deque[Projectile] = deque(maxlen=28)
        for proj in list(state.projectiles):
            proj.life -= dt
            if proj.life <= 0:
                continue
            proj.position += proj.velocity * (60.0 * dt)
            alive_projectiles.append(proj)
        state.projectiles = alive_projectiles

        alive_slashes: Deque[SlashEffect] = deque(maxlen=18)
        for slash in list(state.slashes):
            slash.life -= dt
            if slash.life > 0:
                alive_slashes.append(slash)
        state.slashes = alive_slashes

        alive_waves: Deque[WaveEffect] = deque(maxlen=16)
        for wave in list(state.waves):
            wave.life -= dt
            if wave.life > 0:
                alive_waves.append(wave)
        state.waves = alive_waves

    def render(self, frame: np.ndarray, hands, cooldown_provider: Callable[[str, str], float]) -> None:
        if not hands and not any(state.aura_alpha > 0.01 for state in self.states.values()):
            return

        self._apply_screen_shake(frame)

        effect_layer = np.zeros_like(frame)
        glow_layer = np.zeros_like(frame)
        hands_by_label = {hand.label: hand for hand in hands}

        for label, state in list(self.states.items()):
            if state.aura_alpha <= 0.01:
                continue
            hand = hands_by_label.get(label)
            anchor = state.smoothed_center or (hand.center if hand else None)
            if anchor is None:
                continue
            self._render_hand_aura(effect_layer, glow_layer, state, label)
            self._render_trail(effect_layer, glow_layer, state)
            if hand is not None:
                self._render_hand_spell(effect_layer, glow_layer, hand, state, cooldown_provider)

        if self.dual_hand_effect_enabled and len(hands) >= 2:
            self._render_dual_hand(effect_layer, glow_layer, hands)

        self._blend_layers(frame, effect_layer, glow_layer)

        for state in self.states.values():
            state.particles.render(frame)

        for state in self.states.values():
            if state.flash_intensity > 0:
                flash = np.full_like(frame, 255)
                alpha = min(0.75, state.flash_intensity)
                cv2.addWeighted(flash, alpha, frame, 1.0 - alpha, 0, frame)

        if self.text_timer > 0 and self.active_spell:
            text = self.active_spell.replace("_", " ").title()
            pulse = 0.9 + (math.sin(self.text_timer * 9.0) * 0.08)
            draw_spell_text(frame, text, (30, 60), (230, 230, 230), alpha=0.9, scale=pulse)

    def _render_hand_spell(self, layer: np.ndarray, glow: np.ndarray, hand, state: HandEffectState, cooldown_provider) -> None:
        if state.active_spell is None:
            return

        anchor = state.smoothed_center or hand.center
        spell = state.active_spell
        color = self.colors.get(spell, (220, 220, 220))
        timer = state.spell_timers.get(spell, 0.0)
        pulse = 0.5 + 0.5 * math.sin(self.time * 5.0)
        charge = min(1.0, state.hold_time / 1.4)

        self._draw_magic_circle(layer, glow, anchor, int(72 + pulse * 8), color, self.time * 70.0, alpha=0.65)
        self._draw_pulse_ring(layer, glow, anchor, color, timer, 54, 128)
        self._draw_orbiting_particles(layer, glow, anchor, color, 52, 8)
        self._draw_cooldown_ring(layer, anchor, 90, cooldown_provider(hand.label, spell), color)

        if spell == "charge_energy":
            self._render_charge(layer, glow, state, anchor, charge)
        elif spell == "fire_punch":
            self._render_fire(layer, glow, state, anchor)
        elif spell == "energy_shield":
            self._render_shield(layer, glow, [hand], color)
        elif spell == "wind_blade":
            self._render_wind(layer, glow, state, anchor, [hand])
        elif spell == "lightning_shot":
            self._render_lightning(layer, glow, state, anchor, [hand])
        else:
            self._emit_spell_particles(state, anchor, color, 8)

        self._render_projectiles(layer, glow, state)
        self._render_slashes(layer, glow, state)
        self._render_waves(layer, glow, state)

        if state.hit_timer > 0:
            radius = int(34 + (1.0 - state.hit_timer / 0.38) * 70)
            cv2.circle(layer, anchor, radius, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(glow, anchor, radius + 8, color, 4, cv2.LINE_AA)

    def _render_hand_aura(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState, label: str) -> None:
        if not state.smoothed_landmarks or state.smoothed_center is None:
            return

        points = np.array([(int(x), int(y)) for x, y, _z in state.smoothed_landmarks], dtype=np.int32)
        if len(points) < 3:
            return

        hull = cv2.convexHull(points)
        alpha = max(0.0, min(1.0, state.aura_alpha))
        pulse = 0.5 + 0.5 * math.sin(self.time * 4.8 + (0.7 if label == "Left" else 0.0))
        color = (255, 225, 110) if label == "Right" else (255, 200, 135)
        center = state.smoothed_center

        expanded_outer = self._expand_polygon(hull, center, 1.18 + pulse * 0.05)
        expanded_inner = self._expand_polygon(hull, center, 1.06 + pulse * 0.03)
        outer_color = tuple(int(c * alpha * 0.9) for c in color)
        inner_color = tuple(int(c * alpha) for c in (255, 245, 190))

        cv2.fillConvexPoly(glow, expanded_outer, tuple(int(c * 0.42) for c in outer_color), cv2.LINE_AA)
        cv2.polylines(glow, [expanded_outer], True, outer_color, 10, cv2.LINE_AA)
        cv2.polylines(glow, [expanded_inner], True, inner_color, 5, cv2.LINE_AA)

        cv2.fillConvexPoly(layer, expanded_inner, tuple(int(c * 0.18) for c in color), cv2.LINE_AA)
        cv2.polylines(layer, [expanded_inner], True, inner_color, 2, cv2.LINE_AA)
        cv2.polylines(layer, [hull], True, tuple(int(c * alpha) for c in (255, 255, 230)), 1, cv2.LINE_AA)

        for idx in (0, 4, 8, 12, 16, 20):
            if idx >= len(points):
                continue
            p = tuple(int(v) for v in points[idx])
            radius = int((4 + pulse * 3) * alpha)
            cv2.circle(glow, p, max(2, radius + 4), outer_color, -1, cv2.LINE_AA)
            cv2.circle(layer, p, max(1, radius), inner_color, -1, cv2.LINE_AA)

        self._draw_hand_orbit(layer, glow, state, color, alpha)
        self._emit_hand_aura_particles(state, points, color, alpha)

    def _draw_hand_orbit(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState, color: Tuple[int, int, int], alpha: float) -> None:
        if state.smoothed_center is None or not state.smoothed_landmarks:
            return
        center = state.smoothed_center
        xs = [p[0] for p in state.smoothed_landmarks]
        ys = [p[1] for p in state.smoothed_landmarks]
        radius_x = max(32, int((max(xs) - min(xs)) * 0.58))
        radius_y = max(38, int((max(ys) - min(ys)) * 0.62))
        for i in range(10):
            angle = self.time * (1.4 + i * 0.025) + i * math.tau / 10.0
            wobble = 1.0 + math.sin(self.time * 2.0 + i) * 0.08
            p = (
                int(center[0] + math.cos(angle) * radius_x * wobble),
                int(center[1] + math.sin(angle) * radius_y * wobble),
            )
            orbit_color = tuple(int(c * alpha) for c in color)
            cv2.circle(glow, p, 5, orbit_color, -1, cv2.LINE_AA)
            cv2.circle(layer, p, 2, (255, 255, 240), -1, cv2.LINE_AA)

    def _emit_hand_aura_particles(self, state: HandEffectState, points: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
        if alpha < 0.2 or len(points) == 0:
            return
        state.emit_accumulator += 1.0
        if state.emit_accumulator < 2.0:
            return
        state.emit_accumulator = 0.0
        idx = random.randrange(len(points))
        origin = tuple(int(v) for v in points[idx])
        state.particles.emit(
            origin,
            max(1, int(2 * self.effect_intensity)),
            color,
            speed=(0.15, 0.75),
            life=(0.45, 0.9),
            size=(1.0, 2.8),
            additive=True,
            direction=(0.0, -1.0),
            cone=math.tau,
        )

    def _render_charge(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState, anchor: Tuple[int, int], charge: float) -> None:
        color = self.colors["charge_energy"]
        for idx, radius in enumerate((44, 64, 88, 112)):
            alpha_radius = int(radius + math.sin(self.time * (2.5 + idx) + idx) * (6 + idx * 2) + charge * 18)
            thickness = 2 + idx
            cv2.circle(glow, anchor, alpha_radius, color, thickness, cv2.LINE_AA)
            cv2.circle(layer, anchor, max(8, alpha_radius - 12), (255, 255, 180), 1, cv2.LINE_AA)
        self._draw_spiral(layer, glow, anchor, color, charge)
        self._emit_spell_particles(state, anchor, color, int(10 + charge * 18), speed=(0.25, 1.4), life=(0.7, 1.5), size=(2.0, 5.0))

    def _render_fire(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState, anchor: Tuple[int, int]) -> None:
        direction = self._direction_from_state(state) or (0.0, -1.0)
        warm = random.choice([(30, 80, 255), (20, 40, 220), (70, 180, 255)])
        self._emit_spell_particles(state, anchor, warm, 22, speed=(1.0, 4.2), life=(0.25, 0.75), size=(3.0, 8.0), gravity=-0.035, direction=(-direction[0], -direction[1]), cone=1.8)
        for i in range(5):
            angle = math.atan2(direction[1], direction[0]) + random.uniform(-1.1, 1.1)
            end = (int(anchor[0] - math.cos(angle) * (42 + i * 8)), int(anchor[1] - math.sin(angle) * (42 + i * 8)))
            cv2.line(glow, anchor, end, warm, 5, cv2.LINE_AA)
            cv2.line(layer, anchor, end, (80, 220, 255), 2, cv2.LINE_AA)

    def _render_shield(self, layer: np.ndarray, glow: np.ndarray, hands, color: Tuple[int, int, int]) -> None:
        center = hands[0].center if hands else (layer.shape[1] // 2, layer.shape[0] // 2)
        self._draw_hex_barrier(layer, glow, center, 96, color, self.time * 34.0)
        self._draw_magic_circle(layer, glow, center, 80, color, -self.time * 56.0, alpha=0.55)

    def _render_wind(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState, anchor: Tuple[int, int], hands) -> None:
        direction = self._hand_direction(hands) or self._direction_from_state(state) or (1.0, 0.0)
        angle = math.atan2(direction[1], direction[0]) + math.sin(self.time * 5.0) * 0.35
        for offset in (-0.34, 0.0, 0.34):
            length = 155 + int(35 * math.sin(self.time * 4.0 + offset))
            end = (int(anchor[0] + math.cos(angle + offset) * length), int(anchor[1] + math.sin(angle + offset) * length))
            state.slashes.append(SlashEffect(anchor, end, 0.34, 4))
        self._emit_spell_particles(state, anchor, self.colors["wind_blade"], 16, speed=(1.2, 3.6), life=(0.35, 0.85), size=(1.5, 4.0), direction=direction, cone=1.0)

    def _render_lightning(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState, anchor: Tuple[int, int], hands) -> None:
        direction = self._hand_direction(hands) or self._direction_from_state(state) or (1.0, 0.0)
        target = (int(anchor[0] + direction[0] * 230), int(anchor[1] + direction[1] * 230))
        for _ in range(3):
            self._draw_lightning_arc(layer, glow, anchor, target, self.colors["lightning_shot"], jitter=22)
        state.projectiles.append(
            Projectile(
                np.array([anchor[0], anchor[1]], dtype=np.float32),
                np.array([direction[0] * 9.0, direction[1] * 9.0], dtype=np.float32),
                0.34,
                target,
            )
        )
        self._emit_spell_particles(state, anchor, self.colors["lightning_shot"], 20, speed=(1.8, 5.0), life=(0.2, 0.55), size=(1.5, 4.5), direction=direction, cone=1.4)

    def _render_dual_hand(self, layer: np.ndarray, glow: np.ndarray, hands) -> None:
        left_state = self._ensure_state(hands[0].label)
        right_state = self._ensure_state(hands[1].label)
        left = left_state.smoothed_center or hands[0].center
        right = right_state.smoothed_center or hands[1].center
        mid = (int((left[0] + right[0]) / 2), int((left[1] + right[1]) / 2))
        dist = int(math.hypot(left[0] - right[0], left[1] - right[1]))
        radius = max(100, int(dist * 0.62))
        color = self.colors["energy_shield"]

        self._draw_energy_beam(layer, glow, left, right, color)
        self._draw_hex_barrier(layer, glow, mid, radius, color, self.time * 28.0)
        self._draw_magic_circle(layer, glow, mid, int(radius * 0.72), color, -self.time * 42.0, alpha=0.65)
        self._draw_pulse_ring(layer, glow, mid, color, 1.0 + math.sin(self.time * 2.0), int(radius * 0.55), int(radius * 1.1))

    def _render_trail(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState) -> None:
        points = list(state.trails)
        if len(points) < 2:
            return
        color = self.colors.get(state.active_spell or "charge_energy", (180, 220, 255))
        for idx in range(1, len(points)):
            alpha = idx / len(points)
            p1, p2 = points[idx - 1], points[idx]
            thickness = max(2, int(10 * alpha))
            cv2.line(glow, p1, p2, color, thickness + 5, cv2.LINE_AA)
            cv2.line(layer, p1, p2, color, thickness, cv2.LINE_AA)

    def _render_base_hand(self, layer: np.ndarray, glow: np.ndarray, center: Tuple[int, int], label: str) -> None:
        color = (255, 150, 220) if label == "Left" else (255, 225, 120)
        pulse = 0.5 + 0.5 * math.sin(self.time * 4.0)
        for radius, thickness in ((34, 2), (48, 2), (66, 1)):
            r = int(radius + pulse * 5)
            cv2.circle(glow, center, r, color, thickness + 2, cv2.LINE_AA)
            cv2.circle(layer, center, r, color, thickness, cv2.LINE_AA)

    def _render_projectiles(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState) -> None:
        for proj in state.projectiles:
            start = (int(proj.position[0]), int(proj.position[1]))
            self._draw_lightning_arc(layer, glow, start, proj.target, self.colors["lightning_shot"], jitter=16)
            cv2.circle(glow, start, 8, (255, 255, 255), -1, cv2.LINE_AA)

    def _render_slashes(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState) -> None:
        for slash in state.slashes:
            alpha = max(0.0, min(1.0, slash.life / 0.34))
            color = self.colors["wind_blade"]
            cv2.line(glow, slash.start, slash.end, color, int(slash.width * 4 * alpha) + 2, cv2.LINE_AA)
            cv2.line(layer, slash.start, slash.end, (235, 255, 230), max(1, int(slash.width * alpha)), cv2.LINE_AA)

    def _render_waves(self, layer: np.ndarray, glow: np.ndarray, state: HandEffectState) -> None:
        for wave in state.waves:
            progress = 1.0 - wave.life / wave.max_life
            radius = int(28 + progress * 160)
            center = (
                int(wave.center[0] + wave.direction[0] * progress * 120),
                int(wave.center[1] + wave.direction[1] * progress * 120),
            )
            alpha_color = tuple(int(c * (1.0 - progress)) for c in wave.color)
            cv2.ellipse(glow, center, (radius, max(12, radius // 3)), math.degrees(math.atan2(wave.direction[1], wave.direction[0])), 0, 360, alpha_color, 4, cv2.LINE_AA)
            cv2.ellipse(layer, center, (radius, max(8, radius // 4)), math.degrees(math.atan2(wave.direction[1], wave.direction[0])), 0, 360, alpha_color, 2, cv2.LINE_AA)

    def _draw_magic_circle(self, layer: np.ndarray, glow: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], angle: float, alpha: float = 0.6) -> None:
        draw_color = tuple(int(c * alpha) for c in color)
        for r in (radius, int(radius * 0.72), int(radius * 0.43)):
            cv2.circle(glow, center, r, draw_color, 3, cv2.LINE_AA)
            cv2.circle(layer, center, r, draw_color, 1, cv2.LINE_AA)
        for i in range(24):
            a = math.radians(angle + i * 15)
            inner = int(radius * (0.78 if i % 2 else 0.62))
            outer = radius + (8 if i % 3 == 0 else 0)
            p1 = (int(center[0] + math.cos(a) * inner), int(center[1] + math.sin(a) * inner))
            p2 = (int(center[0] + math.cos(a) * outer), int(center[1] + math.sin(a) * outer))
            cv2.line(layer, p1, p2, draw_color, 1, cv2.LINE_AA)
        for i in range(12):
            a = math.radians(-angle * 1.4 + i * 30)
            p = (int(center[0] + math.cos(a) * radius * 0.92), int(center[1] + math.sin(a) * radius * 0.92))
            self._draw_rune(layer, glow, p, int(5 + (i % 3)), draw_color, a)

    def _draw_rune(self, layer: np.ndarray, glow: np.ndarray, pos: Tuple[int, int], size: int, color: Tuple[int, int, int], angle: float) -> None:
        dx = int(math.cos(angle) * size)
        dy = int(math.sin(angle) * size)
        cv2.line(layer, (pos[0] - dx, pos[1] - dy), (pos[0] + dx, pos[1] + dy), color, 1, cv2.LINE_AA)
        cv2.line(layer, (pos[0] - dy, pos[1] + dx), (pos[0] + dy, pos[1] - dx), color, 1, cv2.LINE_AA)
        cv2.circle(glow, pos, max(2, size // 2), color, 1, cv2.LINE_AA)

    def _draw_pulse_ring(self, layer: np.ndarray, glow: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int], timer: float, min_radius: int, max_radius: int) -> None:
        progress = (self.time * 0.7 + timer) % 1.0
        radius = int(min_radius + (max_radius - min_radius) * progress)
        fade = 1.0 - progress
        draw_color = tuple(int(c * fade) for c in color)
        cv2.circle(glow, center, radius, draw_color, 6, cv2.LINE_AA)
        cv2.circle(layer, center, radius, draw_color, 2, cv2.LINE_AA)

    def _draw_orbiting_particles(self, layer: np.ndarray, glow: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int], radius: int, count: int) -> None:
        for i in range(count):
            a = self.time * (1.2 + i * 0.08) + i * (math.tau / count)
            r = radius + math.sin(self.time * 2.0 + i) * 12
            p = (int(center[0] + math.cos(a) * r), int(center[1] + math.sin(a) * r))
            cv2.circle(glow, p, 6, color, -1, cv2.LINE_AA)
            cv2.circle(layer, p, 2, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_cooldown_ring(self, layer: np.ndarray, center: Tuple[int, int], radius: int, progress: float, color: Tuple[int, int, int]) -> None:
        progress = max(0.0, min(1.0, progress))
        cv2.circle(layer, center, radius, (45, 45, 55), 3, cv2.LINE_AA)
        if progress <= 0:
            return
        end_angle = int(360 * progress)
        cv2.ellipse(layer, center, (radius, radius), -90, 0, end_angle, color, 4, cv2.LINE_AA)

    def _draw_spiral(self, layer: np.ndarray, glow: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int], charge: float) -> None:
        points = []
        for i in range(80):
            t = i / 79.0
            a = self.time * 4.0 + t * math.tau * 3.0
            r = 12 + t * (92 + charge * 28)
            points.append((int(center[0] + math.cos(a) * r), int(center[1] + math.sin(a) * r)))
        for i in range(1, len(points)):
            cv2.line(glow, points[i - 1], points[i], color, 3, cv2.LINE_AA)
            cv2.line(layer, points[i - 1], points[i], (255, 255, 210), 1, cv2.LINE_AA)

    def _draw_hex_barrier(self, layer: np.ndarray, glow: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], angle: float) -> None:
        points = []
        for i in range(6):
            a = math.radians(angle + i * 60)
            points.append((int(center[0] + math.cos(a) * radius), int(center[1] + math.sin(a) * radius)))
        cv2.fillConvexPoly(layer, np.array(points, dtype=np.int32), tuple(int(c * 0.16) for c in color), cv2.LINE_AA)
        for i in range(6):
            cv2.line(glow, points[i], points[(i + 1) % 6], color, 6, cv2.LINE_AA)
            cv2.line(layer, points[i], points[(i + 1) % 6], (255, 255, 230), 2, cv2.LINE_AA)
        for scale in (0.74, 0.48):
            inner = []
            for i in range(6):
                a = math.radians(-angle * 0.8 + i * 60)
                inner.append((int(center[0] + math.cos(a) * radius * scale), int(center[1] + math.sin(a) * radius * scale)))
            for i in range(6):
                cv2.line(layer, inner[i], inner[(i + 1) % 6], color, 1, cv2.LINE_AA)

    def _draw_energy_beam(self, layer: np.ndarray, glow: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        for width in (18, 10, 4):
            cv2.line(glow, start, end, color, width, cv2.LINE_AA)
        for i in range(7):
            t = i / 6.0
            wobble = math.sin(self.time * 8.0 + i) * 10
            x = int(start[0] + (end[0] - start[0]) * t)
            y = int(start[1] + (end[1] - start[1]) * t + wobble)
            cv2.circle(layer, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(layer, start, end, (255, 255, 230), 3, cv2.LINE_AA)

    def _draw_lightning_arc(self, layer: np.ndarray, glow: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], color: Tuple[int, int, int], jitter: int = 14) -> None:
        points = [start]
        segments = 9
        for i in range(1, segments):
            t = i / segments
            x = start[0] + (end[0] - start[0]) * t + random.uniform(-jitter, jitter)
            y = start[1] + (end[1] - start[1]) * t + random.uniform(-jitter, jitter)
            points.append((int(x), int(y)))
        points.append(end)
        for i in range(1, len(points)):
            cv2.line(glow, points[i - 1], points[i], color, 7, cv2.LINE_AA)
            cv2.line(layer, points[i - 1], points[i], (255, 255, 255), 2, cv2.LINE_AA)

    def _emit_spell_particles(
        self,
        state: HandEffectState,
        anchor: Tuple[int, int],
        color: Tuple[int, int, int],
        count: int,
        speed: Tuple[float, float] = (0.6, 2.2),
        life: Tuple[float, float] = (0.4, 1.0),
        size: Tuple[float, float] = (2.0, 4.0),
        gravity: float = 0.0,
        direction: Optional[Tuple[float, float]] = None,
        cone: float = math.tau,
    ) -> None:
        scaled = max(1, int(count * self.effect_intensity))
        state.particles.emit(anchor, scaled, color, speed=speed, life=life, size=size, additive=True, gravity=gravity, direction=direction, cone=cone)

    def _blend_layers(self, frame: np.ndarray, layer: np.ndarray, glow: np.ndarray) -> None:
        if np.any(glow):
            blur_big = cv2.GaussianBlur(glow, (0, 0), 18)
            blur_small = cv2.GaussianBlur(glow, (0, 0), 6)
            frame[:] = cv2.addWeighted(frame, 1.0, blur_big, 0.34 * self.glow_strength, 0)
            frame[:] = cv2.addWeighted(frame, 1.0, blur_small, 0.55 * self.glow_strength, 0)
        if np.any(layer):
            frame[:] = cv2.addWeighted(frame, 1.0, layer, min(1.0, 0.86 * self.effect_intensity), 0)

    def _apply_screen_shake(self, frame: np.ndarray) -> None:
        lightning_power = 0.0
        for state in self.states.values():
            lightning_power = max(lightning_power, state.spell_timers.get("lightning_shot", 0.0))
        if lightning_power <= 0:
            return
        amount = int(min(7, 2 + lightning_power * 5))
        dx = random.randint(-amount, amount)
        dy = random.randint(-amount, amount)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
        frame[:] = shaken

    def _expand_polygon(self, polygon: np.ndarray, center: Tuple[int, int], scale: float) -> np.ndarray:
        pts = polygon.reshape(-1, 2).astype(np.float32)
        c = np.array(center, dtype=np.float32)
        expanded = c + (pts - c) * scale
        return expanded.astype(np.int32).reshape(-1, 1, 2)

    def _smooth_point(self, current: Optional[Tuple[int, int]], target: Tuple[int, int], alpha: float) -> Tuple[int, int]:
        if current is None:
            return target
        return (int(current[0] + (target[0] - current[0]) * alpha), int(current[1] + (target[1] - current[1]) * alpha))

    def _smooth_landmarks(
        self,
        current: Optional[List[Tuple[int, int, float]]],
        target: List[Tuple[int, int, float]],
        alpha: float,
    ) -> List[Tuple[int, int, float]]:
        if current is None or len(current) != len(target):
            return list(target)
        smoothed: List[Tuple[int, int, float]] = []
        for old, new in zip(current, target):
            x = int(old[0] + (new[0] - old[0]) * alpha)
            y = int(old[1] + (new[1] - old[1]) * alpha)
            z = old[2] + (new[2] - old[2]) * alpha
            smoothed.append((x, y, z))
        return smoothed

    def _direction_from_state(self, state: HandEffectState) -> Optional[Tuple[float, float]]:
        if state.last_center is None or state.smoothed_center is None:
            return None
        dx = state.smoothed_center[0] - state.last_center[0]
        dy = state.smoothed_center[1] - state.last_center[1]
        length = math.hypot(dx, dy)
        if length < 2.0:
            return None
        return (dx / length, dy / length)

    def _hand_direction(self, hands) -> Optional[Tuple[float, float]]:
        if not hands:
            return None
        hand = hands[0]
        if not hasattr(hand, "landmarks") or len(hand.landmarks) < 9:
            return None
        wrist = hand.landmarks[0]
        index = hand.landmarks[8]
        dx = index[0] - wrist[0]
        dy = index[1] - wrist[1]
        length = math.hypot(dx, dy) or 1.0
        return (dx / length, dy / length)
