from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    additive: bool = True
    gravity: float = 0.0
    drag: float = 0.97


class ParticleSystem:
    def __init__(self) -> None:
        self.particles: List[Particle] = []

    def emit(
        self,
        origin: Tuple[int, int],
        count: int,
        color: Tuple[int, int, int],
        speed: Tuple[float, float] = (0.6, 2.2),
        spread: float = 1.0,
        life: Tuple[float, float] = (0.4, 1.0),
        size: Tuple[float, float] = (2.0, 4.0),
        additive: bool = True,
        gravity: float = 0.0,
        direction: Optional[Tuple[float, float]] = None,
        cone: float = 6.283185307179586,
    ) -> None:
        ox, oy = origin
        base_angle = None
        if direction is not None:
            base_angle = float(np.arctan2(direction[1], direction[0]))

        for _ in range(max(0, count)):
            if base_angle is None:
                angle = float(np.random.uniform(0.0, np.pi * 2.0))
            else:
                angle = float(base_angle + np.random.uniform(-cone * 0.5, cone * 0.5))

            force = float(np.random.uniform(speed[0], speed[1]) * spread)
            life_span = float(np.random.uniform(life[0], life[1]))
            self.particles.append(
                Particle(
                    x=float(ox),
                    y=float(oy),
                    vx=float(np.cos(angle) * force),
                    vy=float(np.sin(angle) * force),
                    size=float(np.random.uniform(size[0], size[1])),
                    life=life_span,
                    max_life=life_span,
                    color=color,
                    additive=additive,
                    gravity=gravity,
                )
            )

    def update(self, dt: float) -> None:
        alive: List[Particle] = []
        step = 60.0 * dt
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.vy += p.gravity * step
            p.x += p.vx * step
            p.y += p.vy * step
            p.vx *= p.drag
            p.vy *= p.drag
            alive.append(p)
        self.particles = alive

    def render(self, frame: np.ndarray) -> None:
        if not self.particles:
            return
        overlay = np.zeros_like(frame)
        additive = np.zeros_like(frame)
        for p in self.particles:
            x, y = int(p.x), int(p.y)
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                continue
            alpha = max(0.0, min(1.0, p.life / p.max_life))
            eased = alpha * alpha
            color = (
                int(p.color[0] * eased),
                int(p.color[1] * eased),
                int(p.color[2] * eased),
            )
            radius = max(1, int(p.size * (0.5 + alpha)))
            target = additive if p.additive else overlay
            cv2.circle(target, (x, y), radius, color, -1, cv2.LINE_AA)

        if np.any(overlay):
            cv2.addWeighted(overlay, 0.7, frame, 1.0, 0, frame)
        if np.any(additive):
            blur = cv2.GaussianBlur(additive, (0, 0), 7)
            frame[:] = cv2.add(frame, additive)
            frame[:] = cv2.add(frame, blur)
