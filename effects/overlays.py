from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def draw_magic_circle(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    alpha: float = 0.6,
) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, 2)
    cv2.circle(overlay, center, int(radius * 0.7), color, 1)
    cv2.circle(overlay, center, int(radius * 0.4), color, 1)
    for angle in range(0, 360, 45):
        x = int(center[0] + radius * 0.7 * np.cos(np.radians(angle)))
        y = int(center[1] + radius * 0.7 * np.sin(np.radians(angle)))
        cv2.line(overlay, center, (x, y), color, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_energy_pulse(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    alpha: float = 0.3,
) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, color, 6)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_spell_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.9,
    scale: float = 1.0,
) -> None:
    overlay = frame.copy()
    cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_cooldown_bar(
    frame: np.ndarray,
    position: Tuple[int, int],
    size: Tuple[int, int],
    progress: float,
    color: Tuple[int, int, int],
) -> None:
    x, y = position
    w, h = size
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * max(0.0, min(1.0, progress)))
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)


def draw_aura(
    frame: np.ndarray,
    center: Tuple[int, int],
    base_radius: int,
    color: Tuple[int, int, int],
    pulse: float,
) -> None:
    overlay = frame.copy()
    radius = int(base_radius + (pulse * 6))
    cv2.circle(overlay, center, radius, color, 2)
    cv2.circle(overlay, center, int(radius * 1.3), color, 1)
    cv2.circle(overlay, center, int(radius * 0.7), color, 1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


def draw_trail(
    frame: np.ndarray,
    points: list[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    if len(points) < 2:
        return
    overlay = frame.copy()
    max_len = len(points)
    for idx in range(1, max_len):
        p1 = points[idx - 1]
        p2 = points[idx]
        alpha = idx / max_len
        thickness = max(1, int(6 * alpha))
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, frame, 0.7, 0, frame)
