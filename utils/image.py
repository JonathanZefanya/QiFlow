from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def add_alpha_overlay(
    frame: np.ndarray,
    overlay_bgra: np.ndarray,
    position: Tuple[int, int],
    opacity: float = 1.0,
) -> None:
    x, y = position
    h, w = overlay_bgra.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    if x >= frame_w or y >= frame_h:
        return
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    if w <= 0 or h <= 0:
        return

    overlay = overlay_bgra[:h, :w]
    alpha = overlay[:, :, 3:4] / 255.0
    alpha *= opacity
    base = frame[y : y + h, x : x + w].astype(np.float32)
    over = overlay[:, :, :3].astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * over
    frame[y : y + h, x : x + w] = blended.astype(np.uint8)
