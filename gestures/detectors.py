from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import math
import numpy as np


WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18
THUMB_IP = 3
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17


Point = Tuple[int, int, float]


@dataclass
class GestureThresholds:
    finger_extend_ratio: float = 1.1
    thumb_extend_ratio: float = 1.1
    fist_relaxed_ratio: float = 0.85
    circle_radius_scale: float = 0.35


def thresholds_from_sensitivity(sensitivity: dict) -> GestureThresholds:
    open_palm = float(sensitivity.get("open_palm", 0.6))
    fist = float(sensitivity.get("fist", 0.65))
    circle = float(sensitivity.get("circle_motion", 0.35))
    hand_scale = float(sensitivity.get("hand_scale", 1.0))
    finger_ratio = 1.05 + (1.0 - open_palm) * 0.25
    thumb_ratio = 1.05 + (1.0 - open_palm) * 0.25
    fist_ratio = 0.75 + (1.0 - fist) * 0.25
    return GestureThresholds(
        finger_extend_ratio=finger_ratio,
        thumb_extend_ratio=thumb_ratio,
        fist_relaxed_ratio=fist_ratio,
        circle_radius_scale=max(0.2, min(0.6, circle)) * max(0.6, min(1.6, hand_scale)),
    )


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finger_extended(landmarks: Sequence[Point], tip: int, pip: int, ratio: float) -> bool:
    wrist = landmarks[WRIST]
    return _distance(landmarks[tip], wrist) > (_distance(landmarks[pip], wrist) * ratio)


def _thumb_extended(landmarks: Sequence[Point], ratio: float) -> bool:
    wrist = landmarks[WRIST]
    return _distance(landmarks[THUMB_TIP], wrist) > (_distance(landmarks[THUMB_IP], wrist) * ratio)


def hand_size(landmarks: Sequence[Point]) -> float:
    return max(1.0, _distance(landmarks[WRIST], landmarks[MIDDLE_MCP]))


def detect_fist(landmarks: Sequence[Point], thresholds: GestureThresholds | None = None) -> bool:
    cfg = thresholds or GestureThresholds()
    fingers = [
        _finger_extended(landmarks, INDEX_TIP, INDEX_PIP, cfg.finger_extend_ratio),
        _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, cfg.finger_extend_ratio),
        _finger_extended(landmarks, RING_TIP, RING_PIP, cfg.finger_extend_ratio),
        _finger_extended(landmarks, PINKY_TIP, PINKY_PIP, cfg.finger_extend_ratio),
    ]
    thumb = _thumb_extended(landmarks, cfg.thumb_extend_ratio)
    wrist = landmarks[WRIST]
    close_enough = (
        _distance(landmarks[INDEX_TIP], wrist) < hand_size(landmarks) * cfg.fist_relaxed_ratio
    )
    return not any(fingers) and not thumb and close_enough


def detect_open_palm(landmarks: Sequence[Point], thresholds: GestureThresholds | None = None) -> bool:
    cfg = thresholds or GestureThresholds()
    fingers = [
        _finger_extended(landmarks, INDEX_TIP, INDEX_PIP, cfg.finger_extend_ratio),
        _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, cfg.finger_extend_ratio),
        _finger_extended(landmarks, RING_TIP, RING_PIP, cfg.finger_extend_ratio),
        _finger_extended(landmarks, PINKY_TIP, PINKY_PIP, cfg.finger_extend_ratio),
    ]
    thumb = _thumb_extended(landmarks, cfg.thumb_extend_ratio)
    return all(fingers) and thumb


def palm_center(landmarks: Sequence[Point]) -> Tuple[int, int]:
    points = [landmarks[WRIST], landmarks[INDEX_MCP], landmarks[MIDDLE_MCP], landmarks[PINKY_MCP]]
    return (
        int(sum(p[0] for p in points) / len(points)),
        int(sum(p[1] for p in points) / len(points)),
    )


def palm_depth(landmarks: Sequence[Point]) -> float:
    points = [landmarks[WRIST], landmarks[INDEX_MCP], landmarks[MIDDLE_MCP], landmarks[PINKY_MCP]]
    return sum(p[2] for p in points) / len(points)


def detect_spiral_qi_activation(landmarks: Sequence[Point], thresholds: GestureThresholds | None = None) -> bool:
    if len(landmarks) <= PINKY_MCP:
        return False
    if not detect_open_palm(landmarks, thresholds):
        return False

    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    palm_width = _distance(index_mcp, pinky_mcp)
    palm_height = _distance(wrist, middle_mcp)
    if palm_width < hand_size(landmarks) * 0.45:
        return False

    # A palm facing the camera tends to show a broad MCP line relative to wrist-to-middle depth.
    # This keeps side-facing open hands from accidentally charging the sphere.
    return palm_width >= palm_height * 0.45


def detect_finger_gun(landmarks: Sequence[Point], thresholds: GestureThresholds | None = None) -> bool:
    cfg = thresholds or GestureThresholds()
    index = _finger_extended(landmarks, INDEX_TIP, INDEX_PIP, cfg.finger_extend_ratio)
    middle = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, cfg.finger_extend_ratio)
    ring = _finger_extended(landmarks, RING_TIP, RING_PIP, cfg.finger_extend_ratio)
    pinky = _finger_extended(landmarks, PINKY_TIP, PINKY_PIP, cfg.finger_extend_ratio)
    thumb = _thumb_extended(landmarks, cfg.thumb_extend_ratio)
    return index and thumb and not middle and not ring and not pinky


def detect_two_hands(hand_count: int) -> bool:
    return hand_count >= 2


def detect_rotation(landmarks: Sequence[Point]) -> float:
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    dx = index_mcp[0] - wrist[0]
    dy = index_mcp[1] - wrist[1]
    return math.degrees(math.atan2(dy, dx))


def detect_palm_direction(landmarks: Sequence[Point]) -> Tuple[float, float]:
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_MCP]
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    length = math.hypot(dx, dy) or 1.0
    return (dx / length, dy / length)


def finger_distance(landmarks: Sequence[Point], a: int, b: int) -> float:
    return _distance(landmarks[a], landmarks[b])


def detect_circle_motion(
    points: Iterable[Tuple[int, int]],
    min_points: int = 12,
    min_radius: float = 15.0,
) -> bool:
    pts = list(points)
    if len(pts) < min_points:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    radii = [math.hypot(x - cx, y - cy) for x, y in pts]
    radius = sum(radii) / len(radii)
    if radius < min_radius:
        return False
    angles = [math.degrees(math.atan2(y - cy, x - cx)) for x, y in pts]
    angles = np.unwrap(np.radians(angles))
    coverage = abs(angles[-1] - angles[0])
    return coverage > math.radians(1.7)
