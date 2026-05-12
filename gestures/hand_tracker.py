from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency
    mp = None
import numpy as np

from gestures.smoothing import LandmarkSmoother


@dataclass
class HandData:
    landmarks: List[tuple[int, int, float]]
    label: str
    score: float
    wrist: tuple[int, int]
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        smoothing_alpha: float = 0.6,
    ) -> None:
        self.available = mp is not None and hasattr(mp, "solutions")
        if self.available:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.drawer = mp.solutions.drawing_utils
        else:
            self.mp_hands = None
            self.hands = None
            self.drawer = None
        self.last_results = None
        self.smoother = LandmarkSmoother(alpha=smoothing_alpha)

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        if not self.available:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        self.last_results = results
        hands: List[HandData] = []
        if not results.multi_hand_landmarks:
            return hands

        height, width = frame_bgr.shape[:2]
        active_labels: List[str] = []
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            points: List[tuple[int, int, float]] = []
            xs: List[int] = []
            ys: List[int] = []
            for lm in landmarks.landmark:
                px = int(lm.x * width)
                py = int(lm.y * height)
                points.append((px, py, lm.z))
                xs.append(px)
                ys.append(py)

            if not points:
                continue

            label = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_info = results.multi_handedness[idx].classification[0]
                label = hand_info.label
                score = hand_info.score

            active_labels.append(label)
            points = self.smoother.smooth(label, points)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            wrist = points[0]
            center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            bbox = (min(xs), min(ys), max(xs), max(ys))
            hands.append(
                HandData(
                    landmarks=points,
                    label=label,
                    score=score,
                    wrist=(wrist[0], wrist[1]),
                    center=center,
                    bbox=bbox,
                )
            )

        for label in ["Left", "Right"]:
            if label not in active_labels:
                self.smoother.reset(label)

        return hands

    def draw_landmarks(self, frame_bgr: np.ndarray, hands: List[HandData]) -> None:
        if not self.available or not hands or not self.last_results:
            return
        if not self.last_results.multi_hand_landmarks:
            return
        for landmarks in self.last_results.multi_hand_landmarks:
            self.drawer.draw_landmarks(
                frame_bgr,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )
