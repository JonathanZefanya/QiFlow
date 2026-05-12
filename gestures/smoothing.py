from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = alpha
        self._state: Dict[str, np.ndarray] = {}

    def smooth(self, label: str, landmarks: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        points = np.array(landmarks, dtype=np.float32)
        if label not in self._state:
            self._state[label] = points
            return landmarks
        prev = self._state[label]
        smoothed = (self.alpha * prev) + ((1.0 - self.alpha) * points)
        self._state[label] = smoothed
        return [(int(p[0]), int(p[1]), float(p[2])) for p in smoothed]

    def reset(self, label: str) -> None:
        if label in self._state:
            del self._state[label]
