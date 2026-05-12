from __future__ import annotations

from typing import List, Optional

import cv2


class CameraManager:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int,
        auto_exposure: bool,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_exposure = auto_exposure
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.release()
        self.capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if self.auto_exposure:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        return True

    def read(self) -> Optional[cv2.Mat]:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def set_index(self, index: int) -> bool:
        self.index = index
        return self.open()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @staticmethod
    def list_cameras(max_index: int = 4) -> List[int]:
        found: List[int] = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
            cap.release()
        return found
