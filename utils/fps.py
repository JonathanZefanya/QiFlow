import time


class FPSCounter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.last_time = time.perf_counter()
        self.fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0:
            return self.fps
        instant = 1.0 / dt
        if self.fps == 0.0:
            self.fps = instant
        else:
            self.fps = (self.fps * self.smoothing) + (instant * (1.0 - self.smoothing))
        return self.fps
