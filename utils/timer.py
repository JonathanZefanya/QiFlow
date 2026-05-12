import time


def now() -> float:
    return time.perf_counter()


class Cooldown:
    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.last_trigger = -1e9

    def ready(self, t: float | None = None) -> bool:
        current = t if t is not None else now()
        return (current - self.last_trigger) >= self.duration

    def trigger(self, t: float | None = None) -> None:
        self.last_trigger = t if t is not None else now()
