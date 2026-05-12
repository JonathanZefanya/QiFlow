from __future__ import annotations

from collections import deque
from typing import Deque, Iterable


class LogBuffer:
    def __init__(self, max_lines: int = 200) -> None:
        self._lines: Deque[str] = deque(maxlen=max_lines)

    def add(self, message: str) -> None:
        self._lines.append(message)

    def get(self) -> Iterable[str]:
        return list(self._lines)
