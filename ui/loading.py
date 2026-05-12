from __future__ import annotations

import math
import tkinter as tk


class LoadingScreen:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.title("QiFlow Loading")
        self.win.geometry("420x240")
        self.win.configure(bg="#0b0f1a")
        self.win.overrideredirect(True)
        self.canvas = tk.Canvas(self.win, width=420, height=240, bg="#0b0f1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.text = self.canvas.create_text(
            210,
            140,
            text="QiFlow Cultivation Channeling...",
            fill="#cde6ff",
            font=("Segoe UI", 12, "bold"),
        )
        self.angle = 0.0
        self.running = True
        self._tick()

    def _tick(self) -> None:
        if not self.running:
            return
        self.canvas.delete("ring")
        cx, cy = 210, 90
        radius = 40
        for i in range(6):
            angle = self.angle + (i * 60)
            x = cx + radius * math.cos(math.radians(angle))
            y = cy + radius * math.sin(math.radians(angle))
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#7cd4ff", width=0, tags="ring")
        self.angle = (self.angle + 6) % 360
        self.win.after(33, self._tick)

    def close(self) -> None:
        self.running = False
        self.win.destroy()
