from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk

from audio.sound_manager import SoundManager
from camera.camera_manager import CameraManager
from config.spell_config import load_spell_config, save_spell_config
from effects.spell_effects import SpellEffects
from gestures.detectors import detect_rotation, finger_distance, hand_size, INDEX_TIP, THUMB_TIP
from gestures.hand_tracker import HandTracker
from gestures.spell_system import HandSpellEvent, SpellSystem
from config.settings import load_config
from utils.assets import asset_path, ensure_dir
from utils.fps import FPSCounter
from utils.logging_utils import LogBuffer
from utils.timer import now
from ui.loading import LoadingScreen


class QiFlowApp:
    def __init__(self) -> None:
        self.base_dir = Path(__file__).resolve().parents[1]
        self.config = load_config()
        cv2.setUseOptimized(True)

        self.root = tk.Tk()
        self.root.title("QiFlow")
        self.root.configure(bg="#0b0f1a")
        self.root.geometry("1280x760")
        self.root.minsize(980, 620)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.root.bind("<Key>", self._on_key)

        self.loading = LoadingScreen(self.root)
        self.root.update_idletasks()
        self.root.update()

        self.panel = tk.Frame(self.root, bg="#121729", width=280)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel.pack_propagate(False)

        self.video_label = tk.Label(self.root, bg="#0b0f1a")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.log_buffer = LogBuffer(max_lines=200)
        self.training_label = tk.StringVar(value=self.config["training"]["label"])

        self._build_panel()

        cam_cfg = self.config["camera"]
        self.camera = CameraManager(
            cam_cfg["index"],
            cam_cfg["width"],
            cam_cfg["height"],
            cam_cfg["fps"],
            cam_cfg["auto_exposure"],
        )
        self.camera_ready = self.camera.open()
        if not self.camera_ready:
            self.log("Camera failed to open, using fallback frame")

        mp_cfg = self.config["mediapipe"]
        tracking_cfg = self.config.get("tracking", {})
        self.tracker = HandTracker(
            max_hands=mp_cfg["max_hands"],
            detection_confidence=mp_cfg["detection_confidence"],
            tracking_confidence=mp_cfg["tracking_confidence"],
            smoothing_alpha=tracking_cfg.get("smoothing_alpha", 0.6),
        )
        if not self.tracker.available:
            self.log("MediaPipe solutions missing. Install mediapipe==0.10.14 for hand tracking.")
        spell_cfg = load_spell_config()
        cooldowns = spell_cfg.get("cooldowns", self.config["spells"]["cooldowns"])
        combo_cfg = self.config.get("combo", {})
        combo_rules = []
        if combo_cfg.get("enabled", True):
            combo_rules = spell_cfg.get("combo", {}).get("rules", combo_cfg.get("rules", []))
        self.spell_system = SpellSystem(cooldowns, self.config["sensitivity"], combo_rules)
        effects_cfg = self.config.get("effects", {})
        self.effects = SpellEffects(
            effects_cfg.get("screen_flash", 0.5),
            particle_count=int(effects_cfg.get("particle_count", effects_cfg.get("particles", 120))),
            effect_intensity=float(effects_cfg.get("effect_intensity", 1.0)),
            glow_strength=float(effects_cfg.get("glow_strength", effects_cfg.get("glow_intensity", 0.9))),
            animation_speed=float(effects_cfg.get("animation_speed", 1.0)),
            screen_flash_enabled=bool(effects_cfg.get("screen_flash_enabled", True)),
            dual_hand_effect_enabled=bool(effects_cfg.get("dual_hand_effect_enabled", True)),
        )
        self.sounds = SoundManager(
            enabled=self.config["audio"]["enabled"],
            volume=self.config["audio"]["volume"],
        )
        if not self.sounds.enabled:
            self.log("Sound system disabled (pygame missing or disabled)")
        self.fps = FPSCounter()

        self.last_frame_time = now()
        self.active_spell: Optional[str] = None
        self.active_spell_name: Optional[str] = None
        self.record_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.training_enabled = False
        self.demo_enabled = bool(self.config.get("demo", {}).get("enabled", False))
        self.demo_index = 0
        self.demo_last_time = now()
        self.calibrating = False
        self.calibration_samples: list[float] = []
        self.last_camera_attempt = now()

        ensure_dir(asset_path("recordings"))
        ensure_dir(asset_path("screenshots"))
        ensure_dir(asset_path("training"))

        self.loading.close()

    def _build_panel(self) -> None:
        header = tk.Label(
            self.panel,
            text="QiFlow",
            bg="#121729",
            fg="#cde6ff",
            font=("Segoe UI", 18, "bold"),
        )
        header.pack(pady=(16, 8))

        self.spell_label = tk.Label(
            self.panel,
            text="Active Spell: -",
            bg="#121729",
            fg="#f4d58d",
            font=("Segoe UI", 12),
        )
        self.spell_label.pack(pady=6)

        self.status_label = tk.Label(
            self.panel,
            text="Hands: 0",
            bg="#121729",
            fg="#9dd2ff",
            font=("Segoe UI", 11),
        )
        self.status_label.pack(pady=6)

        self.fps_label = tk.Label(
            self.panel,
            text="FPS: 0",
            bg="#121729",
            fg="#9dd2ff",
            font=("Segoe UI", 11),
        )
        self.fps_label.pack(pady=6)

        self.gesture_label = tk.Label(
            self.panel,
            text="Rotation: - | Pinch: -",
            bg="#121729",
            fg="#9dd2ff",
            font=("Segoe UI", 10),
        )
        self.gesture_label.pack(pady=6)

        camera_label = tk.Label(
            self.panel,
            text="Camera",
            bg="#121729",
            fg="#cde6ff",
            font=("Segoe UI", 10, "bold"),
        )
        camera_label.pack(pady=(14, 4))

        cameras = CameraManager.list_cameras()
        self.camera_combo = ttk.Combobox(
            self.panel,
            values=cameras if cameras else [0],
            state="readonly",
            width=12,
        )
        self.camera_combo.current(0)
        self.camera_combo.bind("<<ComboboxSelected>>", self._change_camera)
        self.camera_combo.pack(pady=4)

        controls_label = tk.Label(
            self.panel,
            text="Controls",
            bg="#121729",
            fg="#cde6ff",
            font=("Segoe UI", 10, "bold"),
        )
        controls_label.pack(pady=(14, 4))

        tk.Button(self.panel, text="Toggle Fullscreen", command=self.toggle_fullscreen).pack(pady=4)
        tk.Button(self.panel, text="Toggle Sound", command=self.toggle_sound).pack(pady=4)
        tk.Button(self.panel, text="Screenshot", command=self.save_screenshot).pack(pady=4)
        tk.Button(self.panel, text="Record Video", command=self.toggle_record).pack(pady=4)
        tk.Button(self.panel, text="Toggle Demo", command=self.toggle_demo).pack(pady=4)
        tk.Button(self.panel, text="Calibrate", command=self.toggle_calibration).pack(pady=4)
        tk.Button(self.panel, text="Save Spell Config", command=self.save_spell_config).pack(pady=4)
        tk.Button(self.panel, text="Load Spell Config", command=self.load_spell_config).pack(pady=4)

        training_label = tk.Label(
            self.panel,
            text="Training Label",
            bg="#121729",
            fg="#cde6ff",
            font=("Segoe UI", 10, "bold"),
        )
        training_label.pack(pady=(14, 4))

        training_entry = tk.Entry(self.panel, textvariable=self.training_label)
        training_entry.pack(pady=4)
        tk.Button(self.panel, text="Toggle Training", command=self.toggle_training).pack(pady=4)

        if self.config.get("ui", {}).get("show_debug", True):
            debug_label = tk.Label(
                self.panel,
                text="Debug Log",
                bg="#121729",
                fg="#cde6ff",
                font=("Segoe UI", 10, "bold"),
            )
            debug_label.pack(pady=(14, 4))
            self.debug_text = tk.Text(
                self.panel,
                height=10,
                bg="#0b0f1a",
                fg="#9dd2ff",
                insertbackground="#9dd2ff",
                font=("Consolas", 9),
                relief=tk.FLAT,
            )
            self.debug_text.pack(padx=8, pady=(0, 8), fill=tk.X)
        else:
            self.debug_text = None

    def _change_camera(self, _event: object) -> None:
        try:
            index = int(self.camera_combo.get())
        except ValueError:
            return
        self.camera.set_index(index)

    def toggle_fullscreen(self) -> None:
        is_full = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not is_full)

    def toggle_sound(self) -> None:
        self.sounds.toggle()
        state = "on" if self.sounds.enabled else "off"
        self.log(f"Sound {state}")

    def toggle_record(self) -> None:
        if self.recording:
            self.recording = False
            if self.record_writer:
                self.record_writer.release()
                self.record_writer = None
            self.log("Recording stopped")
            return

        record_cfg = self.config["recording"]
        codec = cv2.VideoWriter_fourcc(*record_cfg["codec"])
        filename = time.strftime("record_%Y%m%d_%H%M%S.avi")
        path = asset_path("recordings", filename)
        self.record_writer = cv2.VideoWriter(
            str(path),
            codec,
            record_cfg["fps"],
            (self.config["camera"]["width"], self.config["camera"]["height"]),
        )
        self.recording = True
        self.log(f"Recording started: {path.name}")

    def toggle_training(self) -> None:
        self.training_enabled = not self.training_enabled
        state = "on" if self.training_enabled else "off"
        self.log(f"Training {state}")

    def toggle_demo(self) -> None:
        self.demo_enabled = not self.demo_enabled
        state = "on" if self.demo_enabled else "off"
        self.demo_last_time = now()
        self.log(f"Demo mode {state}")

    def toggle_calibration(self) -> None:
        self.calibrating = not self.calibrating
        self.calibration_samples.clear()
        state = "on" if self.calibrating else "off"
        self.log(f"Calibration {state}")

    def save_spell_config(self) -> None:
        data = {
            "cooldowns": {k: v.cooldown.duration for k, v in self.spell_system.spells.items()},
            "combo": {"rules": self.config["combo"]["rules"]},
        }
        path = save_spell_config(data)
        self.log(f"Saved spell config: {path.name}")

    def load_spell_config(self) -> None:
        spell_cfg = load_spell_config()
        cooldowns = spell_cfg.get("cooldowns")
        if cooldowns:
            for key, value in cooldowns.items():
                if key in self.spell_system.spells:
                    self.spell_system.spells[key].cooldown.duration = float(value)
            self.log("Loaded cooldowns from custom config")
        combo_rules = spell_cfg.get("combo", {}).get("rules")
        if combo_rules and self.config.get("combo", {}).get("enabled", True):
            self.config["combo"]["rules"] = combo_rules
            self.spell_system = SpellSystem(
                {k: v.cooldown.duration for k, v in self.spell_system.spells.items()},
                self.config["sensitivity"],
                combo_rules,
            )
            self.log("Loaded combos from custom config")

    def save_screenshot(self) -> None:
        frame = self.camera.read()
        if frame is None:
            return
        filename = time.strftime("shot_%Y%m%d_%H%M%S.png")
        path = asset_path("screenshots", filename)
        cv2.imwrite(str(path), frame)
        self.log(f"Screenshot saved: {path.name}")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_buffer.add(f"[{timestamp}] {message}")
        if self.debug_text is None:
            return
        self.debug_text.delete("1.0", tk.END)
        self.debug_text.insert(tk.END, "\n".join(self.log_buffer.get()))
        self.debug_text.see(tk.END)

    def _on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "r":
            self.toggle_record()
        elif key == "p":
            self.save_screenshot()
        elif key == "m":
            self.toggle_sound()

    def _export_training(self, hands) -> None:
        if not self.training_enabled or not hands:
            return
        label = self.training_label.get().strip()
        if not label:
            return
        rows = []
        for hand in hands:
            row = [label, hand.label]
            for x, y, z in hand.landmarks:
                row.extend([x, y, z])
            rows.append(row)

        filename = time.strftime(f"{label}_%Y%m%d.csv")
        path = asset_path("training", filename)
        with path.open("a", encoding="utf-8") as file:
            for row in rows:
                file.write(",".join(map(str, row)) + "\n")

    def _handle_spell_event(self, event: HandSpellEvent, frame: np.ndarray, hands) -> None:
        self.active_spell = event.key
        self.active_spell_name = event.name
        anchor = (frame.shape[1] // 2, frame.shape[0] // 2)
        if hands:
            if event.label == "Both" and len(hands) >= 2:
                anchor = (
                    int((hands[0].center[0] + hands[1].center[0]) / 2),
                    int((hands[0].center[1] + hands[1].center[1]) / 2),
                )
                self.effects.trigger_spell("Left", event.key, hands[0].center, boost=event.is_combo)
                self.effects.trigger_spell("Right", event.key, hands[1].center, boost=event.is_combo)
            else:
                for hand in hands:
                    if hand.label == event.label:
                        anchor = hand.center
                        break
                self.effects.trigger_spell(event.label, event.key, anchor, boost=event.is_combo)
        else:
            self.effects.trigger_spell(event.label, event.key, anchor, boost=event.is_combo)
        self.sounds.play(event.key)
        self.log(f"Spell: {event.name}")

    def _save_config(self) -> None:
        config_path = asset_path("config", "config.json")
        config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    def _update_frame(self) -> None:
        if not self.camera_ready:
            if now() - self.last_camera_attempt > 1.0:
                self.last_camera_attempt = now()
                self.camera_ready = self.camera.open()
                if self.camera_ready:
                    self.log("Camera reconnected")
            frame = np.zeros((self.config["camera"]["height"], self.config["camera"]["width"], 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        else:
            frame = self.camera.read()
            if frame is None:
                self.camera_ready = False
                self.log("Camera lost, switching to fallback")
                self.root.after(10, self._update_frame)
                return

        hands = self.tracker.process(frame) if self.camera_ready else []

        if self.demo_enabled and not hands:
            demo_cfg = self.config.get("demo", {})
            interval = float(demo_cfg.get("interval", 3.0))
            if now() - self.demo_last_time >= interval:
                cycle = demo_cfg.get("cycle", [])
                if cycle:
                    key = cycle[self.demo_index % len(cycle)]
                    event = self.spell_system.trigger_manual_for("Left", key)
                    if event:
                        self._handle_spell_event(event, frame, hands)
                self.demo_index += 1
                self.demo_last_time = now()

        events = self.spell_system.update_multi(hands)
        for event in events:
            self._handle_spell_event(event, frame, hands)

        if self.config["ui"]["show_landmarks"]:
            self.tracker.draw_landmarks(frame, hands)

        current_time = now()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time

        self.effects.update(dt, hands)
        self.effects.render(frame, hands, self.spell_system.get_cooldown_progress)

        fps = self.fps.update()
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.status_label.config(text=f"Hands: {len(hands)}")
        if self.active_spell_name:
            self.spell_label.config(text=f"Active Spell: {self.active_spell_name}")
        else:
            self.spell_label.config(text="Active Spell: -")

        if hands:
            rotation = detect_rotation(hands[0].landmarks)
            pinch = finger_distance(hands[0].landmarks, INDEX_TIP, THUMB_TIP)
            self.gesture_label.config(text=f"Rotation: {rotation:.0f} | Pinch: {pinch:.0f}")
        else:
            self.gesture_label.config(text="Rotation: - | Pinch: -")

        if self.calibrating and hands:
            size = hand_size(hands[0].landmarks)
            self.calibration_samples.append(size)
            if len(self.calibration_samples) >= int(self.config["calibration"]["samples"]):
                avg = sum(self.calibration_samples) / len(self.calibration_samples)
                self.config["sensitivity"]["hand_scale"] = round(avg / 120.0, 2)
                self.spell_system.update_thresholds(self.config["sensitivity"])
                self._save_config()
                self.calibrating = False
                self.calibration_samples.clear()
                self.log("Calibration complete")

        if self.recording and self.record_writer:
            self.record_writer.write(frame)

        self._export_training(hands)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()
        if w < 10 or h < 10:
            h, w = frame.shape[:2]
        img = img.resize((w, h))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        target_fps = int(self.config.get("tracking", {}).get("target_fps", 30))
        delay = max(1, int(1000 / max(10, target_fps)))
        self.root.after(delay, self._update_frame)

    def run(self) -> None:
        self.root.after(10, self._update_frame)
        self.root.mainloop()

    def _on_close(self) -> None:
        if self.record_writer:
            self.record_writer.release()
        self.camera.release()
        self.root.destroy()
