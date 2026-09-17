import math
import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from .counter import LineCounter
from .tracker import TrackVisualizer

CLASS_FILTERS = {
    "all": None,
    "vehicle": [1, 2, 3, 5, 7],      # bicycle, car, motorcycle, bus, truck
    "highway": [2, 3, 5, 7],          # car, motorcycle, bus, truck (kendaraan tol kecepatan tinggi)
    "person": [0],                    # person
    "drone_air": [4, 14, 15, 16],     # airplane, bird (aerial targets)
    "sentry": None,                   # perimeter monitor
}


class DetectionResult:
    def __init__(
        self,
        box: Tuple[int, int, int, int],
        track_id: Optional[int],
        class_id: int,
        class_name: str,
        confidence: float,
        velocity: Tuple[float, float] = (0.0, 0.0),
        speed: float = 0.0,
        speed_kmh: float = 0.0,
        speed_px_s: float = 0.0,
        bearing: float = 0.0,
        is_occluded: bool = False,
    ):
        self.box = box  # (x1, y1, x2, y2)
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.velocity = velocity  # (dx, dy)
        self.speed = speed        # magnitude in px/frame
        self.speed_kmh = speed_kmh  # Speed in km/h
        self.speed_px_s = speed_px_s  # Speed in px/s
        self.bearing = bearing    # heading angle in degrees (0-360)
        self.is_occluded = is_occluded

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))


class YOLOTrackerEngine:
    """
    SpectraTrack AI - Autonomous Object Detection, Multi-Target Tracking, 
    Speed Estimation & Auto-Follow Zoom.
    """

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        enable_trails: bool = True,
        enable_counter: bool = False,
        enable_pip_zoom: bool = True,
        zoom_factor: float = 3.2,
        pixels_per_meter: float = 12.0,
        tactical_mode: str = "all",
        sensor_mode: str = "eo",
        imgsz: int = 480,
    ):
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_trails = enable_trails
        self.enable_counter = enable_counter
        self.enable_pip_zoom = enable_pip_zoom
        self.zoom_factor = zoom_factor
        self.pixels_per_meter = pixels_per_meter
        self.tactical_mode = tactical_mode
        self.sensor_mode = sensor_mode
        self.imgsz = imgsz

        self.locked_target_id: Optional[int] = None
        self.is_user_locked: bool = False
        self.smooth_pip_center: Optional[Tuple[float, float]] = None
        self.prev_pip_target_id: Optional[int] = None
        self.last_frame_shape: Tuple[int, int] = (640, 480)
        self.latest_detections: List[DetectionResult] = []

        self.model = YOLO(model_name)
        
        # Load robust ByteTrack configuration for occlusion handling & lag resistance
        robust_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bytetrack_robust.yaml")
        if os.path.exists(robust_cfg):
            self.tracker_type = os.path.abspath(robust_cfg)
        else:
            self.tracker_type = tracker_type

        # Track Memory & Jitter Elimination Filter
        self.track_memory: Dict[int, Dict] = {}
        self.max_coast_frames: int = 24       # Keep track alive for ~0.8-1.0s during occlusion/lag
        self.box_smooth_alpha: float = 0.72   # EMA smoothing factor: 0.72 (silky smooth, 0 jitter)

        self.visualizer = TrackVisualizer(max_trail_len=35)
        self.counter = LineCounter(start_point=(50, 300), end_point=(590, 300), line_color=(0, 220, 255))
        self.prev_positions: Dict[int, Tuple[int, int]] = {}
        self.last_track_time: Dict[int, float] = {}
        
        # Smoothed speed history per track ID (stores recent speed values for rolling average)
        self.speed_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=8))
        self.cached_detections: List[DetectionResult] = []
        self.cached_class_counts: Dict[str, int] = {}
        self.cached_active_ids: List[int] = []

        self.prev_time = time.time()
        self.current_fps = 0.0
        self.frame_number = 0

    def select_target_by_coord(self, norm_x: float, norm_y: float) -> Tuple[Optional[int], Optional[str]]:
        """
        Locks onto the target containing (norm_x, norm_y) or nearest to it.
        Returns (track_id, class_name) or (None, None).
        """
        fw, fh = self.last_frame_shape
        px = norm_x * fw
        py = norm_y * fh

        best_det = None
        min_dist = float("inf")

        for det in self.latest_detections:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.box
            # If clicked inside bounding box: immediate match
            if x1 <= px <= x2 and y1 <= py <= y2:
                self.locked_target_id = det.track_id
                self.is_user_locked = True
                self.smooth_pip_center = None
                return det.track_id, det.class_name

            # Check distance to center
            cx, cy = det.center
            dist = (cx - px) ** 2 + (cy - py) ** 2
            if dist < min_dist:
                min_dist = dist
                best_det = det

        # Fallback: if within reasonable click range (200 px radius)
        if best_det is not None and min_dist < (200 ** 2):
            self.locked_target_id = best_det.track_id
            self.is_user_locked = True
            self.smooth_pip_center = None
            return best_det.track_id, best_det.class_name

        return None, None

    def release_target(self):
        """Unlocks target and returns to automatic moving target tracking."""
        self.locked_target_id = None
        self.is_user_locked = False
        self.smooth_pip_center = None

    def set_line_coords(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.counter.set_line(start, end)

    def apply_sensor_simulation(self, frame: np.ndarray) -> np.ndarray:
        if self.sensor_mode == "flir":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            thermal = clahe.apply(gray)
            return cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)

        elif self.sensor_mode == "nvg":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            nvg = np.zeros_like(frame)
            nvg[:, :, 1] = enhanced
            nvg[:, :, 0] = (enhanced * 0.15).astype(np.uint8)
            nvg[:, :, 2] = (enhanced * 0.15).astype(np.uint8)
            return nvg

        return frame

    def process_frame(
        self,
        raw_frame: np.ndarray,
        draw_annotations: bool = True,
        skip_inference: bool = False,
    ) -> Tuple[np.ndarray, List[DetectionResult], Dict]:
        self.frame_number += 1
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt) if self.current_fps > 0 else 1.0 / dt
        self.prev_time = now

        effective_fps = self.current_fps if self.current_fps > 5.0 else 25.0

        frame = self.apply_sensor_simulation(raw_frame)
        clean_sensor_frame = frame.copy()
        annotated_frame = frame.copy() if draw_annotations else frame

        detections: List[DetectionResult] = []
        centers: Dict[int, Tuple[int, int]] = {}
        active_ids: List[int] = []
        class_counts: Dict[str, int] = {}
        locked_det: Optional[DetectionResult] = None
        fastest_det: Optional[DetectionResult] = None
        max_speed = 0.0

        fh, fw = frame.shape[:2]
        self.last_frame_shape = (fw, fh)

        if skip_inference and self.cached_detections:
            # During skipped inference frames (interleaved mode), smoothly extrapolate positions
            detections = []
            for d in self.cached_detections:
                vx, vy = d.velocity
                nx1 = int(round(d.box[0] + vx))
                ny1 = int(round(d.box[1] + vy))
                nx2 = int(round(d.box[2] + vx))
                ny2 = int(round(d.box[3] + vy))
                nx1 = max(0, min(fw - 20, nx1))
                ny1 = max(0, min(fh - 20, ny1))
                nx2 = max(nx1 + 10, min(fw - 1, nx2))
                ny2 = max(ny1 + 10, min(fh - 1, ny2))

                new_d = DetectionResult(
                    box=(nx1, ny1, nx2, ny2),
                    track_id=d.track_id,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    velocity=d.velocity,
                    speed=d.speed,
                    speed_kmh=d.speed_kmh,
                    speed_px_s=d.speed_px_s,
                    bearing=d.bearing,
                    is_occluded=d.is_occluded,
                )
                detections.append(new_d)
                if new_d.track_id is not None:
                    centers[new_d.track_id] = new_d.center
                    active_ids.append(new_d.track_id)
                    self.prev_positions[new_d.track_id] = new_d.center
                    self.last_track_time[new_d.track_id] = now
                if new_d.speed_kmh > max_speed:
                    max_speed = new_d.speed_kmh
                    fastest_det = new_d

            class_counts = self.cached_class_counts
            self.cached_detections = detections
        else:
            allowed_classes = CLASS_FILTERS.get(self.tactical_mode, None)

            # Pass lower threshold to model.track so ByteTrack stage-2 can recover occluded objects
            track_conf = max(0.10, min(0.18, self.conf_threshold * 0.6))

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                conf=track_conf,
                iou=self.iou_threshold,
                classes=allowed_classes,
                imgsz=self.imgsz,
                verbose=False,
            )

            seen_track_ids = set()

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                names = results[0].names

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    raw_x1, raw_y1, raw_x2, raw_y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = names.get(cls_id, f"cls_{cls_id}").upper()
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                    # If not tracked yet, ignore weak ghost detections below conf_threshold
                    if track_id is None and conf < self.conf_threshold:
                        continue

                    # Jitter & Pulsing Elimination: Center-Dimension Decomposed Stabilization
                    raw_w = float(raw_x2 - raw_x1)
                    raw_h = float(raw_y2 - raw_y1)
                    raw_cx = float(raw_x1 + raw_x2) / 2.0
                    raw_cy = float(raw_y1 + raw_y2) / 2.0

                    if track_id is not None and track_id in self.track_memory:
                        prev_mem = self.track_memory[track_id]
                        prev_cx, prev_cy = prev_mem["center"]
                        prev_w = float(prev_mem["box"][2] - prev_mem["box"][0])
                        prev_h = float(prev_mem["box"][3] - prev_mem["box"][1])
                        prev_speed = prev_mem.get("speed_kmh", 0.0)

                        # Velocity-Adaptive Smoothing:
                        # High-speed highway vehicles (>50 km/h) require rapid response (alpha -> 0.94-0.96)
                        # to eliminate bounding box lag. Slow/stationary objects use higher damping (0.72) to eliminate jitter.
                        speed_factor = min(1.0, max(0.0, prev_speed / 80.0))
                        alpha_c = 0.72 + 0.24 * speed_factor   # 0.72 at 0 km/h -> 0.96 at 80+ km/h
                        alpha_dim = 0.20 + 0.35 * speed_factor

                        smooth_cx = alpha_c * raw_cx + (1.0 - alpha_c) * prev_cx
                        smooth_cy = alpha_c * raw_cy + (1.0 - alpha_c) * prev_cy

                        smooth_w = alpha_dim * raw_w + (1.0 - alpha_dim) * prev_w
                        smooth_h = alpha_dim * raw_h + (1.0 - alpha_dim) * prev_h

                        x1 = int(round(smooth_cx - smooth_w / 2.0))
                        y1 = int(round(smooth_cy - smooth_h / 2.0))
                        x2 = int(round(smooth_cx + smooth_w / 2.0))
                        y2 = int(round(smooth_cy + smooth_h / 2.0))
                    else:
                        x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2

                    x1 = max(0, min(fw - 20, x1))
                    y1 = max(0, min(fh - 20, y1))
                    x2 = max(x1 + 10, min(fw - 1, x2))
                    y2 = max(y1 + 10, min(fh - 1, y2))
                    curr_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    dx, dy = 0.0, 0.0
                    speed_px = 0.0
                    speed_kmh = 0.0
                    speed_px_s = 0.0
                    bearing = 0.0

                    if track_id is not None:
                        seen_track_ids.add(track_id)
                        if track_id in self.prev_positions:
                            px, py = self.prev_positions[track_id]
                            dx = float(curr_center[0] - px)
                            dy = float(curr_center[1] - py)
                            displacement = math.hypot(dx, dy)
                            speed_px = displacement

                            # Exact dt calculation: Prevents speed doubling/spiking under variable FPS
                            dt_track = (now - self.last_track_time[track_id]) if track_id in self.last_track_time else (1.0 / effective_fps)
                            if dt_track < 0.005 or dt_track > 2.0:
                                dt_track = 1.0 / effective_fps

                            # Speed calculation for moving and walking objects
                            if displacement > 0.15:
                                raw_px_s = displacement / dt_track
                                self.speed_history[track_id].append(raw_px_s)
                                smooth_px_s = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])
                                speed_px_s = smooth_px_s
                                ppm = max(1.0, self.pixels_per_meter)
                                speed_kmh = (smooth_px_s / ppm) * 3.6
                            else:
                                self.speed_history[track_id].append(0.0)
                                speed_kmh = 0.0
                                speed_px_s = 0.0

                            rad = math.atan2(dx, -dy)
                            bearing = (math.degrees(rad) + 360.0) % 360.0

                        self.prev_positions[track_id] = curr_center
                        self.last_track_time[track_id] = now
                        active_ids.append(track_id)
                        centers[track_id] = curr_center

                        # Update Track Memory with smoothed velocity
                        prev_vel = self.track_memory.get(track_id, {}).get("velocity", (dx, dy))
                        smooth_vx = 0.65 * dx + 0.35 * prev_vel[0]
                        smooth_vy = 0.65 * dy + 0.35 * prev_vel[1]

                        self.track_memory[track_id] = {
                            "box": (x1, y1, x2, y2),
                            "center": curr_center,
                            "velocity": (smooth_vx, smooth_vy),
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "speed_px": speed_px,
                            "speed_kmh": speed_kmh,
                            "speed_px_s": speed_px_s,
                            "bearing": bearing,
                            "missed_frames": 0,
                        }

                    det = DetectionResult(
                        box=(x1, y1, x2, y2),
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        velocity=(dx, dy),
                        speed=speed_px,
                        speed_kmh=speed_kmh,
                        speed_px_s=speed_px_s,
                        bearing=bearing,
                        is_occluded=False,
                    )
                    detections.append(det)
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    if speed_kmh > max_speed:
                        max_speed = speed_kmh
                        fastest_det = det

            # OCCLUSION / LAG RESISTANCE: Keep occluded tracks alive in memory
            stale_track_ids = []
            for tid, mem in list(self.track_memory.items()):
                if tid not in seen_track_ids:
                    mem["missed_frames"] += 1
                    if mem["missed_frames"] <= self.max_coast_frames:
                        # Extrapolate position using velocity vector with decay
                        vx, vy = mem["velocity"]
                        mem["velocity"] = (vx * 0.94, vy * 0.94)
                        b = mem["box"]
                        nx1 = int(round(b[0] + mem["velocity"][0]))
                        ny1 = int(round(b[1] + mem["velocity"][1]))
                        nx2 = int(round(b[2] + mem["velocity"][0]))
                        ny2 = int(round(b[3] + mem["velocity"][1]))

                        nx1 = max(0, min(fw - 20, nx1))
                        ny1 = max(0, min(fh - 20, ny1))
                        nx2 = max(nx1 + 10, min(fw - 1, nx2))
                        ny2 = max(ny1 + 10, min(fh - 1, ny2))

                        mem["box"] = (nx1, ny1, nx2, ny2)
                        mem["center"] = (int((nx1 + nx2) / 2), int((ny1 + ny2) / 2))

                        # Re-insert as occluded detection (keeps box alive and protects target lock)
                        coasted_det = DetectionResult(
                            box=(nx1, ny1, nx2, ny2),
                            track_id=tid,
                            class_id=mem["class_id"],
                            class_name=mem["class_name"],
                            confidence=max(0.20, mem["confidence"] * 0.9),
                            velocity=mem["velocity"],
                            speed=mem["speed_px"],
                            speed_kmh=mem["speed_kmh"],
                            speed_px_s=mem["speed_px_s"],
                            bearing=mem["bearing"],
                            is_occluded=True,
                        )
                        detections.append(coasted_det)
                        active_ids.append(tid)
                        centers[tid] = coasted_det.center
                        class_counts[mem["class_name"]] = class_counts.get(mem["class_name"], 0) + 1
                    else:
                        stale_track_ids.append(tid)

            for tid in stale_track_ids:
                self.track_memory.pop(tid, None)
                self.prev_positions.pop(tid, None)
                self.last_track_time.pop(tid, None)
                self.speed_history.pop(tid, None)

            self.cached_detections = detections
            self.cached_class_counts = class_counts
            self.cached_active_ids = active_ids

        # Store for user coordinate click selection
        fh, fw = frame.shape[:2]
        self.last_frame_shape = (fw, fh)
        self.latest_detections = detections

        # Target Selection Logic (Hybrid: Auto-follows moving target by default, locks to user choice on demand)
        is_manual_lock = False
        if self.locked_target_id is not None and self.is_user_locked:
            # User explicitly locked onto a target
            for det in detections:
                if det.track_id == self.locked_target_id:
                    locked_det = det
                    is_manual_lock = True
                    break
        else:
            # Automatic mode: follow the moving object (e.g. moving vehicle or walking person)
            moving_dets = [d for d in detections if d.speed_kmh > 0.8 or d.speed_px_s > 2.0]
            if moving_dets:
                locked_det = max(moving_dets, key=lambda d: d.speed_kmh)
            elif fastest_det is not None:
                locked_det = fastest_det
            elif len(detections) > 0:
                locked_det = detections[0]
            else:
                locked_det = None

        self.visualizer.update(active_ids, centers)

        if self.enable_counter:
            self.counter.update(centers)

        if draw_annotations:
            # 1. Motion trails
            if self.enable_trails:
                self.visualizer.draw_trails(annotated_frame, thickness=2)

            # 2. Virtual tripwire line (if enabled)
            if self.enable_counter:
                self.counter.draw(annotated_frame)

            # 3. Draw bounding boxes with real-time SPEED badge
            for det in detections:
                if is_manual_lock:
                    is_primary = (det.track_id is not None and det.track_id == self.locked_target_id)
                else:
                    is_primary = (locked_det is not None and det.track_id == locked_det.track_id)
                self._draw_annotated_box(annotated_frame, det, is_primary)

            # 4. Auto-Follow Zoom PiP Window for moving / selected target
            if self.enable_pip_zoom and locked_det is not None:
                self._draw_pip_zoom(annotated_frame, clean_sensor_frame, locked_det)

            # 5. Top HUD bar
            self._draw_overlay_hud(annotated_frame, len(active_ids), locked_det)

        # Active targets list for manual dropdown selection
        active_targets = [
            {
                "id": det.track_id,
                "class": det.class_name,
                "speed_kmh": round(det.speed_kmh, 1),
                "is_locked": (det.track_id == (self.locked_target_id if is_manual_lock else (locked_det.track_id if locked_det else None))),
            }
            for det in detections if det.track_id is not None
        ]

        effective_id = self.locked_target_id if is_manual_lock else (locked_det.track_id if locked_det else None)

        stats = {
            "fps": round(self.current_fps, 1),
            "acquisition_rate": f"{int(min(60, self.current_fps))} Hz",
            "active_objects": len(active_ids),
            "tactical_mode": self.tactical_mode,
            "sensor_mode": self.sensor_mode,
            "pip_enabled": self.enable_pip_zoom,
            "zoom_factor": self.zoom_factor,
            "pixels_per_meter": self.pixels_per_meter,
            "classes": class_counts,
            "locked_target_id": effective_id,
            "is_manual_lock": is_manual_lock,
            "active_targets": active_targets,
            "locked_target": {
                "id": effective_id,
                "class": locked_det.class_name if locked_det else "--",
                "conf": f"{int(locked_det.confidence * 100)}%" if locked_det else "--",
                "speed_kmh": f"{locked_det.speed_kmh:.1f} km/h" if locked_det else "0.0 km/h",
                "speed_ms": f"{(locked_det.speed_kmh / 3.6):.1f} m/s" if locked_det else "0.0 m/s",
                "speed_px": f"{locked_det.speed_px_s:.0f} px/s" if locked_det else "0 px/s",
                "bearing": f"{int(locked_det.bearing)}°" if locked_det else "--",
                "coords": f"X:{locked_det.center[0]} Y:{locked_det.center[1]}" if locked_det else "--",
                "is_manual": is_manual_lock,
            } if locked_det else None,
        }

        return annotated_frame, detections, stats

    def _draw_annotated_box(self, frame: np.ndarray, det: DetectionResult, is_primary: bool):
        x1, y1, x2, y2 = det.box
        w, h = x2 - x1, y2 - y1
        cx, cy = det.center

        # Color coding by speed & primary status & occlusion
        if is_primary:
            color = (0, 0, 255)  # Red for user-selected target
        elif det.is_occluded:
            color = (0, 165, 255)  # Amber for occluded / coasting track
        elif det.speed_kmh > 80:
            color = (0, 69, 255)  # Orange-Red for fast moving
        elif det.speed_kmh > 40:
            color = (0, 215, 255)  # Amber
        else:
            color = (255, 200, 50)  # Cyan

        thickness = 2
        if det.is_occluded and not is_primary:
            # Subtle visual indication that track is in memory hold
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            c_len = max(4, min(12, w // 4, h // 4))
            cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, 2)
            cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Tactical corner brackets & reticle if user selected this target
        if is_primary:
            b_len = max(6, min(16, w // 4, h // 4))
            # Top-left
            cv2.line(frame, (x1 - 3, y1 - 3), (x1 + b_len, y1 - 3), (0, 255, 255), 2)
            cv2.line(frame, (x1 - 3, y1 - 3), (x1 - 3, y1 + b_len), (0, 255, 255), 2)
            # Top-right
            cv2.line(frame, (x2 + 3, y1 - 3), (x2 - b_len, y1 - 3), (0, 255, 255), 2)
            cv2.line(frame, (x2 + 3, y1 - 3), (x2 + 3, y1 + b_len), (0, 255, 255), 2)
            # Bottom-left
            cv2.line(frame, (x1 - 3, y2 + 3), (x1 + b_len, y2 + 3), (0, 255, 255), 2)
            cv2.line(frame, (x1 - 3, y2 + 3), (x1 - 3, y2 - b_len), (0, 255, 255), 2)
            # Bottom-right
            cv2.line(frame, (x2 + 3, y2 + 3), (x2 - b_len, y2 + 3), (0, 255, 255), 2)
            cv2.line(frame, (x2 + 3, y2 + 3), (x2 + 3, y2 - b_len), (0, 255, 255), 2)

            # Center target reticle
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 1, cv2.LINE_AA)

        # SPEED & ID BADGE
        speed_text = f"{det.speed_kmh:.0f} km/h"
        hold_tag = " [HOLD]" if det.is_occluded else ""
        if is_primary:
            badge_text = f"TARGET #{det.track_id}{hold_tag} {det.class_name} | {speed_text}"
        elif det.is_occluded:
            badge_text = f"#{det.track_id} [HOLD] | {speed_text}"
        else:
            badge_text = f"#{det.track_id} | {speed_text}" if det.track_id is not None else speed_text

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        b_top = max(0, y1 - th - 8)
        b_bottom = max(th + 8, y1)

        # Draw filled background pill for high contrast
        cv2.rectangle(frame, (x1, b_top), (x1 + tw + 10, b_bottom), (10, 15, 22), -1)
        cv2.rectangle(frame, (x1, b_top), (x1 + tw + 10, b_bottom), color, 1)
        cv2.putText(frame, badge_text, (x1 + 5, b_bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Velocity direction arrow
        if det.speed_px_s > 15:
            vx = int(cx + det.velocity[0] * 3.5)
            vy = int(cy + det.velocity[1] * 3.5)
            cv2.arrowedLine(frame, (cx, cy), (vx, vy), (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)

    def _draw_pip_zoom(self, canvas: np.ndarray, clean_source: np.ndarray, target: DetectionResult):
        """
        Renders Picture-in-Picture Auto-Follow Zoom with adaptive framing, 
        smooth target camera panning, and high-precision tactical reticles.
        """
        fh, fw = canvas.shape[:2]
        cx, cy = target.center
        tx1, ty1, tx2, ty2 = target.box
        target_w = max(18, tx2 - tx1)
        target_h = max(18, ty2 - ty1)

        # 1. Camera Panning: Instant reset on target change, smooth lead tracking on movement
        if self.prev_pip_target_id != target.track_id or self.smooth_pip_center is None:
            self.smooth_pip_center = (float(cx), float(cy))
            self.prev_pip_target_id = target.track_id
        else:
            scx, scy = self.smooth_pip_center
            # Velocity-adaptive camera panning: leads further and pans faster for fast highway vehicles
            speed_factor = min(1.0, max(0.0, target.speed_kmh / 80.0))
            lead_mult = 1.6 + 1.8 * speed_factor   # Predicts ahead based on speed
            cam_alpha = 0.24 + 0.36 * speed_factor # Increases camera tracking speed up to 0.60 for fast cars

            lead_x = float(cx) + target.velocity[0] * lead_mult
            lead_y = float(cy) + target.velocity[1] * lead_mult
            self.smooth_pip_center = (
                (1.0 - cam_alpha) * scx + cam_alpha * lead_x,
                (1.0 - cam_alpha) * scy + cam_alpha * lead_y,
            )

        tcx, tcy = int(self.smooth_pip_center[0]), int(self.smooth_pip_center[1])

        # 2. Smart Optical Framing: Automatically scales crop window based on vehicle size
        # Ensures large buses/trucks and small motorbikes are both perfectly framed with margin
        crop_w = max(int(target_w * 2.0), int(fw / (self.zoom_factor * 1.8)))
        crop_h = max(int(target_h * 2.0), int(fh / (self.zoom_factor * 1.8)))
        crop_half_w = max(28, crop_w // 2)
        crop_half_h = max(24, crop_h // 2)

        x1_crop = max(0, tcx - crop_half_w)
        y1_crop = max(0, tcy - crop_half_h)
        x2_crop = min(fw, tcx + crop_half_w)
        y2_crop = min(fh, tcy + crop_half_h)

        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            return

        cropped = clean_source[y1_crop:y2_crop, x1_crop:x2_crop]
        if cropped.size == 0:
            return

        pip_w = int(fw * 0.38)
        pip_h = int(fh * 0.33)

        try:
            zoomed = cv2.resize(cropped, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
        except Exception:
            zoomed = cv2.resize(cropped, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

        margin = 12
        top_y = 44
        left_x = fw - pip_w - margin
        bottom_y = top_y + pip_h
        right_x = left_x + pip_w

        # Render zoomed feed onto canvas
        canvas[top_y:bottom_y, left_x:right_x] = zoomed

        # 3. Draw High-Precision Target Bounding Box INSIDE the Zoom Window!
        crop_span_x = max(1, x2_crop - x1_crop)
        crop_span_y = max(1, y2_crop - y1_crop)
        scale_x = pip_w / crop_span_x
        scale_y = pip_h / crop_span_y

        z_bx1 = left_x + int((tx1 - x1_crop) * scale_x)
        z_by1 = top_y + int((ty1 - y1_crop) * scale_y)
        z_bx2 = left_x + int((tx2 - x1_crop) * scale_x)
        z_by2 = top_y + int((ty2 - y1_crop) * scale_y)

        # Clamping within pip viewport
        z_bx1 = max(left_x + 1, min(right_x - 6, z_bx1))
        z_by1 = max(top_y + 22, min(bottom_y - 24, z_by1))
        z_bx2 = max(z_bx1 + 6, min(right_x - 2, z_bx2))
        z_by2 = max(z_by1 + 6, min(bottom_y - 22, z_by2))

        # Distinct tactical colors inside lens
        if target.is_occluded:
            lens_color = (0, 165, 255)  # Amber
        elif self.is_user_locked:
            lens_color = (0, 255, 120)  # High-vis Green
        else:
            lens_color = (0, 240, 255)  # Cyan

        # Draw vehicle framing box inside lens
        cv2.rectangle(canvas, (z_bx1, z_by1), (z_bx2, z_by2), lens_color, 1, cv2.LINE_AA)
        zw, zh = z_bx2 - z_bx1, z_by2 - z_by1
        zc_len = max(4, min(14, zw // 3, zh // 3))
        # Lens Corner brackets
        cv2.line(canvas, (z_bx1, z_by1), (z_bx1 + zc_len, z_by1), lens_color, 2)
        cv2.line(canvas, (z_bx1, z_by1), (z_bx1, z_by1 + zc_len), lens_color, 2)
        cv2.line(canvas, (z_bx2, z_by1), (z_bx2 - zc_len, z_by1), lens_color, 2)
        cv2.line(canvas, (z_bx2, z_by1), (z_bx2, z_by1 + zc_len), lens_color, 2)
        cv2.line(canvas, (z_bx1, z_by2), (z_bx1 + zc_len, z_by2), lens_color, 2)
        cv2.line(canvas, (z_bx1, z_by2), (z_bx1, z_by2 - zc_len), lens_color, 2)
        cv2.line(canvas, (z_bx2, z_by2), (z_bx2 - zc_len, z_by2), lens_color, 2)
        cv2.line(canvas, (z_bx2, z_by2), (z_bx2, z_by2 - zc_len), lens_color, 2)

        # Precision target center crosshair inside box
        z_cx = (z_bx1 + z_bx2) // 2
        z_cy = (z_by1 + z_by2) // 2
        cv2.circle(canvas, (z_cx, z_cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (z_cx, z_cy), 8, lens_color, 1, cv2.LINE_AA)

        # 4. Tactical Zoom Window Frame & HUD
        cv2.rectangle(canvas, (left_x, top_y), (right_x, bottom_y), (10, 15, 22), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (left_x - 1, top_y - 1), (right_x + 1, bottom_y + 1), (0, 240, 255), 1, cv2.LINE_AA)

        # Top Header Pill
        status_tag = "HOLD" if target.is_occluded else ("LOCKED" if self.is_user_locked else "AUTO")
        tag_str = f"TARGET ZOOM [{status_tag}] #{target.track_id} {target.class_name}"
        cv2.rectangle(canvas, (left_x, top_y), (right_x, top_y + 20), (10, 15, 22), -1)
        cv2.putText(canvas, tag_str, (left_x + 8, top_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        # Pulse indicator dot
        dot_color = (0, 165, 255) if target.is_occluded else (0, 255, 120)
        cv2.circle(canvas, (right_x - 12, top_y + 10), 4, dot_color, -1, cv2.LINE_AA)

        # Bottom Speed Banner inside Zoom Window
        speed_banner = f"KECEPATAN: {target.speed_kmh:.1f} KM/H  ({target.speed_kmh / 3.6:.1f} m/s)"
        cv2.rectangle(canvas, (left_x, bottom_y - 20), (right_x, bottom_y), (10, 15, 22), -1)
        cv2.putText(canvas, speed_banner, (left_x + 8, bottom_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 255, 120), 1, cv2.LINE_AA)

    def _draw_overlay_hud(self, frame: np.ndarray, track_count: int, locked: Optional[DetectionResult]):
        fh, fw = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, 36), (14, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, 36), (fw, 36), (6, 182, 212), 1, cv2.LINE_AA)

        mode_str = self.tactical_mode.upper()
        cv2.putText(frame, f"SPECTRATRACK AI | {mode_str}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.current_fps:.1f} FPS", (260, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (6, 182, 212), 1, cv2.LINE_AA)

        if locked and locked.track_id is not None:
            if self.is_user_locked:
                lock_str = f"TARGET TERKUNCI (USER): #{locked.track_id} ({locked.class_name}) | {locked.speed_kmh:.1f} KM/H"
                cv2.putText(frame, lock_str, (fw - 410, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
            else:
                lock_str = f"MENGIKUTI OBJEK BERGERAK: #{locked.track_id} ({locked.class_name}) | {locked.speed_kmh:.1f} KM/H"
                cv2.putText(frame, lock_str, (fw - 430, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)
        elif self.locked_target_id is not None:
            lock_str = f"MENCARI TARGET #{self.locked_target_id}..."
            cv2.putText(frame, lock_str, (fw - 300, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 1, cv2.LINE_AA)
        else:
            hint_str = "MEMINDAI OBJEK BERGERAK..."
            cv2.putText(frame, hint_str, (fw - 280, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (6, 182, 212), 1, cv2.LINE_AA)
