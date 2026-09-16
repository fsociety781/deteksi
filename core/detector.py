import math
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
        model_name: str = "yolo11n.pt",
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

        self.locked_target_id: Optional[int] = None
        self.is_user_locked: bool = False
        self.smooth_pip_center: Optional[Tuple[float, float]] = None
        self.last_frame_shape: Tuple[int, int] = (640, 480)
        self.latest_detections: List[DetectionResult] = []

        self.model = YOLO(model_name)
        self.visualizer = TrackVisualizer(max_trail_len=35)
        self.counter = LineCounter(start_point=(50, 300), end_point=(590, 300), line_color=(0, 220, 255))
        self.prev_positions: Dict[int, Tuple[int, int]] = {}
        
        # Smoothed speed history per track ID (stores recent speed values for rolling average)
        self.speed_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=8))

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

        allowed_classes = CLASS_FILTERS.get(self.tactical_mode, None)

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=allowed_classes,
            imgsz=384,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        centers: Dict[int, Tuple[int, int]] = {}
        active_ids: List[int] = []
        class_counts: Dict[str, int] = {}
        locked_det: Optional[DetectionResult] = None
        fastest_det: Optional[DetectionResult] = None
        max_speed = 0.0

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = names.get(cls_id, f"cls_{cls_id}").upper()

                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                curr_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                dx, dy = 0.0, 0.0
                speed_px = 0.0
                speed_kmh = 0.0
                speed_px_s = 0.0
                bearing = 0.0

                if track_id is not None:
                    if track_id in self.prev_positions:
                        px, py = self.prev_positions[track_id]
                        dx = float(curr_center[0] - px)
                        dy = float(curr_center[1] - py)
                        displacement = math.hypot(dx, dy)
                        speed_px = displacement

                        # Speed calculation for moving and walking objects
                        if displacement > 0.2:
                            raw_px_s = displacement * effective_fps
                            # Add to rolling history
                            self.speed_history[track_id].append(raw_px_s)
                            smooth_px_s = sum(self.speed_history[track_id]) / len(self.speed_history[track_id])
                            speed_px_s = smooth_px_s
                            # Convert pixels/s to km/h using pixels_per_meter
                            ppm = max(1.0, self.pixels_per_meter)
                            speed_kmh = (smooth_px_s / ppm) * 3.6
                        else:
                            self.speed_history[track_id].append(0.0)
                            speed_kmh = 0.0
                            speed_px_s = 0.0

                        rad = math.atan2(dx, -dy)
                        bearing = (math.degrees(rad) + 360.0) % 360.0

                    self.prev_positions[track_id] = curr_center
                    active_ids.append(track_id)
                    centers[track_id] = curr_center

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
                )
                detections.append(det)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                if speed_kmh > max_speed:
                    max_speed = speed_kmh
                    fastest_det = det

                if self.locked_target_id is not None and track_id == self.locked_target_id:
                    locked_det = det

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

        # Color coding by speed & primary status
        if is_primary:
            color = (0, 0, 255)  # Red for user-selected target
        elif det.speed_kmh > 80:
            color = (0, 69, 255)  # Orange-Red for fast moving
        elif det.speed_kmh > 40:
            color = (0, 215, 255)  # Amber
        else:
            color = (255, 200, 50)  # Cyan

        thickness = 2
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
        if is_primary:
            badge_text = f"TARGET #{det.track_id} {det.class_name} | {speed_text}"
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
        Renders Picture-in-Picture Auto-Follow Zoom with live SPEED telemetry HUD.
        """
        fh, fw = canvas.shape[:2]
        cx, cy = target.center

        # Smooth camera movement
        if self.smooth_pip_center is None:
            self.smooth_pip_center = (float(cx), float(cy))
        else:
            scx, scy = self.smooth_pip_center
            self.smooth_pip_center = (0.75 * scx + 0.25 * cx, 0.75 * scy + 0.25 * cy)

        tcx, tcy = int(self.smooth_pip_center[0]), int(self.smooth_pip_center[1])

        crop_half_w = max(30, int(fw / (self.zoom_factor * 2)))
        crop_half_h = max(24, int(fh / (self.zoom_factor * 2)))

        x1_crop = max(0, tcx - crop_half_w)
        y1_crop = max(0, tcy - crop_half_h)
        x2_crop = min(fw, tcx + crop_half_w)
        y2_crop = min(fh, tcy + crop_half_h)

        if x2_crop <= x1_crop or y2_crop <= y1_crop:
            return

        cropped = clean_source[y1_crop:y2_crop, x1_crop:x2_crop]
        if cropped.size == 0:
            return

        pip_w = int(fw * 0.36)
        pip_h = int(fh * 0.32)

        try:
            zoomed = cv2.resize(cropped, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return

        margin = 12
        top_y = 44
        left_x = fw - pip_w - margin
        bottom_y = top_y + pip_h
        right_x = left_x + pip_w

        canvas[top_y:bottom_y, left_x:right_x] = zoomed

        # Red Border
        border_color = (0, 0, 255)
        cv2.rectangle(canvas, (left_x, top_y), (right_x, bottom_y), border_color, 2, cv2.LINE_AA)

        # Center Reticle
        pip_cx = left_x + pip_w // 2
        pip_cy = top_y + pip_h // 2
        ret_len = 8
        cv2.line(canvas, (pip_cx - ret_len, pip_cy), (pip_cx + ret_len, pip_cy), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (pip_cx, pip_cy - ret_len), (pip_cx, pip_cy + ret_len), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (pip_cx, pip_cy), 12, (0, 0, 255), 1, cv2.LINE_AA)

        # Header Badge
        tag_str = f"ZOOM TARGET {self.zoom_factor:.1f}X | #{target.track_id} {target.class_name}"
        cv2.rectangle(canvas, (left_x, top_y), (left_x + 245, top_y + 18), (10, 15, 22), -1)
        cv2.putText(canvas, tag_str, (left_x + 5, top_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        # Bottom Speed Banner inside Zoom Window
        speed_banner = f"KECEPATAN: {target.speed_kmh:.1f} KM/H  ({target.speed_kmh / 3.6:.1f} m/s)"
        cv2.rectangle(canvas, (left_x, bottom_y - 20), (right_x, bottom_y), (10, 15, 22), -1)
        cv2.putText(canvas, speed_banner, (left_x + 8, bottom_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1, cv2.LINE_AA)

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
