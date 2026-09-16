import colorsys
from collections import defaultdict, deque
from typing import Dict, List, Tuple
import cv2
import numpy as np


class TrackVisualizer:
    """
    Manages track visual properties like unique distinct colors and motion trails.
    """

    def __init__(self, max_trail_len: int = 30, max_inactive_frames: int = 30):
        self.max_trail_len = max_trail_len
        self.max_inactive_frames = max_inactive_frames
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trail_len))
        self.last_seen: Dict[int, int] = {}
        self.frame_count = 0

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Generate a deterministic, vibrant, aesthetically pleasing BGR color for an ID.
        """
        golden_ratio = 0.618033988749895
        hue = (track_id * golden_ratio) % 1.0
        saturation = 0.85
        value = 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Return as OpenCV BGR tuple
        return int(b * 255), int(g * 255), int(r * 255)

    def update(self, active_track_ids: List[int], centers: Dict[int, Tuple[int, int]]):
        """
        Record the center point for currently active tracks and remove stale tracks.
        """
        self.frame_count += 1
        for tid, pt in centers.items():
            self.track_history[tid].append(pt)
            self.last_seen[tid] = self.frame_count

        # Clean up tracks not seen for max_inactive_frames
        stale_ids = [
            tid for tid, last_f in self.last_seen.items()
            if self.frame_count - last_f > self.max_inactive_frames
        ]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.last_seen.pop(tid, None)

    def draw_trails(self, frame: np.ndarray, thickness: int = 2) -> np.ndarray:
        """
        Draw smoothed motion trajectories on the frame.
        """
        for track_id, points in self.track_history.items():
            if len(points) < 2:
                continue
            color = self.get_color(track_id)
            pts = list(points)
            for i in range(1, len(pts)):
                # Dynamically taper line thickness from tail to head
                t = max(1, int(thickness * (i / len(pts))))
                cv2.line(frame, pts[i - 1], pts[i], color, t, cv2.LINE_AA)
        return frame
