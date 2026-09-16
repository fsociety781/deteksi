from typing import Dict, List, Set, Tuple
import cv2
import numpy as np


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Check if three points are listed in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]) -> bool:
    """Return True if line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class LineCounter:
    """
    Counts objects crossing a defined 2D virtual tripwire line.
    """

    def __init__(
        self,
        start_point: Tuple[int, int] = (100, 250),
        end_point: Tuple[int, int] = (540, 250),
        line_color: Tuple[int, int, int] = (0, 220, 255),
    ):
        self.start = start_point
        self.end = end_point
        self.line_color = line_color

        self.in_count = 0
        self.out_count = 0
        self.total_count = 0

        # Memory of counted tracks
        self.counted_ids: Set[int] = set()
        self.prev_positions: Dict[int, Tuple[int, int]] = {}

    def set_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = start
        self.end = end

    def update(self, current_centers: Dict[int, Tuple[int, int]], class_names: Dict[int, str] = None):
        """
        Check if any object center crossed the line between the previous and current frame.
        """
        for track_id, curr_pt in current_centers.items():
            if track_id in self.counted_ids:
                self.prev_positions[track_id] = curr_pt
                continue

            if track_id in self.prev_positions:
                prev_pt = self.prev_positions[track_id]

                # Check if segment (prev_pt -> curr_pt) intersects (self.start -> self.end)
                if intersect(prev_pt, curr_pt, self.start, self.end):
                    # Determine directional vector relative to the line normal
                    # Vector of line
                    lx = self.end[0] - self.start[0]
                    ly = self.end[1] - self.start[1]
                    # Normal vector (-ly, lx)
                    # Motion vector
                    mx = curr_pt[0] - prev_pt[0]
                    my = curr_pt[1] - prev_pt[1]
                    dot = (-ly) * mx + (lx) * my

                    if dot > 0:
                        self.in_count += 1
                    else:
                        self.out_count += 1

                    self.total_count += 1
                    self.counted_ids.add(track_id)

            self.prev_positions[track_id] = curr_pt

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the virtual line and badge on the frame.
        """
        # Draw neon line
        cv2.line(frame, self.start, self.end, self.line_color, 3, cv2.LINE_AA)
        cv2.circle(frame, self.start, 6, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, self.end, 6, (0, 165, 255), -1, cv2.LINE_AA)

        # Draw HUD label near the line midpoint
        mid_x = (self.start[0] + self.end[0]) // 2
        mid_y = (self.start[1] + self.end[1]) // 2 - 10
        cv2.putText(
            frame,
            f"COUNT LINE [Crossed: {self.total_count}]",
            (max(10, mid_x - 110), max(25, mid_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return frame
