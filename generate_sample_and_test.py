"""
Script to create a synthetic video clip and run verification tests on YOLOTrackerEngine.
"""
import os
import sys
import numpy as np
import cv2

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.detector import YOLOTrackerEngine


def create_synthetic_video(output_path: str = "sample_test.mp4", duration_sec: int = 4, fps: int = 25):
    """
    Generate a synthetic video with moving objects to test tracking and line crossing.
    """
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    print(f"Creating synthetic test video: {output_path} ({total_frames} frames)...")

    for i in range(total_frames):
        # Dark canvas
        frame = np.full((h, w, 3), 30, dtype=np.uint8)

        # Draw a simulated road / ground plane
        cv2.rectangle(frame, (0, 100), (w, 400), (45, 50, 60), -1)
        cv2.line(frame, (0, 250), (w, 250), (100, 100, 100), 2, cv2.LINE_AA)

        # Object 1: Moving left to right across the vertical/horizontal plane
        x1 = int(50 + (i / total_frames) * 500)
        y1 = int(220 + np.sin(i / 10.0) * 20)
        cv2.rectangle(frame, (x1, y1), (x1 + 60, y1 + 50), (60, 160, 240), -1)
        cv2.putText(frame, "TEST OBJ 1", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Object 2: Moving top to bottom crossing the middle line (y=250)
        x2 = 320
        y2 = int(120 + (i / total_frames) * 260)
        cv2.circle(frame, (x2, y2), 25, (80, 220, 100), -1)
        cv2.putText(frame, "TEST OBJ 2", (x2 - 30, y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        writer.write(frame)

    writer.release()
    print("Synthetic video created successfully.")


def run_verification():
    print("\n--- Running Automated Engine Verification ---")
    sample_path = "sample_test.mp4"
    if not os.path.exists(sample_path):
        create_synthetic_video(sample_path)

    print("Initializing YOLOTrackerEngine (model=yolo11n.pt)...")
    engine = YOLOTrackerEngine(model_name="yolo11n.pt", conf_threshold=0.25)

    cap = cv2.VideoCapture(sample_path)
    assert cap.isOpened(), "Failed to open synthetic test video!"

    processed_count = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret or processed_count >= 30:
            break

        annotated_frame, detections, stats = engine.process_frame(frame)
        processed_count += 1
        total_detections += len(detections)

    cap.release()

    print(f"Frames processed: {processed_count}")
    print(f"Engine FPS: {engine.current_fps:.1f}")
    print(f"Total counted: {engine.counter.total_count}")
    print("Engine verified successfully! All core methods executed without exceptions.")


if __name__ == "__main__":
    create_synthetic_video("sample_test.mp4")
    run_verification()
