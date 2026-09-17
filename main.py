import argparse
import sys
import time
import cv2

from core.detector import YOLOTrackerEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection and Tracking with YOLO & ByteTrack/BoT-SORT"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: '0' for default webcam, or path to video file (e.g. video.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="YOLO model checkpoint (e.g. yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt, yolo11n.pt, yolov8n.pt)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm config",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (0.1 to 1.0)",
    )
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable motion trajectory trails",
    )
    parser.add_argument(
        "--no-counter",
        action="store_true",
        help="Disable virtual line counter",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save output video (.mp4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine source (integer for webcam, string for video path)
    if args.source.isdigit():
        source = int(args.source)
        is_webcam = True
    else:
        source = args.source
        is_webcam = False

    print("=" * 60)
    print("  🚀 YOLO Object Detection & Multi-Object Tracking")
    print("=" * 60)
    print(f" Source   : {args.source}")
    print(f" Model    : {args.model}")
    print(f" Tracker  : {args.tracker}")
    print(f" Conf Thresh: {args.conf}")
    print(" Controls :")
    print("   [q / ESC] : Keluar")
    print("   [SPACE]   : Jeda / Lanjut (Pause / Resume)")
    print("   [t]       : Toggle Jejak Lintasan (Trails)")
    print("   [c]       : Toggle Garis Penghitung (Line Counter)")
    print("   [r]       : Reset Hitungan")
    print("   [s]       : Ambil Tangkapan Layar (Screenshot)")
    print("=" * 60)

    print("\n⏳ Memuat model YOLO dan tracker...")
    engine = YOLOTrackerEngine(
        model_name=args.model,
        tracker_type=args.tracker,
        conf_threshold=args.conf,
        enable_trails=not args.no_trails,
        enable_counter=not args.no_counter,
    )
    print("✅ Model siap!\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Tidak dapat membuka sumber video: {source}")
        sys.exit(1)

    # Setup video writer if requested
    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"💾 Menyimpan video hasil ke: {args.save}")

    # Set default line based on frame size
    ret, test_frame = cap.read()
    if ret:
        fh, fw = test_frame.shape[:2]
        engine.set_line_coords((int(fw * 0.1), int(fh * 0.55)), (int(fw * 0.9), int(fh * 0.55)))
        # rewind if it was a video file
        if not is_webcam:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    window_name = "YOLO Detection & Tracking (Press Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if not is_webcam:
                        print("\n🏁 Pemutaran video selesai.")
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("\n⚠️ Kehilangan feed kamera.")
                        break

                annotated_frame, detections, stats = engine.process_frame(frame)

                if writer is not None:
                    writer.write(annotated_frame)

                cv2.imshow(window_name, annotated_frame)

            # Keyboard handler
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), 27]:  # 'q' or ESC
                print("\nMenutup aplikasi...")
                break
            elif key == ord(" "):  # SPACE
                paused = not paused
                print("⏸️ Dijeda" if paused else "▶️ Dilanjutkan")
            elif key == ord("t"):  # 't'
                engine.enable_trails = not engine.enable_trails
                status = "Aktif" if engine.enable_trails else "Nonaktif"
                print(f"🌀 Motion Trails: {status}")
            elif key == ord("c"):  # 'c'
                engine.enable_counter = not engine.enable_counter
                status = "Aktif" if engine.enable_counter else "Nonaktif"
                print(f"📏 Line Counter: {status}")
            elif key == ord("r"):  # 'r'
                engine.counter.in_count = 0
                engine.counter.out_count = 0
                engine.counter.total_count = 0
                engine.counter.counted_ids.clear()
                print("🔄 Hitungan berhasil di-reset!")
            elif key == ord("s"):  # 's'
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Tangkapan layar disimpan: {filename}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == "__main__":
    main()
