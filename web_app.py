import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.detector import YOLOTrackerEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


DEFAULT_LIVE_URL = "https://atcs-dishub.bandung.go.id:1990/MochToha/index.m3u8"


class StreamManager:
    """
    Dedicated background capture worker that keeps draining live M3U8/RTSP network frames,
    eliminating buffer lag so the application always processes the latest live frame in real time.
    """
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.source_path: Optional[str] = None
        self.source_type: str = "live_stream"
        self.consecutive_fails: int = 0

    def start(self, source_path: str, source_type: str = "live_stream") -> bool:
        self.stop()
        self.source_path = source_path
        self.source_type = source_type
        self.consecutive_fails = 0

        is_network = (
            source_path.startswith("http://")
            or source_path.startswith("https://")
            or source_path.startswith("rtsp://")
        )

        print(f"Membuka sumber video: {source_path}")
        if is_network:
            self.cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source_path)
        else:
            if not os.path.exists(source_path):
                print(f"File tidak ditemukan: {source_path}")
                return False
            self.cap = cv2.VideoCapture(source_path)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print(f"Gagal membuka capture: {source_path}")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"Stream worker aktif: {source_path}")
        return True

    def _worker(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.consecutive_fails = 0
                with self.lock:
                    self.latest_frame = frame

                if self.source_type == "video":
                    # Pace local video to ~30 FPS
                    time.sleep(0.03)
            else:
                if self.source_type == "video":
                    # Loop local video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                else:
                    self.consecutive_fails += 1
                    if self.consecutive_fails >= 8:
                        print("Stream terputus di latar belakang, menghubungkan ulang...")
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = cv2.VideoCapture(self.source_path, cv2.CAP_FFMPEG)
                        self.consecutive_fails = 0
                    time.sleep(0.04)

    def read_latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.6)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self.lock:
            self.latest_frame = None


class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.stream_manager = StreamManager()
        self.source_type = "live_stream"
        self.source_path: Optional[str] = DEFAULT_LIVE_URL
        self.engine: Optional[YOLOTrackerEngine] = None
        self.latest_stats = {
            "fps": 0.0,
            "acquisition_rate": "0 Hz",
            "active_objects": 0,
            "classes": {},
            "tactical_mode": "all",
            "sensor_mode": "eo",
            "locked_target": None,
            "source_type": "live_stream",
            "is_live": True,
        }
        self.is_running = True
        self.paused = False
        self.video_name = "🔴 LIVE CCTV: ATCS Moch Toha Bandung"

    def init_engine(self, model_name: str = "yolo11n.pt", conf: float = 0.25):
        with self.lock:
            self.engine = YOLOTrackerEngine(
                model_name=model_name,
                conf_threshold=conf,
                tactical_mode="all",
                sensor_mode="eo",
            )

    def init_capture(self) -> bool:
        with self.lock:
            if not self.source_path:
                return False
            return self.stream_manager.start(self.source_path, self.source_type)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Memuat SpectraTrack AI Engine...")
    state.init_engine()
    state.init_capture()
    print("SpectraTrack AI siap di http://127.0.0.1:8000")
    yield
    state.is_running = False
    state.stream_manager.stop()


app = FastAPI(title="SpectraTrack AI - Detection & Multi-Object Tracking", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def make_waiting_frame() -> bytes:
    """Frame placeholder sebelum video diunggah."""
    w, h = 640, 480
    frame = np.full((h, w, 3), 16, dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (30, 45, 60), 1)
    # Tech circle
    cv2.circle(frame, (w // 2, h // 2 - 25), 35, (6, 182, 212), 1)
    cv2.line(frame, (w // 2 - 45, h // 2 - 25), (w // 2 + 45, h // 2 - 25), (6, 182, 212), 1)
    cv2.line(frame, (w // 2, h // 2 - 70), (w // 2, h // 2 + 20), (6, 182, 212), 1)

    cv2.putText(
        frame,
        "SPECTRATRACK AI: SIAP MENGANALISIS",
        (w // 2 - 180, h // 2 + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (6, 182, 212),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Menghubungkan ke siaran video / stream CCTV...",
        (w // 2 - 190, h // 2 + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 175, 190),
        1,
        cv2.LINE_AA,
    )
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes()


def generate_mjpeg():
    waiting_bytes = make_waiting_frame()

    while state.is_running:
        if state.paused:
            time.sleep(0.04)
            continue

        frame = state.stream_manager.read_latest()
        if frame is None:
            # Belum ada frame / menghubungkan
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + waiting_bytes + b"\r\n"
            )
            time.sleep(0.08)
            continue

        # AI Detection & Tracking
        with state.lock:
            if state.engine is not None:
                annotated_frame, detections, stats = state.engine.process_frame(frame)
                stats["video_name"] = state.video_name
                stats["source_type"] = state.source_type
                stats["is_live"] = (state.source_type == "live_stream")
                state.latest_stats = stats
            else:
                annotated_frame = frame

        # JPEG Encoding & Yield di LUAR lock
        success, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.012)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stats")
def get_stats():
    return JSONResponse(state.latest_stats)


class SettingsPayload(BaseModel):
    conf_threshold: Optional[float] = None
    enable_trails: Optional[bool] = None
    enable_counter: Optional[bool] = None
    enable_pip_zoom: Optional[bool] = None
    zoom_factor: Optional[float] = None
    pixels_per_meter: Optional[float] = None
    model_name: Optional[str] = None
    tactical_mode: Optional[str] = None
    sensor_mode: Optional[str] = None
    lock_id: Optional[int] = None


@app.post("/api/settings")
def update_settings(payload: SettingsPayload):
    with state.lock:
        if payload.conf_threshold is not None:
            state.engine.conf_threshold = payload.conf_threshold
        if payload.enable_trails is not None:
            state.engine.enable_trails = payload.enable_trails
        if payload.enable_counter is not None:
            state.engine.enable_counter = payload.enable_counter
        if payload.enable_pip_zoom is not None:
            state.engine.enable_pip_zoom = payload.enable_pip_zoom
        if payload.zoom_factor is not None:
            state.engine.zoom_factor = payload.zoom_factor
        if payload.pixels_per_meter is not None:
            state.engine.pixels_per_meter = payload.pixels_per_meter
        if payload.tactical_mode is not None:
            state.engine.tactical_mode = payload.tactical_mode
        if payload.sensor_mode is not None:
            state.engine.sensor_mode = payload.sensor_mode
        if payload.lock_id is not None:
            state.engine.locked_target_id = payload.lock_id
            state.engine.is_user_locked = True
        if payload.model_name and payload.model_name != state.engine.model_name:
            state.engine = YOLOTrackerEngine(
                model_name=payload.model_name,
                conf_threshold=state.engine.conf_threshold,
                enable_trails=state.engine.enable_trails,
                enable_counter=state.engine.enable_counter,
                enable_pip_zoom=state.engine.enable_pip_zoom,
                zoom_factor=state.engine.zoom_factor,
                pixels_per_meter=state.engine.pixels_per_meter,
                tactical_mode=state.engine.tactical_mode,
                sensor_mode=state.engine.sensor_mode,
            )

    return {"status": "ok", "message": "Konfigurasi diperbarui"}


class TargetSelectPayload(BaseModel):
    norm_x: Optional[float] = None
    norm_y: Optional[float] = None
    track_id: Optional[int] = None


@app.post("/api/select_target")
def select_target(payload: TargetSelectPayload):
    with state.lock:
        if state.engine is None:
            return {"status": "error", "message": "Engine belum siap"}

        if payload.track_id is not None:
            state.engine.locked_target_id = payload.track_id
            state.engine.is_user_locked = True
            state.engine.smooth_pip_center = None
            return {"status": "ok", "locked_id": payload.track_id}
        elif payload.norm_x is not None and payload.norm_y is not None:
            locked_id, class_name = state.engine.select_target_by_coord(payload.norm_x, payload.norm_y)
            return {
                "status": "ok" if locked_id is not None else "not_found",
                "locked_id": locked_id,
                "class_name": class_name,
            }

    return {"status": "error", "message": "Parameter tidak valid"}


@app.post("/api/release_target")
def release_target():
    with state.lock:
        if state.engine is not None:
            state.engine.release_target()
    return {"status": "ok", "message": "Kunci target dilepaskan"}


class SourceSelectPayload(BaseModel):
    source_type: str  # "live" | "sample" | "custom_url"
    url: Optional[str] = None


@app.post("/api/set_source")
def set_source(payload: SourceSelectPayload):
    with state.lock:
        if payload.source_type == "live":
            url = payload.url or DEFAULT_LIVE_URL
            state.source_type = "live_stream"
            state.source_path = url
            state.video_name = "🔴 LIVE CCTV: ATCS Moch Toha Bandung" if "MochToha" in url else f"🔴 LIVE: {url}"
        elif payload.source_type == "sample":
            default_video = os.path.join(BASE_DIR, "sample_test.mp4")
            state.source_type = "video"
            state.source_path = default_video
            state.video_name = "sample_test.mp4"
        elif payload.source_type == "custom_url" and payload.url:
            state.source_type = "live_stream"
            state.source_path = payload.url.strip()
            state.video_name = f"🔴 STREAM: {state.source_path[:32]}..."

        if state.engine is not None:
            state.engine.release_target()
            state.engine.visualizer.track_history.clear()

        success = state.init_capture()
        return {
            "status": "ok" if success else "error",
            "video_name": state.video_name,
            "source_type": state.source_type,
            "source_path": state.source_path,
        }


@app.post("/api/toggle_pause")
def toggle_pause():
    state.paused = not state.paused
    return {"paused": state.paused}


@app.post("/api/reset_counter")
def reset_counter():
    with state.lock:
        if state.engine is not None:
            state.engine.counter.in_count = 0
            state.engine.counter.out_count = 0
            state.engine.counter.total_count = 0
            state.engine.counter.counted_ids.clear()
            state.engine.visualizer.track_history.clear()
    return {"status": "ok", "message": "Penghitung berhasil di-reset"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Sanitize filename
        safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
        if not safe_name:
            safe_name = "video.mp4"
        filename = f"upload_{int(time.time())}_{safe_name}"
        filepath = os.path.join(UPLOADS_DIR, filename)

        with open(filepath, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        with state.lock:
            state.source_type = "video"
            state.source_path = filepath
            state.video_name = file.filename
            success = state.init_capture()
            if state.engine is not None:
                state.engine.visualizer.track_history.clear()
                state.engine.locked_target_id = None
                state.engine.smooth_pip_center = None

        if not success:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Video berhasil disimpan, namun format codec tidak dapat diputar OpenCV. Silakan gunakan format MP4 standard (H.264/AVC).",
                },
            )

        return {"status": "ok", "filename": filename, "message": "Video berhasil diunggah dan siap dianalisis!"}
    except Exception as e:
        print("Upload error:", e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Terjadi kesalahan: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
