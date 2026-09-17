# 🎯 VisionTrack AI - Object Detection & Multi-Object Tracking

Proyek **Object Detection & Multi-Object Tracking** mutakhir berbasis Python menggunakan **Ultralytics YOLO (YOLO26 / YOLO11 / YOLOv8)** terintegrasi dengan algoritma pelacak teruji **ByteTrack & BoT-SORT**. Mendukung seluruh varian YOLO26 (Nano, Small, Medium, Large, Extra Large) dengan arsitektur *End-to-End NMS-Free*.

Dilengkapi dengan dua mode penggunaan:
1. **Desktop OpenCV Mode (`main.py`)**: Jendela visualisasi desktop interaktif dengan latensi ultra-rendah dan tombol shortcut (*hotkeys*).
2. **Web Dashboard Mode (`web_app.py`)**: Antarmuka web modern (*Dark Glassmorphism*) berbasis FastAPI dengan live stream MJPEG, statistik analitik real-time, dan kontrol parameter dinamis.

---

## 🚀 Fitur Utama

- **Real-Time Object Detection**: Mendeteksi 80 kategori objek (COCO dataset: orang, mobil, motor, tas, hewan, dll.) secara akurat.
- **Multi-Object Tracking (MOT)**: Memberikan ID unik permanen pada setiap objek (`#1`, `#2`, dst.) menggunakan **ByteTrack** atau **BoT-SORT**.
- **Motion Trajectory (Trails)**: Menggambar jejak lintasan pergerakan setiap objek untuk menganalisis arah pergerakan.
- **Line Crossing / Tripwire Counter**: Menghitung otomatis objek yang melintasi garis batas virtual (penghitung Masuk / Keluar / Total).
- **Fleksibilitas Input**: Mendukung webcam lokal (`0`), berkas video (`.mp4`, `.avi`, dll.), atau RTSP stream.
- **UI Web Modern**: Dashboard berbasis FastAPI + HTML5 Vanilla Glassmorphic dengan slider *confidence*, pemilih model, drag-and-drop video upload, dan badge metrik real-time.

---

## 📂 Struktur Proyek

```text
object_detection_tracking/
├── core/
│   ├── __init__.py
│   ├── detector.py      # Engine utama: YOLO tracking, anotasi bounding box & HUD
│   ├── tracker.py       # Pengelola visual jejak lintasan (motion trails) & warna unik
│   └── counter.py       # Logika penghitung persilangan garis (line crossing counter)
├── templates/
│   └── index.html       # Tampilan antarmuka Web Dashboard
├── static/
│   ├── app.css          # Desain modern Glassmorphic Dark UI
│   └── app.js           # Logika interaktif frontend (polling data, upload, slider)
├── uploads/             # Direktori penyimpanan video yang diunggah via web
├── main.py              # Runner mode Desktop (OpenCV window)
├── web_app.py           # Runner mode Web Dashboard (FastAPI server)
├── requirements.txt     # Daftar pustaka Python
└── README.md            # Dokumentasi panduan
```

---

## 🛠️ Panduan Instalasi

### 1. Prasyarat
Pastikan Python 3.10+ telah terpasang di sistem Anda.

### 2. Instalasi Dependensi
Buka terminal / PowerShell di folder proyek ini, lalu jalankan:
```bash
pip install -r requirements.txt
```

---

## 💻 Cara Menjalankan

### Mode 1: Desktop OpenCV (`main.py`)
Gunakan mode ini untuk webcam langsung dengan latensi minimal.

```bash
# 1. Menjalankan dengan webcam default (device 0)
python main.py

# 2. Menjalankan dengan berkas video
python main.py --source jalan_raya.mp4

# 3. Menjalankan dengan model tertentu dan tracker BoT-SORT
python main.py --model yolo11n.pt --tracker botsort.yaml --conf 0.40

# 4. Menyimpan hasil video ke berkas MP4
python main.py --source video.mp4 --save hasil_tracking.mp4
```

#### Tombol Kontrol Keyboard (Hotkeys):
| Tombol | Fungsi |
|---|---|
| `q` atau `ESC` | Menutup aplikasi |
| `SPACE` | Jeda (Pause) / Lanjutkan pemutaran |
| `t` | Mengaktifkan / Menonaktifkan Jejak Lintasan (*Trails*) |
| `c` | Mengaktifkan / Menonaktifkan Garis Penghitung (*Line Counter*) |
| `r` | Mereset hitungan objek yang melintas |
| `s` | Mengambil tangkapan layar (*Screenshot*) ke berkas JPG |

---

### Mode 2: Web Dashboard Interaktif (`web_app.py`)
Gunakan mode ini untuk pengalaman visual terbaik dengan dashboard berbasis web.

```bash
python web_app.py
```
Atau menggunakan uvicorn:
```bash
uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Buka peramban (browser) Anda di:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

**Fitur di Web Dashboard:**
- **Live Video Feed**: Streaming MJPEG real-time dengan anotasi lengkap.
- **KPI Real-time**: Jumlah objek aktif saat ini, FPS inferensi, serta jumlah total yang melintas (IN/OUT).
- **Pilihan Sumber**: Beralih antara Webcam laptop atau Mengunggah berkas video sendiri melalui drag-and-drop.
- **Interactive Controls**: Ubah *Confidence Threshold* (10% - 90%), ganti model YOLO (`yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt`, `yolo11n.pt`, `yolov8n.pt`), aktifkan/nonaktifkan trails dan counter tanpa perlu restart server.
