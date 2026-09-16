document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const videoFeed = document.getElementById("videoFeed");
  const streamPlaceholder = document.getElementById("streamPlaceholder");
  const btnPause = document.getElementById("btnPause");
  const btnRefresh = document.getElementById("btnRefresh");
  const lblMissionFile = document.getElementById("lblMissionFile");
  const lblBearing = document.getElementById("lblBearing");
  const lblLockStatus = document.getElementById("lblLockStatus");

  // Telemetry Metrics
  const txtRate = document.getElementById("txtRate");
  const lockBadge = document.getElementById("lockBadge");
  const tValId = document.getElementById("tValId");
  const tValClass = document.getElementById("tValClass");
  const tValConf = document.getElementById("tValConf");
  const tValSpeed = document.getElementById("tValSpeed");
  const tValBearing = document.getElementById("tValBearing");
  const tValCoords = document.getElementById("tValCoords");

  const valActiveTracks = document.getElementById("valActiveTracks");
  const classesContainer = document.getElementById("classesContainer");

  // Speedometer Elements & Calibration Controls
  const valMainSpeed = document.getElementById("valMainSpeed");
  const valSpeedSubtext = document.getElementById("valSpeedSubtext");
  const scaleSlider = document.getElementById("scaleSlider");
  const scaleValDisplay = document.getElementById("scaleValDisplay");
  const scalePresetBtns = document.querySelectorAll(".scale-preset-btn");

  // Mode & Sensor Buttons
  const modeButtons = document.querySelectorAll(".mode-pill");
  const sensorButtons = document.querySelectorAll(".sensor-pill");

  // Upload Controls
  const dropzone = document.getElementById("dropzone");
  const videoFileInput = document.getElementById("videoFileInput");
  const uploadProgress = document.getElementById("uploadProgress");
  const progressFill = document.getElementById("progressFill");
  const uploadStatusText = document.getElementById("uploadStatusText");

  // Forms
  const confSlider = document.getElementById("confSlider");
  const confValDisplay = document.getElementById("confValDisplay");
  const modelSelect = document.getElementById("modelSelect");
  const chkTrails = document.getElementById("chkTrails");

  let isPaused = false;

  // 1. Fetch & Update Telemetry
  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const data = await res.json();

      if (data.fps) {
        txtRate.textContent = `${(data.fps || 0).toFixed(1)} FPS`;
      }
      valActiveTracks.textContent = data.active_objects || 0;

      if (data.video_name && lblMissionFile) {
        lblMissionFile.textContent = data.video_name;
      }

      // Update Primary Target Focus Card
      const locked = data.locked_target;
      if (locked && locked.id !== null) {
        lockBadge.textContent = "TERKUNCI";
        lockBadge.style.color = "#10b981";
        tValId.textContent = `#${locked.id}`;
        tValClass.textContent = locked.class || "OBJECT";
        tValConf.textContent = locked.conf || "--";
        tValSpeed.textContent = locked.speed_kmh || "0.0 km/h";
        tValBearing.textContent = locked.bearing || "--";
        tValCoords.textContent = locked.coords || "--";
        lblLockStatus.textContent = `TARGET TERFOKUS: #${locked.id}`;
        lblBearing.textContent = locked.bearing || "000°";

        if (valMainSpeed) {
          const numSpeed = parseFloat(locked.speed_kmh) || 0.0;
          let speedColor = "#10b981";
          if (numSpeed > 70) speedColor = "#ef4444";
          else if (numSpeed > 40) speedColor = "#f59e0b";
          valMainSpeed.style.color = speedColor;
          valMainSpeed.innerHTML = `${numSpeed.toFixed(1)} <small style="font-size: 1rem; color: var(--text-dark); font-weight: 500;">km/h</small>`;
        }
        if (valSpeedSubtext) {
          valSpeedSubtext.textContent = `${locked.speed_ms || '0.0 m/s'} | ${locked.speed_px || '0 px/s'}`;
        }
      } else {
        lockBadge.textContent = "MEMINDAI";
        lockBadge.style.color = "#f59e0b";
        tValId.textContent = "MEMINDAI";
        tValClass.textContent = "--";
        tValConf.textContent = "--";
        tValSpeed.textContent = "0.0 km/h";
        tValBearing.textContent = "--";
        tValCoords.textContent = "--";
        lblLockStatus.textContent = "MEMINDAI FRAME";

        if (valMainSpeed) {
          valMainSpeed.style.color = "#10b981";
          valMainSpeed.innerHTML = `0.0 <small style="font-size: 1rem; color: var(--text-dark); font-weight: 500;">km/h</small>`;
        }
        if (valSpeedSubtext) {
          valSpeedSubtext.textContent = "0.0 m/s | 0 px/s";
        }
      }

      // Update Detected Classes Badges
      const classes = data.classes || {};
      const keys = Object.keys(classes);
      if (keys.length === 0) {
        classesContainer.innerHTML = `<span class="empty-notice">Menunggu deteksi objek pada video...</span>`;
      } else {
        classesContainer.innerHTML = keys
          .map(
            (cls) => `
          <div class="category-chip">
            <span>${cls.toUpperCase()}</span>
            <span class="chip-counter">${classes[cls]}</span>
          </div>
        `
          )
          .join("");
      }
    } catch (err) {
      console.warn("Stats warning:", err);
    }
  }

  setInterval(fetchStats, 500);

  // 2. Settings Helper
  async function updateSettings(payload) {
    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      fetchStats();
    } catch (err) {
      console.error("Gagal memperbarui pengaturan:", err);
    }
  }

  // 3. Operational Mode Switching
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const mode = btn.getAttribute("data-mode");
      updateSettings({ tactical_mode: mode });
    });
  });

  // 4. Sensor Spectrum Switching (Optical RGB / FLIR Thermal / Night Vision)
  sensorButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      sensorButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const sensor = btn.getAttribute("data-sensor");
      updateSettings({ sensor_mode: sensor });
      btnRefresh.click();
    });
  });

  // 5. Video Controls
  btnPause.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/toggle_pause", { method: "POST" });
      const data = await res.json();
      isPaused = data.paused;
      btnPause.innerHTML = isPaused ? "▶️ LANJUT" : "⏸️ JEDA";
    } catch (err) {
      console.error(err);
    }
  });

  btnRefresh.addEventListener("click", () => {
    const src = videoFeed.src.split("?")[0];
    videoFeed.src = `${src}?t=${new Date().getTime()}`;
  });

  // 6. Confidence Threshold
  let debounceTimer = null;
  confSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    confValDisplay.textContent = `${val}%`;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      updateSettings({ conf_threshold: parseFloat(val) / 100.0 });
    }, 200);
  });

  // Speed Scale Calibration (Pixels per Meter)
  let scaleDebounce = null;
  if (scaleSlider && scaleValDisplay) {
    scaleSlider.addEventListener("input", (e) => {
      const val = e.target.value;
      scaleValDisplay.textContent = `${val} px/m`;
      scalePresetBtns.forEach((b) => {
        if (b.getAttribute("data-scale") === val) {
          b.classList.add("active");
        } else {
          b.classList.remove("active");
        }
      });
      clearTimeout(scaleDebounce);
      scaleDebounce = setTimeout(() => {
        updateSettings({ pixels_per_meter: parseFloat(val) });
      }, 250);
    });
  }

  scalePresetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      scalePresetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const scaleVal = btn.getAttribute("data-scale");
      if (scaleSlider) scaleSlider.value = scaleVal;
      if (scaleValDisplay) scaleValDisplay.textContent = `${scaleVal} px/m`;
      updateSettings({ pixels_per_meter: parseFloat(scaleVal) });
    });
  });

  // 7. Inference Model
  modelSelect.addEventListener("change", (e) => {
    updateSettings({ model_name: e.target.value });
  });

  const chkPipZoom = document.getElementById("chkPipZoom");
  const selectZoomLevel = document.getElementById("selectZoomLevel");

  // 8. Visual Toggles & Zoom
  if (chkPipZoom) {
    chkPipZoom.addEventListener("change", (e) => {
      updateSettings({ enable_pip_zoom: e.target.checked });
    });
  }

  if (selectZoomLevel) {
    selectZoomLevel.addEventListener("change", (e) => {
      updateSettings({ zoom_factor: parseFloat(e.target.value) });
    });
  }

  chkTrails.addEventListener("change", (e) => {
    updateSettings({ enable_trails: e.target.checked });
  });

  // 9. Video Upload (Drag & Drop)
  dropzone.addEventListener("click", () => videoFileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#06b6d4";
    dropzone.style.background = "rgba(6, 182, 212, 0.12)";
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "";
    dropzone.style.background = "";
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "";
    dropzone.style.background = "";
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  async function handleFileUpload(file) {
    uploadProgress.style.display = "flex";
    progressFill.style.width = "40%";
    uploadStatusText.textContent = `Memuat berkas: ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Gagal mengunggah");
      const data = await res.json();

      progressFill.style.width = "100%";
      uploadStatusText.textContent = "Video siap! Memulai analisis...";
      if (lblMissionFile) {
        lblMissionFile.textContent = file.name;
      }

      setTimeout(() => {
        uploadProgress.style.display = "none";
        progressFill.style.width = "0%";
        btnRefresh.click();
      }, 1000);
    } catch (err) {
      uploadStatusText.textContent = "Gagal memuat video.";
      progressFill.style.background = "#f43f5e";
    }
  }

  // Stream Error Recovery
  window.handleStreamError = () => {
    videoFeed.style.display = "none";
    streamPlaceholder.style.display = "flex";
    setTimeout(() => {
      videoFeed.style.display = "block";
      streamPlaceholder.style.display = "none";
      btnRefresh.click();
    }, 2000);
  };
});
