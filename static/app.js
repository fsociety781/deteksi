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

  // Manual Target Selection Elements
  const selectManualTarget = document.getElementById("selectManualTarget");
  const btnReleaseTarget = document.getElementById("btnReleaseTarget");
  const btnUnlockTarget = document.getElementById("btnUnlockTarget");
  const lblTargetSelected = document.getElementById("lblTargetSelected");
  const streamWrapper = document.getElementById("streamWrapper");

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
        if (data.is_live) {
          lblMissionFile.innerHTML = `<span class="pulse-dot-red" style="vertical-align: middle; margin-right: 4px;"></span> ${data.video_name}`;
        } else {
          lblMissionFile.textContent = data.video_name;
        }
      }

      // Update Primary Target Focus Card
      const locked = data.locked_target;
      if (locked && locked.id !== null) {
        const isManual = data.is_manual_lock;
        lockBadge.textContent = isManual ? "KUNCI USER" : "OTOMATIS";
        lockBadge.style.color = isManual ? "#10b981" : "#06b6d4";
        tValId.textContent = `#${locked.id}`;
        tValClass.textContent = locked.class || "OBJECT";
        tValConf.textContent = locked.conf || "--";
        tValSpeed.textContent = locked.speed_kmh || "0.0 km/h";
        tValBearing.textContent = locked.bearing || "--";
        tValCoords.textContent = locked.coords || "--";
        lblLockStatus.textContent = isManual ? `TARGET KUNCI USER: #${locked.id}` : `MENGIKUTI OBJEK BERGERAK: #${locked.id}`;
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
        lockBadge.style.color = "#06b6d4";
        tValId.textContent = "MEMINDAI";
        tValClass.textContent = "--";
        tValConf.textContent = "--";
        tValSpeed.textContent = "0.0 km/h";
        tValBearing.textContent = "--";
        tValCoords.textContent = "--";
        lblLockStatus.textContent = "MEMINDAI OBJEK BERGERAK...";

        if (valMainSpeed) {
          valMainSpeed.style.color = "#10b981";
          valMainSpeed.innerHTML = `0.0 <small style="font-size: 1rem; color: var(--text-dark); font-weight: 500;">km/h</small>`;
        }
        if (valSpeedSubtext) {
          valSpeedSubtext.textContent = "0.0 m/s | 0 px/s";
        }
      }

      // Update Target Selector Dropdown
      if (selectManualTarget) {
        const activeTargets = data.active_targets || [];
        const isUserInteracting = (document.activeElement === selectManualTarget);

        if (!isUserInteracting) {
          let html = `<option value="">-- Mode Otomatis (Objek Bergerak) --</option>`;
          activeTargets.forEach((t) => {
            const isSel = (t.id === data.locked_target_id && data.is_manual_lock);
            html += `<option value="${t.id}" ${isSel ? "selected" : ""}>#${t.id} - ${t.class} (${t.speed_kmh} km/h)</option>`;
          });
          selectManualTarget.innerHTML = html;
        }
      }

      // Update Target Lock Status & Unlock Button
      if (data.is_manual_lock && data.locked_target_id !== null) {
        if (lblTargetSelected) {
          lblTargetSelected.textContent = `TERKUNCI #${data.locked_target_id}`;
          lblTargetSelected.style.color = "#10b981";
        }
        if (btnUnlockTarget) {
          btnUnlockTarget.style.display = "inline-block";
          btnUnlockTarget.textContent = "🔄 KEMBALI KE OTOMATIS";
        }
      } else {
        if (lblTargetSelected) {
          lblTargetSelected.textContent = data.locked_target_id ? `OTOMATIS #${data.locked_target_id}` : "OTOMATIS";
          lblTargetSelected.style.color = "#06b6d4";
        }
        if (btnUnlockTarget) btnUnlockTarget.style.display = "none";
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

  const chkHighSpeed = document.getElementById("chkHighSpeed");
  if (chkHighSpeed) {
    chkHighSpeed.addEventListener("change", (e) => {
      updateSettings({ high_speed_mode: e.target.checked });
    });
  }

  const selectImgsz = document.getElementById("selectImgsz");
  if (selectImgsz) {
    selectImgsz.addEventListener("change", (e) => {
      updateSettings({ imgsz: parseInt(e.target.value, 10) });
    });
  }

  // Interactive Target Selection: Click directly on Video Feed
  function showClickRipple(x, y) {
    if (!streamWrapper) return;
    const ripple = document.createElement("div");
    ripple.className = "click-reticle-ripple";
    ripple.style.left = `${x}px`;
    ripple.style.top = `${y}px`;
    streamWrapper.appendChild(ripple);
    setTimeout(() => ripple.remove(), 650);
  }

  videoFeed.addEventListener("click", async (e) => {
    const rect = videoFeed.getBoundingClientRect();

    // Account for object-fit: contain letterbox/pillarbox
    const naturalWidth = videoFeed.naturalWidth || 640;
    const naturalHeight = videoFeed.naturalHeight || 480;
    const elemWidth = rect.width;
    const elemHeight = rect.height;

    const imgRatio = naturalWidth / naturalHeight;
    const elemRatio = elemWidth / elemHeight;

    let renderWidth, renderHeight, offsetX, offsetY;
    if (elemRatio > imgRatio) {
      renderHeight = elemHeight;
      renderWidth = elemHeight * imgRatio;
      offsetX = (elemWidth - renderWidth) / 2;
      offsetY = 0;
    } else {
      renderWidth = elemWidth;
      renderHeight = elemWidth / imgRatio;
      offsetX = 0;
      offsetY = (elemHeight - renderHeight) / 2;
    }

    const clickX = e.clientX - rect.left - offsetX;
    const clickY = e.clientY - rect.top - offsetY;

    if (clickX < 0 || clickX > renderWidth || clickY < 0 || clickY > renderHeight) {
      return; // Click outside the video pixels
    }

    const normX = Math.max(0, Math.min(1, clickX / renderWidth));
    const normY = Math.max(0, Math.min(1, clickY / renderHeight));

    showClickRipple(e.clientX - rect.left, e.clientY - rect.top);

    try {
      const res = await fetch("/api/select_target", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ norm_x: normX, norm_y: normY }),
      });
      const data = await res.json();
      fetchStats();
    } catch (err) {
      console.error("Gagal mengunci target:", err);
    }
  });

  if (selectManualTarget) {
    selectManualTarget.addEventListener("change", async (e) => {
      const val = e.target.value;
      if (!val) {
        await fetch("/api/release_target", { method: "POST" });
      } else {
        await fetch("/api/select_target", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ track_id: parseInt(val) }),
        });
      }
      fetchStats();
    });
  }

  const unlockHandler = async () => {
    try {
      await fetch("/api/release_target", { method: "POST" });
      if (selectManualTarget) selectManualTarget.value = "";
      fetchStats();
    } catch (err) {
      console.error("Gagal melepas target:", err);
    }
  };

  if (btnReleaseTarget) btnReleaseTarget.addEventListener("click", unlockHandler);
  if (btnUnlockTarget) btnUnlockTarget.addEventListener("click", unlockHandler);

  // 9. Video Source Switcher Handlers
  const btnSourceLive = document.getElementById("btnSourceLive");
  const btnSourceSample = document.getElementById("btnSourceSample");
  const btnSourceUpload = document.getElementById("btnSourceUpload");
  const uploadPanelContainer = document.getElementById("uploadPanelContainer");
  const btnConnectStream = document.getElementById("btnConnectStream");
  const txtCustomStreamUrl = document.getElementById("txtCustomStreamUrl");
  const selectBandungCctv = document.getElementById("selectBandungCctv");
  const groupBandungCctv = document.getElementById("groupBandungCctv");

  async function switchSource(sourceType, url = null) {
    try {
      if (lblMissionFile) {
        lblMissionFile.textContent = "Menghubungkan ke sumber stream...";
      }
      const res = await fetch("/api/set_source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_type: sourceType, url: url }),
      });
      const data = await res.json();
      if (data.status === "ok") {
        if (lblMissionFile) {
          lblMissionFile.textContent = data.video_name;
        }
        const baseSrc = videoFeed.src.split("?")[0];
        videoFeed.src = `${baseSrc}?t=${Date.now()}`;
        fetchStats();
      } else {
        alert("Gagal menghubungkan ke feed video.");
      }
    } catch (e) {
      console.error("Gagal mengganti sumber:", e);
    }
  }

  // Handle Bandung CCTV Presets selection
  if (selectBandungCctv) {
    selectBandungCctv.addEventListener("change", (e) => {
      const url = e.target.value;
      if (txtCustomStreamUrl) txtCustomStreamUrl.value = url;
      document.querySelectorAll(".source-pill-btn").forEach((b) => b.classList.remove("active", "live-active"));
      if (btnSourceLive) btnSourceLive.classList.add("active", "live-active");
      if (uploadPanelContainer) uploadPanelContainer.style.display = "none";
      switchSource("live", url);
    });
  }

  if (btnSourceLive) {
    btnSourceLive.addEventListener("click", () => {
      document.querySelectorAll(".source-pill-btn").forEach((b) => b.classList.remove("active", "live-active"));
      btnSourceLive.classList.add("active", "live-active");
      if (uploadPanelContainer) uploadPanelContainer.style.display = "none";
      if (groupBandungCctv) groupBandungCctv.style.display = "block";
      const liveUrl = selectBandungCctv ? selectBandungCctv.value : (txtCustomStreamUrl ? txtCustomStreamUrl.value.trim() : "https://atcs-dishub.bandung.go.id:1990/MochToha/index.m3u8");
      switchSource("live", liveUrl);
    });
  }

  if (btnSourceSample) {
    btnSourceSample.addEventListener("click", () => {
      document.querySelectorAll(".source-pill-btn").forEach((b) => b.classList.remove("active", "live-active"));
      btnSourceSample.classList.add("active");
      if (uploadPanelContainer) uploadPanelContainer.style.display = "none";
      switchSource("sample");
    });
  }

  if (btnSourceUpload) {
    btnSourceUpload.addEventListener("click", () => {
      document.querySelectorAll(".source-pill-btn").forEach((b) => b.classList.remove("active", "live-active"));
      btnSourceUpload.classList.add("active");
      if (uploadPanelContainer) {
        uploadPanelContainer.style.display = (uploadPanelContainer.style.display === "none") ? "block" : "none";
      }
    });
  }

  if (btnConnectStream && txtCustomStreamUrl) {
    btnConnectStream.addEventListener("click", () => {
      const url = txtCustomStreamUrl.value.trim();
      if (!url) return;
      document.querySelectorAll(".source-pill-btn").forEach((b) => b.classList.remove("active", "live-active"));
      if (btnSourceLive) btnSourceLive.classList.add("active", "live-active");
      switchSource("custom_url", url);
    });
  }

  // 10. Video Upload (Drag & Drop)
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
    if (!file) return;

    uploadProgress.style.display = "flex";
    progressFill.style.background = "#06b6d4";
    progressFill.style.width = "40%";
    uploadStatusText.textContent = `Mengunggah berkas: ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok || data.status !== "ok") {
        throw new Error(data.message || "Gagal memproses berkas video");
      }

      progressFill.style.width = "100%";
      uploadStatusText.textContent = "Video siap! Menjalankan stream...";
      if (lblMissionFile) {
        lblMissionFile.textContent = file.name;
      }

      setTimeout(() => {
        uploadProgress.style.display = "none";
        progressFill.style.width = "0%";
        videoFileInput.value = "";
        const baseSrc = videoFeed.src.split("?")[0];
        videoFeed.src = `${baseSrc}?t=${Date.now()}`;
        fetchStats();
      }, 800);
    } catch (err) {
      console.error("Upload error:", err);
      uploadStatusText.textContent = `Gagal: ${err.message || "Gagal memuat video"}`;
      progressFill.style.background = "#f43f5e";
      progressFill.style.width = "100%";
      setTimeout(() => {
        uploadProgress.style.display = "none";
        progressFill.style.background = "#06b6d4";
        progressFill.style.width = "0%";
        videoFileInput.value = "";
      }, 5000);
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
