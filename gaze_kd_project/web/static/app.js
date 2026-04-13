/**
 * Webcam capture -> POST /predict (multipart) -> draw gaze dot on overlay canvas.
 */
const cam = document.getElementById("cam");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const btnStart = document.getElementById("btnStart");
const btnPredict = document.getElementById("btnPredict");
const chkLoop = document.getElementById("chkLoop");
const radiusEl = document.getElementById("radius");
const radiusVal = document.getElementById("radiusVal");
const statusEl = document.getElementById("status");
const coordsEl = document.getElementById("coords");
const fileInput = document.getElementById("file");
const uploadCanvas = document.getElementById("uploadCanvas");
const uctx = uploadCanvas.getContext("2d");
const btnUploadPredict = document.getElementById("btnUploadPredict");

let loopId = null;
const capCanvas = document.createElement("canvas");
const capCtx = capCanvas.getContext("2d");

/** @type {{ img: HTMLImageElement; ox: number; oy: number; dw: number; dh: number } | null} */
let uploadLayout = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function getRadius() {
  return Number(radiusEl.value);
}

/** @returns {null | "crop" | "original"} */
function getFaceCropFormValue() {
  const el = document.querySelector('input[name="faceCrop"]:checked');
  if (!el || el.value === "default") return null;
  return el.value === "crop" ? "crop" : "original";
}

radiusEl.addEventListener("input", () => {
  radiusVal.textContent = radiusEl.value;
});

function resizeOverlayToVideo() {
  const w = cam.videoWidth || 640;
  const h = cam.videoHeight || 480;
  if (w && h) {
    overlay.width = w;
    overlay.height = h;
  }
}

function faceStatusText(data) {
  if (data.face_crop_enabled === false) return "人脸: 整图推理（未裁剪）";
  if (!("face_crop_enabled" in data)) return "";
  if (data.face_detected === true) return "人脸: 已检测并裁剪";
  return "人脸: 未检测到正脸（已用整图推理）";
}

function formatCoordsLine(data) {
  const g = `gaze: (${data.gaze_x.toFixed(4)}, ${data.gaze_y.toFixed(4)})`;
  const tag = data.demo ? " [demo]" : "";
  const face = faceStatusText(data);
  if (!face) return `${g}${tag}`;
  return `${g}  |  ${face}${tag}`;
}

function drawFaceBbox(ctx2d, bbox) {
  if (!bbox || bbox.w <= 0 || bbox.h <= 0) return;
  ctx2d.save();
  ctx2d.strokeStyle = "rgba(46, 204, 113, 0.95)";
  ctx2d.lineWidth = 3;
  ctx2d.setLineDash([8, 6]);
  ctx2d.strokeRect(bbox.x + 0.5, bbox.y + 0.5, bbox.w - 1, bbox.h - 1);
  ctx2d.restore();
}

function drawGazeDot(ctx2d, w, h, gx, gy, radius) {
  const cx = w / 2 + gx * radius;
  const cy = h / 2 - gy * radius;
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, 12, 0, Math.PI * 2);
  ctx2d.fillStyle = "rgba(29, 155, 240, 0.85)";
  ctx2d.fill();
  ctx2d.strokeStyle = "rgba(255,255,255,0.9)";
  ctx2d.lineWidth = 3;
  ctx2d.stroke();
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, 22, 0, Math.PI * 2);
  ctx2d.strokeStyle = "rgba(29, 155, 240, 0.35)";
  ctx2d.lineWidth = 4;
  ctx2d.stroke();
}

function redrawUploadCanvas() {
  if (!uploadLayout) return;
  const w = uploadCanvas.width;
  const h = uploadCanvas.height;
  const { img, ox, oy, dw, dh } = uploadLayout;
  uctx.fillStyle = "#000";
  uctx.fillRect(0, 0, w, h);
  uctx.drawImage(img, ox, oy, dw, dh);
}

async function predictBlob(blob) {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  const mode = getFaceCropFormValue();
  if (mode) fd.append("face_crop", mode);
  const res = await fetch("/predict", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function predictFromVideo() {
  resizeOverlayToVideo();
  const w = cam.videoWidth;
  const h = cam.videoHeight;
  if (!w || !h) {
    setStatus("摄像头尚未就绪。");
    return;
  }
  capCanvas.width = w;
  capCanvas.height = h;
  capCtx.drawImage(cam, 0, 0, w, h);
  const blob = await new Promise((resolve) =>
    capCanvas.toBlob((b) => resolve(b), "image/jpeg", 0.85)
  );
  const data = await predictBlob(blob);
  coordsEl.textContent = formatCoordsLine(data);
  ctx.clearRect(0, 0, w, h);
  drawFaceBbox(ctx, data.face_bbox);
  drawGazeDot(ctx, w, h, data.gaze_x, data.gaze_y, getRadius());
}

btnStart.addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    cam.srcObject = stream;
    await cam.play();
    btnPredict.disabled = false;
    setStatus("摄像头已开启。点击「预测一帧」或勾选连续预测。");
    cam.addEventListener(
      "loadedmetadata",
      () => {
        resizeOverlayToVideo();
      },
      { once: true }
    );
  } catch (e) {
    setStatus("无法打开摄像头: " + e.message);
  }
});

btnPredict.addEventListener("click", async () => {
  try {
    await predictFromVideo();
  } catch (e) {
    setStatus("预测失败: " + e.message);
  }
});

chkLoop.addEventListener("change", () => {
  if (loopId) {
    clearInterval(loopId);
    loopId = null;
  }
  if (chkLoop.checked) {
    loopId = setInterval(() => {
      predictFromVideo().catch((e) => setStatus("连续预测: " + e.message));
    }, 250);
  }
});

fileInput.addEventListener("change", () => {
  const f = fileInput.files && fileInput.files[0];
  if (!f) {
    btnUploadPredict.disabled = true;
    return;
  }
  const img = new Image();
  img.onload = () => {
    const w = 640;
    const h = 480;
    uploadCanvas.width = w;
    uploadCanvas.height = h;
    const scale = Math.min(w / img.width, h / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    const ox = (w - dw) / 2;
    const oy = (h - dh) / 2;
    uctx.fillStyle = "#000";
    uctx.fillRect(0, 0, w, h);
    uctx.drawImage(img, ox, oy, dw, dh);
    uploadLayout = { img, ox, oy, dw, dh };
    btnUploadPredict.disabled = false;
  };
  img.src = URL.createObjectURL(f);
});

btnUploadPredict.addEventListener("click", async () => {
  const blob = await new Promise((resolve) =>
    uploadCanvas.toBlob((b) => resolve(b), "image/jpeg", 0.92)
  );
  try {
    const data = await predictBlob(blob);
    coordsEl.textContent = formatCoordsLine(data);
    const w = uploadCanvas.width;
    const h = uploadCanvas.height;
    redrawUploadCanvas();
    drawFaceBbox(uctx, data.face_bbox);
    drawGazeDot(uctx, w, h, data.gaze_x, data.gaze_y, getRadius());
  } catch (e) {
    setStatus("上传预测失败: " + e.message);
  }
});

fetch("/health")
  .then((r) => r.json())
  .then((j) => {
    let s = "";
    if (j.demo_mode) s = "服务器运行在 DEMO 模式（无真实模型）。";
    if (j.face_crop_enabled === false) {
      s += (s ? " " : "") + "服务器默认：整图（GAZE_FACE_CROP=0）；页面仍可选「人脸裁剪」覆盖。";
    }
    if (s) setStatus(s);
  })
  .catch(() => {});
