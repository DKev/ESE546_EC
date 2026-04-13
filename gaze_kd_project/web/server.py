"""
Local web server for webcam / upload gaze demo.

Run from the ``gaze_kd_project`` directory::

    pip install fastapi uvicorn python-multipart
    export GAZE_CKPT=checkpoints/student_kd_best.pt
    python -m uvicorn web.server:app --host 127.0.0.1 --port 8765

Open http://127.0.0.1:8765/

Environment variables:
    GAZE_CKPT       Path to .pt checkpoint (required unless GAZE_WEB_DEMO=1)
    GAZE_MODEL      ``student`` or ``teacher`` (default: student)
    GAZE_WEB_DEMO   If ``1``, return fake smooth gaze (no model, UI test only)
    GAZE_DEVICE     Optional: cuda, cuda:0, cpu
    GAZE_FACE_CROP  If ``1`` (default), run Haar face detection and crop before inference
    GAZE_FACE_EXPAND  Multiplier on face box width/height (default: 1.35)

Each ``POST /predict`` multipart form may include optional ``face_crop``:
    omit / empty  use ``GAZE_FACE_CROP`` env default
    crop, 1, true, on   detect face and crop (same as env on)
    original, full, 0, false, off   use full frame, no detection
"""

from __future__ import annotations

import io
import math
import os
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.face_crop import crop_largest_face
from web.infer_session import GazeInferenceSession

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Gaze demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: GazeInferenceSession | None = None
_demo_t0: float | None = None


def _face_crop_enabled() -> bool:
    v = os.environ.get("GAZE_FACE_CROP", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def _parse_face_crop_form(value: Optional[str]) -> Optional[bool]:
    """
    None / empty -> use env default.
    Otherwise return True (crop) or False (full frame).
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    v = s.lower()
    if v in ("crop", "1", "true", "yes", "on"):
        return True
    if v in ("original", "full", "0", "false", "no", "off"):
        return False
    raise HTTPException(
        status_code=400,
        detail="Invalid face_crop; use crop|original (or 1|0), or omit for server default.",
    )


def _prepare_for_gaze(pil_rgb: Image.Image, *, use_face_crop: bool) -> tuple[Image.Image, dict]:
    """
    Optionally crop to the largest detected face. Always returns an RGB PIL image
    for the gaze model and metadata for the client overlay.
    """
    if not use_face_crop:
        return pil_rgb.convert("RGB"), {
            "face_crop_enabled": False,
            "face_detected": None,
            "face_bbox": None,
        }
    try:
        expand = float(os.environ.get("GAZE_FACE_EXPAND", "1.35"))
    except ValueError:
        expand = 1.35
    cropped, found, bbox = crop_largest_face(pil_rgb, expand=expand)
    box = None
    if bbox is not None:
        box = {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])}
    return cropped, {
        "face_crop_enabled": True,
        "face_detected": found,
        "face_bbox": box,
    }


def _get_session() -> GazeInferenceSession:
    global _session
    if _session is None:
        ckpt = os.environ.get("GAZE_CKPT", "").strip()
        if not ckpt:
            raise HTTPException(
                status_code=503,
                detail="Set GAZE_CKPT to a checkpoint path, or GAZE_WEB_DEMO=1 for UI-only demo.",
            )
        model = os.environ.get("GAZE_MODEL", "student").strip().lower()
        if model not in ("student", "teacher"):
            raise HTTPException(status_code=500, detail="GAZE_MODEL must be student or teacher")
        dev = os.environ.get("GAZE_DEVICE", "").strip() or None
        _session = GazeInferenceSession(ckpt, model_name=model, device=dev)
    return _session


def _demo_gaze() -> tuple[float, float]:
    """Smooth Lissajous-like point in [-1, 1] for UI testing."""
    global _demo_t0
    if _demo_t0 is None:
        _demo_t0 = time.perf_counter()
    t = time.perf_counter() - _demo_t0
    gx = 0.75 * math.sin(t * 0.9)
    gy = 0.55 * math.sin(t * 0.7 + 1.2)
    return gx, gy


@app.get("/")
def index() -> FileResponse:
    path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(path, media_type="text/html")


@app.get("/health")
def health() -> dict:
    demo = os.environ.get("GAZE_WEB_DEMO", "").strip() == "1"
    return {
        "ok": True,
        "demo_mode": demo,
        "has_checkpoint": bool(os.environ.get("GAZE_CKPT", "").strip()) or demo,
        "face_crop_enabled": _face_crop_enabled(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    face_crop: Optional[str] = Form(None),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    try:
        pil_rgb = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    crop_flag = _parse_face_crop_form(face_crop)
    use_crop = _face_crop_enabled() if crop_flag is None else crop_flag
    model_img, face_meta = _prepare_for_gaze(pil_rgb, use_face_crop=use_crop)

    if os.environ.get("GAZE_WEB_DEMO", "").strip() == "1":
        gx, gy = _demo_gaze()
        return {"gaze_x": gx, "gaze_y": gy, "demo": True, **face_meta}

    try:
        sess = _get_session()
        gx, gy = sess.predict_pil(model_img)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    return {"gaze_x": gx, "gaze_y": gy, "demo": False, **face_meta}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
