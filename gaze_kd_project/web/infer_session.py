"""
Load a gaze checkpoint once and run single-image inference (used by the web server).

Must be imported with project root (``gaze_kd_project``) on ``sys.path``.
"""

from __future__ import annotations

import io
import os
from typing import Literal, Optional

import torch
from PIL import Image
from torchvision import transforms

from models.student_model import build_student
from models.teacher_model import build_teacher, resolve_teacher_arch
from utils import configure_training_runtime, load_checkpoint

ModelName = Literal["student", "teacher"]


def _build_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class GazeInferenceSession:
    """Loads weights at startup; thread-safe enough for single-process uvicorn."""

    def __init__(
        self,
        checkpoint_path: str,
        model_name: ModelName = "student",
        image_size: int = 224,
        device: Optional[str] = None,
        teacher_arch: str = "",
    ) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.image_size = image_size
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        configure_training_runtime(self.device)

        if model_name == "teacher":
            arch = resolve_teacher_arch(teacher_arch, checkpoint_path)
            self.model = build_teacher(pretrained=False, arch=arch).to(self.device)
        else:
            self.model = build_student(pretrained=False).to(self.device)

        load_checkpoint(checkpoint_path, self.model, optimizer=None, device=self.device)
        self.model.eval()
        self.preprocess = _build_preprocess(image_size)

    @torch.inference_mode()
    def predict_tensor(self, batch: torch.Tensor) -> tuple[float, float]:
        batch = batch.to(self.device, non_blocking=True)
        out = self.model(batch)[0]
        return float(out[0].item()), float(out[1].item())

    def predict_bytes(self, image_bytes: bytes) -> tuple[float, float]:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = self.preprocess(img).unsqueeze(0)
        return self.predict_tensor(x)

    def predict_pil(self, img: Image.Image) -> tuple[float, float]:
        x = self.preprocess(img.convert("RGB")).unsqueeze(0)
        return self.predict_tensor(x)
