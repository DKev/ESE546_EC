# Lightweight Gaze Estimation + Knowledge Distillation (PyTorch)

Course-style project: **2D gaze regression** from face images using a **ResNet18 teacher**, a **MobileNetV3-Small student**, and **knowledge distillation** (MSE to teacher + MSE to labels). The goal is to compare **accuracy**, **model size**, and **inference speed**, showing that distillation can improve a small model for mobile-style deployment.

**中文分步教程（从零到训练 / 评估 / 网页）：** [docs/GETTING_STARTED_ZH.md](docs/GETTING_STARTED_ZH.md)

## Requirements

- Python 3.10+ recommended
- PyTorch + CUDA (optional but recommended, e.g. RTX 3060 Ti)

Install dependencies:

```bash
cd gaze_kd_project
pip install -r requirements.txt
```

## Project layout

| Path | Description |
|------|-------------|
| `config.py` | Default hyperparameters and checkpoint paths |
| `utils.py` | Seeding, train/val/KD loops, checkpoints, latency, metrics |
| `datasets/gaze_dataset.py` | CSV-based `Dataset` (image + gaze x, y) |
| `models/teacher_model.py` | ResNet18 → 2 outputs |
| `models/student_model.py` | MobileNetV3-Small → 2 outputs |
| `train_teacher.py` | Supervised teacher training (MSE) |
| `train_student.py` | Supervised student baseline (MSE) |
| `train_kd.py` | Distilled student: `MSE(s,y) + alpha * MSE(s, teacher)` |
| `evaluate.py` | MSE, MAE, mean L2, params, checkpoint size, latency / FPS |
| `scripts/generate_synthetic_gaze_dataset.py` | Quick synthetic CSV + images for debugging |
| `scripts/make_paper_figures.py` | Build PDF plots for the written report |
| `scripts/build_eval_summary.py` | Merge several `evaluate.py` JSON exports |
| `docs/data_sources.md` | Where to get MPIIGaze / how it maps to this CSV format |
| `paper/report.tex` | AMS-style 5-page report template (fill table with your numbers) |
| `web/server.py` | 本机浏览器眼动演示（人脸检测裁剪 + 注视点叠加） |
| `web/face_crop.py` | OpenCV Haar 正脸检测与裁剪 |

## Web 眼动演示（浏览器）

在 `gaze_kd_project` 目录下启动服务（需已安装 `requirements.txt` 中的 FastAPI / Uvicorn）：

```bash
cd gaze_kd_project
# 使用你训练好的 student 或 KD student 权重：
export GAZE_CKPT=checkpoints/student_kd_best.pt
export GAZE_MODEL=student   # 或 teacher
python -m uvicorn web.server:app --host 127.0.0.1 --port 8765
```

浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)：允许摄像头权限后，可「预测一帧」或勾选「连续预测」；也可用上传图片测试。默认会先用 **OpenCV Haar 正脸检测** 框出最大人脸、按比例放大裁剪后再送入 gaze 模型；页面上用**绿色虚线**画检测框，蓝点为注视点映射。

**人脸相关环境变量（可选）：**

```bash
export GAZE_FACE_CROP=1          # 默认 1；设为 0 关闭检测，整图推理
export GAZE_FACE_EXPAND=1.35     # 框扩大倍数，略大于 1 可带上额头/下巴
```

**仅测试 UI、不加载模型：**

```bash
export GAZE_WEB_DEMO=1
python -m uvicorn web.server:app --host 127.0.0.1 --port 8765
```

此时 `/predict` 仍会对上传图做人脸检测（便于看绿框），gaze 数值为平滑假轨迹。

**说明：** Haar 对侧脸、强背光较弱；若未检测到正脸，会自动退回**整图**推理并在页面上提示。若你需要更高鲁棒性，可日后换成 MediaPipe / RetinaFace 等（需额外依赖）。

## Data: synthetic vs. real benchmarks

- **Fast path (no download):** see [docs/data_sources.md](docs/data_sources.md) and run:

  ```bash
  python scripts/generate_synthetic_gaze_dataset.py --out_root data/synthetic
  ```

  Then point `--data_root`, `--train_csv`, and `--val_csv` at `data/synthetic/`.

- **MPIIGaze (paper-quality):** download from MPI-INF or DaRUS (see the same doc), preprocess to face crops, then export `image_path,gaze_x,gaze_y` compatible with `GazeDataset`.

## Paper, plots, and LaTeX

The course template you have locally can be compared with [paper/report.tex](paper/report.tex) (already filled with structure, text, and figure includes).

1. **Train with metric logs** (optional but recommended):

   ```bash
   python train_teacher.py ... --metrics_csv runs/m_teacher.csv
   python train_student.py ... --metrics_csv runs/m_student.csv
   python train_kd.py ... --metrics_csv runs/m_kd.csv
   ```

2. **Export evaluation JSON** after you have checkpoints:

   ```bash
   python evaluate.py --model teacher --checkpoint checkpoints/teacher_best.pt --csv data/synthetic/val.csv --data_root data/synthetic --export_json runs/eval_teacher.json
   python evaluate.py --model student --checkpoint checkpoints/student_baseline_best.pt --csv data/synthetic/val.csv --data_root data/synthetic --export_json runs/eval_student.json
   python evaluate.py --model student --checkpoint checkpoints/student_kd_best.pt --csv data/synthetic/val.csv --data_root data/synthetic --export_json runs/eval_kd.json
   ```

   Optional scatter figure: add `--save_predictions runs/student_kd_val.npz` on the last command.

3. **Merge JSON and build figures:**

   ```bash
   python scripts/build_eval_summary.py --out runs/summary.json \
     --teacher runs/eval_teacher.json \
     --student_baseline runs/eval_student.json \
     --student_kd runs/eval_kd.json

   python scripts/make_paper_figures.py --out_dir paper/figures \
     --summary runs/summary.json \
     --metrics_teacher runs/m_teacher.csv \
     --metrics_student runs/m_student.csv \
     --metrics_kd runs/m_kd.csv \
     --scatter_npz runs/student_kd_val.npz
   ```

   If you have not trained yet, generate **placeholder** PDFs with `python scripts/make_paper_figures.py --demo --out_dir paper/figures`.

4. **Compile the PDF** (needs a LaTeX install such as MacTeX / TeX Live):

   ```bash
   cd paper && pdflatex report.tex && pdflatex report.tex
   ```

## CSV format

Create **separate** CSV files for training and validation (e.g. `data/train.csv`, `data/val.csv`).

Header (required):

```text
image_path,gaze_x,gaze_y
```

- `image_path`: path to an RGB image (absolute, or relative to `--data_root`)
- `gaze_x`, `gaze_y`: regression targets (floats; use the same convention for all splits)

### Tiny example CSV

```csv
image_path,gaze_x,gaze_y
samples/face_001.jpg,0.12,-0.05
samples/face_002.jpg,-0.03,0.21
data/person_a/003.png,0.0,0.0
```

Place images accordingly (e.g. under `data/samples/` if you use `data/` as `--data_root`).

**Adapting to MPIIGaze / GazeCapture:** keep this CSV interface; write a small script that exports `(path, gaze_x, gaze_y)` from those datasets’ native labels into this format.

## Training

Run all commands from `gaze_kd_project/` unless you adjust `PYTHONPATH`.

### 1) Teacher (ResNet18)

```bash
python train_teacher.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --data_root data \
  --checkpoint checkpoints/teacher_best.pt
```

### 2) Student baseline (no KD)

```bash
python train_student.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --data_root data \
  --checkpoint checkpoints/student_baseline_best.pt
```

### 3) Student + knowledge distillation

Requires a trained teacher checkpoint.

```bash
python train_kd.py \
  --teacher_ckpt checkpoints/teacher_best.pt \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --data_root data \
  --checkpoint checkpoints/student_kd_best.pt \
  --alpha 0.5
```

**Loss:** `total = MSE(student, ground_truth) + alpha * MSE(student, teacher_prediction)`. The teacher is frozen and in eval mode.

## Evaluation

Use the same `--data_root` and `--image_size` as in training when comparing fairly.

**Teacher:**

```bash
python evaluate.py --model teacher --checkpoint checkpoints/teacher_best.pt \
  --csv data/val.csv --data_root data
```

**Student (baseline or KD checkpoint):**

```bash
python evaluate.py --model student --checkpoint checkpoints/student_baseline_best.pt \
  --csv data/val.csv --data_root data

python evaluate.py --model student --checkpoint checkpoints/student_kd_best.pt \
  --csv data/val.csv --data_root data
```

Reported metrics:

- **MSE:** mean over samples of \(\| \hat{g} - g \|_2^2\) (sum of squared errors on x and y)
- **MAE:** mean absolute error averaged over both coordinates
- **Mean L2:** mean Euclidean distance between predicted and true 2D gaze
- **Parameters**, **checkpoint file size**, **latency** (dummy tensor, same spatial size), **approx FPS**

## Default hyperparameters (see `config.py`)

- Batch size: 64  
- Learning rate: 1e-4 (Adam)  
- Epochs: 20  
- Image size: 224  
- KD `alpha`: 0.5  

Override with CLI flags on any script.

## Suggested run order

1. Prepare `train.csv` / `val.csv` and images.  
2. `train_teacher.py`  
3. `train_student.py`  
4. `train_kd.py` (needs teacher checkpoint)  
5. `evaluate.py` on the same val/test CSV for teacher, student baseline, and KD student.

## Results table for your report

| Model | MSE | MAE | Mean L2 | Params | CKPT (MB) | ms/img | FPS |
|-------|-----|-----|---------|--------|-----------|--------|-----|
| Teacher (ResNet18) | | | | | | | |
| Student baseline | | | | | | | |
| Student + KD | | | | | | | |

You typically want: **KD student better than baseline** on error metrics, while **student stays smaller/faster** than the teacher.

## Future improvements

- **Angular error** (standard in gaze papers): needs camera/head pose to convert 2D/3D vectors; add a metric module once you fix your label convention.  
- **Data augmentation**: random crops, color jitter, slight blur — watch label consistency if gaze is defined in image space.  
- **Feature distillation**: match intermediate feature maps, not only outputs.  
- **Deployment**: export student to **ONNX**, optimize with **TensorRT** or mobile runtimes (Core ML / TFLite) for on-device FPS.

## License / course use

Code is intended for educational use; swap in your dataset and cite MPIIGaze / GazeCapture if you use them.
