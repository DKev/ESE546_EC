# Lightweight Gaze Estimation + Knowledge Distillation (PyTorch)

Course-style project: **2D gaze regression** from face images using a **teacher** (default in docs: **MobileNetV2**), a **student** (default: **MobileNetV3-Small**), and **knowledge distillation** (MSE to teacher + MSE to labels). You can use **`--teacher_arch mobilenet_v3_small`** if that backbone fits your data best, and smaller students: **`--student_arch shufflenet_v2_x0_5`** (~0.34M) or **`gaze_micro`** (~50k params, custom CNN, no ImageNet weights). The teacher is deliberately lighter than a classic ResNet so small datasets overfit less; you can still use **ResNet18** via `--teacher_arch resnet18`. The goal is to compare **accuracy**, **model size**, and **inference speed**, showing that distillation can improve the small student for mobile-style deployment.

**中文分步教程（从零到训练 / 评估 / 网页）：** [docs/GETTING_STARTED_ZH.md](docs/GETTING_STARTED_ZH.md)  
**真实数据从哪下、怎么变成 CSV：** [docs/data_sources.md](docs/data_sources.md) 开头的「真实数据怎么拿」  
**MPIIGaze 已下载后的下一步：** [docs/mpiigaze_next_steps.md](docs/mpiigaze_next_steps.md)  
**MPIIGaze 一条龙（下载 → 训练 → 评估 → 论文图）：** 见本文 [Real data workflow (MPIIGaze)](#real-data-workflow-mpiigaze)

## Requirements

- Python 3.10+ recommended
- PyTorch + CUDA (optional but recommended, e.g. RTX 3060 Ti)

Install dependencies:

```bash
cd gaze_kd_project
pip install -r requirements.txt
```

### GPU 上跑一通（命令备忘）

在新机器上请**先**从 [pytorch.org](https://pytorch.org) 安装带 **CUDA** 的 `torch` / `torchvision`，再 `pip install -r requirements.txt`（避免只装到 CPU 版）。`data/synthetic/` 与 `checkpoints/` 默认被 `.gitignore` 忽略，若仓库里没有数据，需要重新生成或拷贝整个 `data/synthetic`。

```bash
cd gaze_kd_project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) 按官网安装 CUDA 版 PyTorch 后：
pip install -r requirements.txt

# 2) 若无合成数据，先生成：
python scripts/generate_synthetic_gaze_dataset.py --out_root data/synthetic

# 3) 训练（默认 20 epochs；老师 MobileNetV2，学生 MobileNetV3-Small；可加 --metrics_csv 等）：
python train_teacher.py \
  --teacher_arch mobilenet_v2 \
  --train_csv data/synthetic/train.csv \
  --val_csv data/synthetic/val.csv \
  --data_root data/synthetic \
  --checkpoint checkpoints/teacher_best.pt \
  --epochs 20

python train_student.py \
  --train_csv data/synthetic/train.csv \
  --val_csv data/synthetic/val.csv \
  --data_root data/synthetic \
  --checkpoint checkpoints/student_baseline_best.pt \
  --epochs 20

python train_kd.py \
  --teacher_ckpt checkpoints/teacher_best.pt \
  --train_csv data/synthetic/train.csv \
  --val_csv data/synthetic/val.csv \
  --data_root data/synthetic \
  --checkpoint checkpoints/student_kd_best.pt \
  --epochs 20
```

DataLoader 若报错可尝试 `--num_workers 0`；显存不够可把 `--batch_size 32`。

使用 **MPIIGaze 真实数据** 的完整命令流见下文 **[Real data workflow (MPIIGaze)](#real-data-workflow-mpiigaze)**。

## Project layout

| Path | Description |
|------|-------------|
| `config.py` | Default hyperparameters and checkpoint paths |
| `utils.py` | Seeding, train/val/KD loops, checkpoints, latency, metrics |
| `datasets/gaze_dataset.py` | CSV-based `Dataset` (image + gaze x, y) |
| `datasets/mpiigaze_dataset.py` | MPIIGaze `Data/Normalized` `.mat` → tensors |
| `datasets/factory.py` | `--dataset csv` or `mpiigaze` wiring for train / eval |
| `models/teacher_model.py` | Teacher: **ResNet18**, **MobileNetV2**, or **MobileNetV3-Small** → 2 outputs |
| `models/student_model.py` | Student: **MV3-Small** (default), **ShuffleNetV2 x0.5**, or **gaze_micro** (~50k params) → 2 outputs |
| `train_teacher.py` | Supervised teacher training (MSE) |
| `train_student.py` | Supervised student baseline (MSE) |
| `train_kd.py` | Distilled student: `MSE(s,y) + alpha * MSE(s, teacher)` |
| `evaluate.py` | MSE, MAE, mean L2, params, checkpoint size, latency / FPS |
| `scripts/generate_synthetic_gaze_dataset.py` | Quick synthetic CSV + images for debugging |
| `scripts/inspect_mpiigaze_layout.py` | After downloading MPIIGaze: peek folder layout |
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
# 使用你训练好的 MobileNetV3-Small student / KD student 权重（或 MobileNetV2 teacher）：
export GAZE_CKPT=checkpoints/student_kd_best.pt
export GAZE_MODEL=student   # 或 teacher（架构默认按 ckpt 的 extra.args 推断）
# 仅当 ckpt 无元数据时需指定： export GAZE_TEACHER_ARCH=resnet18
# 若 ckpt 无 student_arch： export GAZE_STUDENT_ARCH=gaze_micro   # 或 shufflenet_v2_x0_5
python -m uvicorn web.server:app --host 127.0.0.1 --port 8765
```

浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)：允许摄像头权限后，可「预测一帧」或勾选「连续预测」；也可用上传图片测试。页面上可选 **服务器默认 / 人脸裁剪 / 原始整图**：裁剪模式会用 **OpenCV Haar** 检测正脸并放大框后送入模型，**绿色虚线**为检测框；原始整图则不做检测。蓝点为注视点映射。

`POST /predict` 的 multipart 表单可带可选字段 **`face_crop`**：`crop` 或 `original`（也可用 `1`/`0`、`true`/`false`）；省略时沿用环境变量 `GAZE_FACE_CROP`。

**人脸相关环境变量（可选）：**

```bash
export GAZE_FACE_CROP=1          # 默认 1；设为 0 时「服务器默认」为整图推理
export GAZE_FACE_EXPAND=1.35     # 框扩大倍数，略大于 1 可带上额头/下巴
```

**仅测试 UI、不加载模型：**

```bash
export GAZE_WEB_DEMO=1
python -m uvicorn web.server:app --host 127.0.0.1 --port 8765
```

此时 `/predict` 仍会按所选模式做人脸检测或整图处理（便于看绿框），gaze 数值为平滑假轨迹。

**说明：** Haar 对侧脸、强背光较弱；若未检测到正脸，会自动退回**整图**推理并在页面上提示。若你需要更高鲁棒性，可日后换成 MediaPipe / RetinaFace 等（需额外依赖）。

## Data: synthetic vs. real benchmarks

- **Fast path (no download):** see [docs/data_sources.md](docs/data_sources.md) and run:

  ```bash
  python scripts/generate_synthetic_gaze_dataset.py --out_root data/synthetic
  ```

  Then point `--data_root`, `--train_csv`, and `--val_csv` at `data/synthetic/`.

- **MPIIGaze (real data):** full train → eval → figures pipeline is in **[Real data workflow (MPIIGaze)](#real-data-workflow-mpiigaze)** below. You can use the built-in `--dataset mpiigaze` reader **or** export your own CSV + images (see [docs/mpiigaze_next_steps.md](docs/mpiigaze_next_steps.md), route B).

## Real data workflow (MPIIGaze)

This section is the **end-to-end checklist** for training and reporting on **MPIIGaze** with this repo. Defaults match the synthetic quickstart: **teacher = MobileNetV2**, **student = MobileNetV3-Small**; you can switch to **`--teacher_arch mobilenet_v3_small`** and **`--student_arch gaze_micro`** (or ShuffleNet) on the train scripts. Labels come from `Data/Normalized/pXX/dayYY.mat` (unit gaze vectors → yaw/pitch scaled to about `[-1, 1]`, same scale as the synthetic toy data). For folder layout and license links, see [docs/data_sources.md](docs/data_sources.md) and [docs/mpiigaze_next_steps.md](docs/mpiigaze_next_steps.md).

### 1) Download, layout, and gitignore

1. Download MPIIGaze from the sources listed in [docs/data_sources.md](docs/data_sources.md) and accept the dataset terms.  
2. After unpacking, you should have **`<MPIIGAZE_ROOT>/Data/Normalized/pXX/dayYY.mat`** (and usually `Data/Original/`, etc.).  
3. Keep the dataset **outside** git-tracked project folders or add the folder name to your **repo root** `.gitignore` (e.g. `MPIIGaze/`) so multi‑GB files are never committed.  
4. `pip install -r requirements.txt` installs **SciPy** (required to read `.mat`).

Optional layout check from the repo root:

```bash
python scripts/inspect_mpiigaze_layout.py /path/to/MPIIGaze
```

### 2) Train / val split flags

- **`--mpi_val_persons 14,15`** → participants **`p14` and `p15`** are **validation**; **all other `pXX` folders** under `Data/Normalized` are **training**. Change the list as needed (comma-separated ids, no `p` prefix).  
- **`--mpi_max_train_samples N`** (optional) → cap **training** split at `N` samples only; validation stays **full** unless you also set **`--mpi_max_val_samples`**. Use this for fast epochs (e.g. **`--mpi_max_train_samples 10000`**) when the full train set is huge. Subsample order is deterministic (sorted participants → days → frames → left/right).
- **`--mpi_max_val_samples N`** (optional) → cap **validation** split only.
- **`--mpi_max_samples N`** (optional) → cap **both** train and val at `N` each (ignored for a split if the split-specific cap above is set).

Run all training commands from **`gaze_kd_project/`**. Set the dataset root and validation person ids as shown (dataset folder next to `gaze_kd_project` → `..\MPIIGaze` on Windows, `../MPIIGaze` on Unix).

**macOS / Linux**

```bash
export MPI_ROOT=../MPIIGaze
export MPI_VAL=14,15
```

**Windows (Command Prompt)**

```bat
set MPI_ROOT=..\MPIIGaze && set MPI_VAL=14,15
```

**Windows (PowerShell)**

```powershell
$env:MPI_ROOT="..\MPIIGaze"; $env:MPI_VAL="14,15"
```

### 3) Training (teacher, student, KD)

Examples below use **`--mpi_max_train_samples 10000`** so each epoch stays fast; drop that flag to train on the full MPII train split. Validation on `p14`/`p15` stays complete unless you add **`--mpi_max_val_samples`**.

Also included: **`--teacher_arch`** on **`train_teacher.py`** (documented default **`mobilenet_v2`**); **`--student_arch`** on **`train_student.py`** / **`train_kd.py`** (default **`mobilenet_v3_small`**). Plus **`--amp`** (CUDA mixed precision; omit on CPU-only), **`--num_workers 0`** (recommended on **Windows** to avoid duplicating `.mat` preload RAM; on Linux you can try **`--num_workers 4`**). Remove **`--amp`** if you do not have a GPU.

**macOS / Linux**

```bash
python train_teacher.py --dataset mpiigaze --mpi_root "$MPI_ROOT" --mpi_val_persons "$MPI_VAL" \
  --teacher_arch mobilenet_v2 \
  --mpi_max_train_samples 10000 --amp --num_workers 4 \
  --checkpoint checkpoints/teacher_mpi.pt --epochs 20 \
  --metrics_csv runs/m_teacher_mpi.csv

python train_student.py --dataset mpiigaze --mpi_root "$MPI_ROOT" --mpi_val_persons "$MPI_VAL" \
  --mpi_max_train_samples 10000 --amp --num_workers 4 \
  --checkpoint checkpoints/student_baseline_mpi.pt --epochs 20 \
  --metrics_csv runs/m_student_mpi.csv

python train_kd.py --dataset mpiigaze --mpi_root "$MPI_ROOT" --mpi_val_persons "$MPI_VAL" \
  --mpi_max_train_samples 10000 --amp --num_workers 4 \
  --teacher_ckpt checkpoints/teacher_mpi.pt \
  --checkpoint checkpoints/student_kd_mpi.pt --epochs 20 \
  --metrics_csv runs/m_kd_mpi.csv
```

**Windows (Command Prompt)** — use `%MPI_ROOT%` / `%MPI_VAL%` (set them in §2); each command is **one line**:

```bat
python train_teacher.py --dataset mpiigaze --mpi_root %MPI_ROOT% --mpi_val_persons %MPI_VAL% --teacher_arch mobilenet_v2 --mpi_max_train_samples 10000 --amp --num_workers 0 --checkpoint checkpoints\teacher_mpi.pt --epochs 20 --metrics_csv runs\m_teacher_mpi.csv

python train_student.py --dataset mpiigaze --mpi_root %MPI_ROOT% --mpi_val_persons %MPI_VAL% --mpi_max_train_samples 10000 --amp --num_workers 0 --checkpoint checkpoints\student_baseline_mpi.pt --epochs 20 --metrics_csv runs\m_student_mpi.csv

python train_kd.py --dataset mpiigaze --mpi_root %MPI_ROOT% --mpi_val_persons %MPI_VAL% --mpi_max_train_samples 10000 --amp --num_workers 0 --teacher_ckpt checkpoints\teacher_mpi.pt --checkpoint checkpoints\student_kd_mpi.pt --epochs 20 --metrics_csv runs\m_kd_mpi.csv
```

**Windows (PowerShell)** — use `$env:MPI_ROOT` / `$env:MPI_VAL`; **one line** per command. This block uses **teacher** `mobilenet_v3_small` and **student** `gaze_micro` (~50k params):

```powershell
python train_teacher.py --dataset mpiigaze --mpi_root $env:MPI_ROOT --mpi_val_persons $env:MPI_VAL --teacher_arch mobilenet_v3_small --mpi_max_train_samples 10000 --amp --num_workers 0 --checkpoint checkpoints/teacher_mpi.pt --epochs 20 --metrics_csv runs/m_teacher_mpi.csv

python train_student.py --dataset mpiigaze --mpi_root $env:MPI_ROOT --mpi_val_persons $env:MPI_VAL --student_arch gaze_micro --mpi_max_train_samples 10000 --amp --num_workers 0 --checkpoint checkpoints/student_baseline_mpi.pt --epochs 20 --metrics_csv runs/m_student_mpi.csv

python train_kd.py --dataset mpiigaze --mpi_root $env:MPI_ROOT --mpi_val_persons $env:MPI_VAL --student_arch gaze_micro --mpi_max_train_samples 10000 --amp --num_workers 0 --teacher_ckpt checkpoints/teacher_mpi.pt --checkpoint checkpoints/student_kd_mpi.pt --epochs 20 --metrics_csv runs/m_kd_mpi.csv
```

Use **`--num_workers 0`** if the DataLoader workers fail on your OS; reduce **`--batch_size`** if you run out of GPU memory.

**Speed (MPIIGaze + training):**

- **`.mat` preload:** If a split touches at most **`--mpi_preload_max_unique`** distinct `dayYY.mat` files (default **512**), they are loaded into RAM once per process (log: `MPIIGaze [train]: preloaded …`). This avoids repeated `loadmat` on every sample when batches jump across files. Disable with **`--mpi_no_preload`**, or set **`--mpi_preload_max_unique 0`** to force lazy IO. If the unique count exceeds the limit, the code falls back to lazy load and prints a hint.
- **GPU:** Add **`--amp`** to any of `train_teacher.py` / `train_student.py` / `train_kd.py` for CUDA **mixed precision** (often noticeably faster on modern GPUs).
- **DataLoader:** When **`num_workers > 0`**, workers stay alive across epochs (**`persistent_workers`**). Try **`--num_workers 4`** (or higher if CPU allows). On **Windows**, spawn + multiple workers can **duplicate** a large RAM preload—prefer **`--num_workers 0`** with preload, or **`--mpi_no_preload`** with more workers.
- **Larger batches / smaller input:** Increase **`--batch_size`** if VRAM allows; lower **`--image_size`** (e.g. 128) speeds up each step but changes the setup—keep it consistent for eval.

### 4) Evaluation

Do **not** pass **`--csv`** when using **`--dataset mpiigaze`**. Use the **same** **`--mpi_root`** and **`--mpi_val_persons`** as in training. By default **`evaluate.py`** scores the **validation** participants; set **`--mpi_eval_split train`** to score the **training** participants instead.

```bash
python evaluate.py --model teacher --checkpoint checkpoints/teacher_mpi.pt \
  --dataset mpiigaze --mpi_root "$MPI_ROOT" --mpi_val_persons "$MPI_VAL" \
  --teacher_arch mobilenet_v2 \
  --num_workers 4 --export_json runs/eval_teacher_mpi.json

python evaluate.py --model student --checkpoint checkpoints/student_baseline_mpi.pt \
  --dataset mpiigaze --mpi_root "$MPI_ROOT" --mpi_val_persons "$MPI_VAL" \
  --num_workers 4 --export_json runs/eval_student_mpi.json

python evaluate.py --model student --checkpoint checkpoints/student_kd_mpi.pt \
  --dataset mpiigaze --mpi_root "$MPI_ROOT" --mpi_val_persons "$MPI_VAL" \
  --num_workers 4 --export_json runs/eval_kd_mpi.json \
  --save_predictions runs/student_kd_mpi_val.npz
```

**Windows (Command Prompt)** — **one line** per command:

```bat
python evaluate.py --model teacher --checkpoint checkpoints\teacher_mpi.pt --dataset mpiigaze --mpi_root %MPI_ROOT% --mpi_val_persons %MPI_VAL% --teacher_arch mobilenet_v2 --num_workers 0 --export_json runs\eval_teacher_mpi.json

python evaluate.py --model student --checkpoint checkpoints\student_baseline_mpi.pt --dataset mpiigaze --mpi_root %MPI_ROOT% --mpi_val_persons %MPI_VAL% --num_workers 0 --export_json runs\eval_student_mpi.json

python evaluate.py --model student --checkpoint checkpoints\student_kd_mpi.pt --dataset mpiigaze --mpi_root %MPI_ROOT% --mpi_val_persons %MPI_VAL% --num_workers 0 --export_json runs\eval_kd_mpi.json --save_predictions runs\student_kd_mpi_val.npz
```

**Windows (PowerShell)** — **one line** per command (match **teacher** `mobilenet_v3_small` and **student** `gaze_micro` from training above):

```powershell
python evaluate.py --model teacher --checkpoint checkpoints/teacher_mpi.pt --dataset mpiigaze --mpi_root $env:MPI_ROOT --mpi_val_persons $env:MPI_VAL --teacher_arch mobilenet_v3_small --num_workers 0 --export_json runs/eval_teacher_mpi.json

python evaluate.py --model student --checkpoint checkpoints/student_baseline_mpi.pt --dataset mpiigaze --mpi_root $env:MPI_ROOT --mpi_val_persons $env:MPI_VAL --student_arch gaze_micro --num_workers 0 --export_json runs/eval_student_mpi.json

python evaluate.py --model student --checkpoint checkpoints/student_kd_mpi.pt --dataset mpiigaze --mpi_root $env:MPI_ROOT --mpi_val_persons $env:MPI_VAL --student_arch gaze_micro --num_workers 0 --export_json runs/eval_kd_mpi.json --save_predictions runs/student_kd_mpi_val.npz
```

### 5) Paper figures (same scripts as synthetic)

Merge the three JSON files, then build PDFs under **`paper/figures/`**.

**`loss_curves.pdf` and your real training runs:** the loss figure is **not** built from `summary.json`. It only uses the three **per-epoch metric CSVs** you wrote during training with `--metrics_csv`. You must pass **`--metrics_teacher`**, **`--metrics_student`**, and **`--metrics_kd`** to `make_paper_figures.py`, and each file must exist. If any path is missing or you use **`--demo`**, the script draws **placeholder exponential curves** instead. For the MPII commands in §3 above, those files are exactly:

- `runs/m_teacher_mpi.csv`
- `runs/m_student_mpi.csv`
- `runs/m_kd_mpi.csv`

Full example (merge eval JSON, then figures — **includes the CSVs so the loss plot matches your logs**):

```bash
python scripts/build_eval_summary.py --out runs/summary_mpi.json \
  --teacher runs/eval_teacher_mpi.json \
  --student_baseline runs/eval_student_mpi.json \
  --student_kd runs/eval_kd_mpi.json

python scripts/make_paper_figures.py --out_dir paper/figures \
  --summary runs/summary_mpi.json \
  --metrics_teacher runs/m_teacher_mpi.csv \
  --metrics_student runs/m_student_mpi.csv \
  --metrics_kd runs/m_kd_mpi.csv \
  --scatter_npz runs/student_kd_mpi_val.npz
```

**Windows (Command Prompt)** — **one line** per command (`runs\` paths; forward slashes also work in Python):

```bat
python scripts\build_eval_summary.py --out runs\summary_mpi.json --teacher runs\eval_teacher_mpi.json --student_baseline runs\eval_student_mpi.json --student_kd runs\eval_kd_mpi.json

python scripts\make_paper_figures.py --out_dir paper\figures --summary runs\summary_mpi.json --metrics_teacher runs\m_teacher_mpi.csv --metrics_student runs\m_student_mpi.csv --metrics_kd runs\m_kd_mpi.csv --scatter_npz runs\student_kd_mpi_val.npz
```

**Windows (PowerShell)** — **one line** per command:

```powershell
python scripts/build_eval_summary.py --out runs/summary_mpi.json --teacher runs/eval_teacher_mpi.json --student_baseline runs/eval_student_mpi.json --student_kd runs/eval_kd_mpi.json

python scripts/make_paper_figures.py --out_dir paper/figures --summary runs/summary_mpi.json --metrics_teacher runs/m_teacher_mpi.csv --metrics_student runs/m_student_mpi.csv --metrics_kd runs/m_kd_mpi.csv --scatter_npz runs/student_kd_mpi_val.npz
```

If you skip **`--save_predictions`**, omit **`--scatter_npz ...`** from the last `make_paper_figures.py` command. Placeholder figures (synthetic loss + bars): **`python scripts/make_paper_figures.py --demo --out_dir paper/figures`**.

### 6) Web demo vs. MPIIGaze crops

The **browser demo** resizes **webcam / upload** images to **224×224** (optional face crop). Models trained on **`--dataset mpiigaze`** see **Normalized eye patches** upsampled to 224. Expect a **domain gap** unless you match preprocessing (e.g. eye-region crops or a model trained on full-face CSV data).

### 7) Optional: CSV + `GazeDataset` instead

To train with **`--dataset csv`**, export **`image_path,gaze_x,gaze_y`** yourself and keep the same convention everywhere. See [docs/mpiigaze_next_steps.md](docs/mpiigaze_next_steps.md) (route B) and the [CSV format](#csv-format) section in this README.

## Paper, plots, and LaTeX

The course template you have locally can be compared with [paper/report.tex](paper/report.tex) (already filled with structure, text, and figure includes).

1. **Train with metric logs** (optional but recommended):

   ```bash
   python train_teacher.py ... --metrics_csv runs/m_teacher.csv
   python train_student.py ... --metrics_csv runs/m_student.csv
   python train_kd.py ... --metrics_csv runs/m_kd.csv
   ```

2. **Export evaluation JSON** after you have checkpoints.

   **Synthetic (CSV) example:**

   ```bash
   python evaluate.py --model teacher --checkpoint checkpoints/teacher_best.pt --teacher_arch mobilenet_v2 --csv data/synthetic/val.csv --data_root data/synthetic --export_json runs/eval_teacher.json
   python evaluate.py --model student --checkpoint checkpoints/student_baseline_best.pt --csv data/synthetic/val.csv --data_root data/synthetic --export_json runs/eval_student.json
   python evaluate.py --model student --checkpoint checkpoints/student_kd_best.pt --csv data/synthetic/val.csv --data_root data/synthetic --export_json runs/eval_kd.json
   ```

   **MPIIGaze:** use the same three models but **`--dataset mpiigaze --mpi_root ... --mpi_val_persons ...`** and **no `--csv`** — copy the block in **[Real data workflow (MPIIGaze)](#real-data-workflow-mpiigaze)** §4.

   Optional scatter figure: add `--save_predictions runs/student_kd_val.npz` (or `runs/student_kd_mpi_val.npz` for MPII) on the last `evaluate.py` command.

3. **Merge JSON and build figures:** pass the **same `--metrics_csv` paths** from training so **`loss_curves.pdf`** shows real validation MSE (not the built-in dummy curves).

   **Synthetic data** (paths from the quickstart in this README):

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

   **MPIIGaze** (same CSV names as [§3 in Real data workflow](#real-data-workflow-mpiigaze)):

   ```bash
   python scripts/build_eval_summary.py --out runs/summary_mpi.json \
     --teacher runs/eval_teacher_mpi.json \
     --student_baseline runs/eval_student_mpi.json \
     --student_kd runs/eval_kd_mpi.json

   python scripts/make_paper_figures.py --out_dir paper/figures \
     --summary runs/summary_mpi.json \
     --metrics_teacher runs/m_teacher_mpi.csv \
     --metrics_student runs/m_student_mpi.csv \
     --metrics_kd runs/m_kd_mpi.csv \
     --scatter_npz runs/student_kd_mpi_val.npz
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

**MPIIGaze without CSV:** use **`--dataset mpiigaze`** (see [Real data workflow (MPIIGaze)](#real-data-workflow-mpiigaze)). **With CSV:** keep this interface and export `(path, gaze_x, gaze_y)` from native labels (see [docs/mpiigaze_next_steps.md](docs/mpiigaze_next_steps.md)). **GazeCapture** and others: same CSV idea once paths and labels align.

## Training

Run all commands from `gaze_kd_project/` unless you adjust `PYTHONPATH`.

### 1) Teacher (MobileNetV2 by default in this README)

The documented pipeline uses **MobileNetV2** as the teacher (~2.2M params): strong enough to supervise the student but lighter than ResNet18 on small gaze splits. **MobileNetV3-Small** (~1.5M) is also supported as teacher via **`--teacher_arch mobilenet_v3_small`**. **`--teacher_arch`** is stored in the checkpoint; **`train_kd.py`** / **`evaluate.py`** read it unless you override with **`--teacher_arch`**.

```bash
python train_teacher.py \
  --teacher_arch mobilenet_v2 \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --data_root data \
  --checkpoint checkpoints/teacher_best.pt
```

**Optional heavier teacher:** **`--teacher_arch resnet18`** (~11M params).

### 2) Student baseline (no KD)

Default student backbone is **MobileNetV3-Small**. Use **`--student_arch shufflenet_v2_x0_5`** (~0.34M) or **`gaze_micro`** (~50k, from scratch) for smaller models.

```bash
python train_student.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --data_root data \
  --checkpoint checkpoints/student_baseline_best.pt
```

### 3) Student + knowledge distillation

Requires a trained teacher checkpoint (paths above). Match **`--student_arch`** across **`train_student.py`**, **`train_kd.py`**, and **`evaluate.py`** when not using the default (e.g. **gaze_micro** or ShuffleNet).

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

**Teacher (MobileNetV2):**

```bash
python evaluate.py --model teacher --checkpoint checkpoints/teacher_best.pt \
  --teacher_arch mobilenet_v2 \
  --csv data/val.csv --data_root data
```
(Omit **`--teacher_arch`** if the checkpoint already stores it in `extra.args`.)

**Student (baseline or KD checkpoint):**

```bash
python evaluate.py --model student --checkpoint checkpoints/student_baseline_best.pt \
  --csv data/val.csv --data_root data

python evaluate.py --model student --checkpoint checkpoints/student_kd_best.pt \
  --csv data/val.csv --data_root data
```
(Omit **`--student_arch`** if the checkpoint stores it in `extra.args`; otherwise pass **`gaze_micro`** / **`shufflenet_v2_x0_5`** as needed.)

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

1. **Data:** either generate **synthetic** CSV + images, or follow **[Real data workflow (MPIIGaze)](#real-data-workflow-mpiigaze)** / prepare your own **CSV + images**.  
2. `train_teacher.py`  
3. `train_student.py`  
4. `train_kd.py` (needs teacher checkpoint)  
5. `evaluate.py` on the **same split** (same `--csv` / `--dataset` / `--mpi_*` flags as training) for teacher, student baseline, and KD student.

## Results table for your report

| Model | MSE | MAE | Mean L2 | Params | CKPT (MB) | ms/img | FPS |
|-------|-----|-----|---------|--------|-----------|--------|-----|
| Teacher (MobileNetV2) | | | | | | | |
| Student baseline (MobileNetV3-Small) | | | | | | | |
| Student + KD (MobileNetV3-Small) | | | | | | | |

You typically want: **KD student better than baseline** on error metrics. Here the student (**MobileNetV3-Small**) is usually **smaller / faster** than the **MobileNetV2** teacher as well.

## Future improvements

- **Angular error** (standard in gaze papers): needs camera/head pose to convert 2D/3D vectors; add a metric module once you fix your label convention.  
- **Data augmentation**: random crops, color jitter, slight blur — watch label consistency if gaze is defined in image space.  
- **Feature distillation**: match intermediate feature maps, not only outputs.  
- **Deployment**: export student to **ONNX**, optimize with **TensorRT** or mobile runtimes (Core ML / TFLite) for on-device FPS.

## License / course use

Code is intended for educational use; swap in your dataset and cite MPIIGaze / GazeCapture if you use them.
