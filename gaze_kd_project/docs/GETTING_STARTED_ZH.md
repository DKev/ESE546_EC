# 一步一步使用本项（gaze_kd_project）

下面假设你在项目里的 **`gaze_kd_project` 文件夹** 下操作（终端先 `cd` 进去）。系统可以是 macOS / Linux / Windows；有 NVIDIA GPU 时建议装好 **CUDA 版 PyTorch**。

---

## 第 0 步：准备 Python 环境

1. 安装 **Python 3.10+**（推荐）。
2. （推荐）建虚拟环境，避免和系统包冲突：

   ```bash
   cd gaze_kd_project
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. 安装 PyTorch：到 [https://pytorch.org](https://pytorch.org) 按你的 CUDA 版本选一条 `pip install`，再装其余依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. 快速自检：

   ```bash
   python3 -c "import torch; print('cuda:', torch.cuda.is_available())"
   ```

---

## 第 1 步：准备数据（二选一）

### 方案 A：合成数据（最快，适合先把流程跑通）

在 `gaze_kd_project` 下执行：

```bash
python scripts/generate_synthetic_gaze_dataset.py --out_root data/synthetic --n_train 4000 --n_val 800
```

会生成：

- `data/synthetic/train.csv`、`data/synthetic/val.csv`
- `data/synthetic/images/train/*.png`、`images/val/*.png`

后面所有命令里：

- `--data_root` 用 **`data/synthetic`**
- `--train_csv` 用 **`data/synthetic/train.csv`**
- `--val_csv` 用 **`data/synthetic/val.csv`**

### 方案 B：自己的数据 / MPIIGaze

CSV 必须包含表头：

```text
image_path,gaze_x,gaze_y
```

`image_path` 可以是绝对路径，或相对 **`--data_root`** 的路径。更细的说明见 [data_sources.md](data_sources.md)。

---

## 第 2 步：训练 Teacher（ResNet18）

```bash
python train_teacher.py \
  --train_csv data/synthetic/train.csv \
  --val_csv data/synthetic/val.csv \
  --data_root data/synthetic \
  --checkpoint checkpoints/teacher_best.pt \
  --metrics_csv runs/m_teacher.csv
```

- 默认会下载 **ImageNet 预训练** backbone；不想用预训练可加 `--no_pretrained`。
- 训练过程中终端会打印每个 epoch 的 train/val MSE；`--metrics_csv` 会把曲线记到文件里，后面画图用。
- 验证集上 **MSE 最低** 的那次会存到 `checkpoints/teacher_best.pt`。

---

## 第 3 步：训练 Student 基线（无蒸馏）

```bash
python train_student.py \
  --train_csv data/synthetic/train.csv \
  --val_csv data/synthetic/val.csv \
  --data_root data/synthetic \
  --checkpoint checkpoints/student_baseline_best.pt \
  --metrics_csv runs/m_student.csv
```

得到 **`checkpoints/student_baseline_best.pt`**。

---

## 第 4 步：知识蒸馏训练 Student（需要 Teacher 权重）

```bash
python train_kd.py \
  --teacher_ckpt checkpoints/teacher_best.pt \
  --train_csv data/synthetic/train.csv \
  --val_csv data/synthetic/val.csv \
  --data_root data/synthetic \
  --checkpoint checkpoints/student_kd_best.pt \
  --alpha 0.5 \
  --metrics_csv runs/m_kd.csv
```

- `alpha`：蒸馏项权重（与 `config.py` 默认一致时可省略）。
- 最佳 checkpoint 按 **验证集 total loss**（GT MSE + α×KD MSE）保存。

---

## 第 5 步：评估（指标 + 参数量 + 速度）

对 **同一 val CSV** 分别评估三个模型（路径按你实际 checkpoint 修改）：

```bash
python evaluate.py --model teacher --checkpoint checkpoints/teacher_best.pt \
  --csv data/synthetic/val.csv --data_root data/synthetic \
  --export_json runs/eval_teacher.json

python evaluate.py --model student --checkpoint checkpoints/student_baseline_best.pt \
  --csv data/synthetic/val.csv --data_root data/synthetic \
  --export_json runs/eval_student.json

python evaluate.py --model student --checkpoint checkpoints/student_kd_best.pt \
  --csv data/synthetic/val.csv --data_root data/synthetic \
  --export_json runs/eval_kd.json \
  --save_predictions runs/student_kd_val.npz
```

终端会打印 MSE、MAE、Mean L2、参数量、权重文件大小、延迟/FPS；JSON 留给画图或写报告。

---

## 第 6 步（可选）：生成论文用图表

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

若还没训练完，可先用占位图：

```bash
python scripts/make_paper_figures.py --demo --out_dir paper/figures
```

LaTeX 论文见 `paper/report.tex`，在 `paper/` 下执行 `pdflatex report.tex`（需本机安装 TeX）。

---

## 第 7 步（可选）：浏览器里看眼动效果 + 人脸框

1. 安装依赖里已有 FastAPI、Uvicorn、OpenCV 等（`requirements.txt`）。
2. 启动服务（**在 `gaze_kd_project` 目录**）：

   ```bash
   export GAZE_CKPT=checkpoints/student_kd_best.pt
   export GAZE_MODEL=student
   python -m uvicorn web.server:app --host 127.0.0.1 --port 8765
   ```

3. 浏览器打开：**http://127.0.0.1:8765/**  
   - 允许摄像头 → 「预测一帧」或勾选「连续预测」  
   - 绿色虚线为人脸检测框，蓝点为注视点映射  

4. **只测界面、不加载模型**：

   ```bash
   export GAZE_WEB_DEMO=1
   python -m uvicorn web.server:app --host 127.0.0.1 --port 8765
   ```

5. 关闭人脸裁剪（整图推理）：

   ```bash
   export GAZE_FACE_CROP=0
   ```

更多说明见仓库根目录的 [README.md](../README.md)。

---

## 建议的整体顺序（小结）

| 顺序 | 做什么 |
|------|--------|
| 0 | 建 venv、`pip install`、确认 CUDA |
| 1 | 生成或准备 CSV + 图片 |
| 2 | `train_teacher.py` |
| 3 | `train_student.py` |
| 4 | `train_kd.py`（依赖 teacher 权重） |
| 5 | 三次 `evaluate.py` + 可选 `export_json` |
| 6 | 可选：`build_eval_summary` + `make_paper_figures` + 写 `paper/report.tex` |
| 7 | 可选：`uvicorn` 开网页 demo |

遇到问题先看终端报错：常见是 **CSV 路径不对**、**图片路径与 `data_root` 不一致**、或 **checkpoint 路径写错**。
