# MPIIGaze 下载之后怎么做？

你的代码只认 **`image_path,gaze_x,gaze_y` 的 CSV**。MPIIGaze 解压后**不会**直接给你这个文件，需要你自己从官方目录里**对齐「每张图 ↔ 两个数」**再导出。

---

## 1. 先确认解压对了

顶层通常能看到类似结构（不同压缩包可能略有差异）：

- **`Data/`**  
  - `Original/` — 原始采集  
  - **`Normalized/`** — **最常用**：已做几何归一化的眼部区域图像（适合做 appearance-based）  
  - `Calibration/` — 标定相关  
- **`Evaluation Subset/`** — 论文里评估用的样本列表（文件名列表等）  
- **`Annotation Subset/`** — 带人工标注点的子集说明  

做本项目时，优先在 **`Data/Normalized/`** 下面找按被试编号的文件夹（如 `p00`, `p01`, …）。

---

## 2. 两条可行路线（选一条）

### 路线 A：用现成 PyTorch 代码（最省事，推荐）

1. 在 GitHub 搜 **`MPIIGaze pytorch`** 或 **`MPIIGaze dataloader`**，选一个维护较新、说明清楚的仓库。  
2. 按它的 README 把 `MPIIGAZE_ROOT` 指到你解压的目录，能正常 `len(dataset)`、`__getitem__` 出图像和 gaze。  
3. 在它的 `__getitem__` 里看清：  
   - 图像文件**磁盘路径**是什么；  
   - gaze 是 **yaw/pitch（弧度）** 还是别的（务必记下来写进报告）。  
4. 写一个小脚本，循环 dataset，写出两行 CSV：  
   - `train.csv` / `val.csv`  
   - 每行：`相对路径,gaze_x,gaze_y`（`gaze_x,gaze_y` 与 dataloader 里那两个数**完全一致**，不要混用单位）。  
5. 把**所有会在 CSV 里出现的图片**放到你定的 **`--data_root`** 下（或 CSV 里写绝对路径），保证 `GazeDataset` 能打开。

这样你不用自己啃 `.mat` / HDF5 的细节。

### 路线 B：自己从原始 `Data/` 里抠标签

1. 进入例如 `Data/Normalized/p00/`（具体以你磁盘为准），看图片是按 **`day01/0001.jpg`** 这种层级还是平铺。  
2. 在同一被试目录或官方文档说明的位置，找 **`.mat`** 或其它标注文件（不同下载版本位置可能不同）。  
3. 用 Python 探结构（需安装 `scipy`）：  

   ```bash
   pip install scipy
   python -c "import scipy.io as sio; d=sio.loadmat('你的.mat', struct_as_record=False, squeeze_me=True); print([k for k in d if not k.startswith('__')])"
   ```  

   找到存放 **逐帧 gaze** 的数组（常见为 **2 维：yaw、pitch，弧度**）。  
4. 建立 **与图像文件名一一对应** 的索引（有的 mat 里带路径或下标，要对齐官方 readme）。  
5. 导出 CSV：路径列用**相对 `data_root` 的路径**，标签列用与训练时一致的 `gaze_x,gaze_y`（若用弧度，论文里写清楚）。  

**注意：** 原 Caffe 流程里对 Normalized 图有时还有 **BGR→RGB、水平翻转、旋转 90°** 等预处理；若你跳过这些，要和「从图直接读入」的设定一致，否则标签与图像不对齐。

---

## 3. 划分 train / val

- **课程作业常见做法：**  
  - **按人划分**（例如 14 人训练、1 人验证），或  
  - 使用 **`Evaluation Subset`** 里给的列表做测试/验证，其余做训练。  
- 无论哪种，在报告里写清楚：**划分规则 + 人数/帧数**。

---

## 4. 接到 `gaze_kd_project`

### 4a 内置：`--dataset mpiigaze`（直接读 `Data/Normalized/*.mat`）

本仓库已实现 **`MPIIGazeNormalizedDataset`**：从 `Data/Normalized/pXX/dayYY.mat` 读取 `left` / `right` 的裁剪眼图 `(36×60)` 与单位视线向量，并映射为 `(gaze_x, gaze_y)`（由 yaw / pitch 缩放至约 `[-1, 1]`，与合成数据标度一致）。验证集默认按 **被试 id** 划分：`--mpi_val_persons 14,15` 表示 `p14`、`p15` 做 val，其余做 train。

```bash
cd gaze_kd_project
pip install -r requirements.txt   # 含 scipy

python train_teacher.py --dataset mpiigaze --mpi_root ../MPIIGaze \
  --mpi_val_persons 14,15 --epochs 20 --checkpoint checkpoints/teacher_mpi.pt

python train_student.py --dataset mpiigaze --mpi_root ../MPIIGaze \
  --mpi_val_persons 14,15 --epochs 20 --checkpoint checkpoints/student_mpi.pt

python train_kd.py --dataset mpiigaze --mpi_root ../MPIIGaze \
  --teacher_ckpt checkpoints/teacher_mpi.pt --mpi_val_persons 14,15
```

评估（默认评 val 被试）：

```bash
python evaluate.py --model teacher --checkpoint checkpoints/teacher_mpi.pt \
  --dataset mpiigaze --mpi_root ../MPIIGaze --mpi_eval_split val
```

快速冒烟可加 `--mpi_max_samples 2048`（train / val 各最多保留这么多条，按扫描顺序截断）。若只想**缩小训练集**、保留完整验证集，用 **`--mpi_max_train_samples 10000`**（示例）即可。

### 4b CSV + `GazeDataset`

假设你把导出用的图片都放在 `data/mpii/` 下，CSV 在 `data/mpii/train.csv`：

```bash
python train_teacher.py \
  --train_csv data/mpii/train.csv \
  --val_csv data/mpii/val.csv \
  --data_root data/mpii \
  --checkpoint checkpoints/teacher_best.pt
```

`image_path` 列写 **`data_root` 下的相对路径**（例如 `p01/day01/0001.png`）。

---

## 5. 本仓库里的小工具

在项目根目录执行（把路径换成你的 MPIIGaze 根目录）：

```bash
python scripts/inspect_mpiigaze_layout.py /path/to/MPIIGaze
```

它会打印顶层目录、是否找到 `Data/Normalized`，并抽样列出某个被试文件夹里的文件，方便你对照路线 B 找标签文件。

---

## 6. 进一步参考

- 官方页面：<http://www.mpi-inf.mpg.de/MPIIGazeDataset>  
- 论文：Zhang et al., *Appearance-Based Gaze Estimation in the Wild*, CVPR 2015  
- 预处理相关代码可参考：`https://github.com/xucong-zhang/data-preprocessing-gaze`（第三方，与官方包配合使用）

若你贴出 **`Data/Normalized` 下某一层的 `ls` 截图或文件名列表**，可以更具体地说标签文件可能在哪儿、CSV 该怎么对齐。
