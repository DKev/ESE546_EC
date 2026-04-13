# Data sources for gaze estimation

## 真实数据怎么拿？（中文概要）

本项目的 `GazeDataset` 只认 CSV：`image_path,gaze_x,gaze_y`。真实数据集**一般不会直接给你这个 CSV**，需要你自己下载官方包，再按论文/文档把标签转成两个数，并导出成我们的格式。

### 首选：MPIIGaze（论文里最常用之一）

1. **下载**（任选一个镜像，体积约 2 GB 量级）  
   - MPI-INF 直链：<http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz>  
   - 或 DaRUS：<https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-3230>  
2. **许可**：多为 **非商业科研**，使用前请阅读包内 / 网页上的条款。  
3. **解压后**：里面是规范化眼图、标注、划分列表等，**没有**现成的 `train.csv`。  
4. **对接本项目**：  
   - 选定用哪种图像（例如 `Normalized` 子集里的图）。  
   - 从官方标注里取出 gaze 方向（常为 3D 向量），再按你的定义变成 **2D 的 `(gaze_x, gaze_y)`**（例如投到屏幕平面、或归一化到 `[-1,1]`，**全程保持一致**）。  
   - 写一个小脚本：对每张训练/验证图写一行 `相对 data_root 的路径, gaze_x, gaze_y`，得到 `train.csv` / `val.csv`。  
5. **省事做法**：在 GitHub 上搜 `MPIIGaze pytorch` / `MPIIGaze dataloader`，复用别人已经写好的 **读图 + 读 label** 代码，只在最后加一段把 `(path, gx, gy)` 存成 CSV。

**已经下好压缩包之后具体步骤（解压 → 对齐标签 → CSV → 训练命令）：** 见 **[mpiigaze_next_steps.md](mpiigaze_next_steps.md)**。也可运行 `python scripts/inspect_mpiigaze_layout.py /你的/MPIIGaze路径` 查看目录结构。

### 其他公开集（更重）

- **GazeCapture**：规模大，获取与预处理通常比 MPIIGaze 麻烦，需看官方说明。  
- 无论用哪个，最后都落回同一种 CSV 接口即可。

### 和「合成数据」的关系

- 合成数据脚本只用于 **跑通代码、画曲线**；写报告若要求真实场景，应明确写你用的是 **MPIIGaze（或别的）**，并说明预处理与标签含义。

---

## Quick start (no download): synthetic dataset

For debugging, course demos, and reproducible plots without large files:

```bash
cd gaze_kd_project
python scripts/generate_synthetic_gaze_dataset.py --out_root data/synthetic --n_train 4000 --n_val 800
```

Then train with `--data_root data/synthetic`, `--train_csv data/synthetic/train.csv`, `--val_csv data/synthetic/val.csv`.

**Limitation:** Images are cartoon ellipses; numbers are **not** comparable to in-the-wild benchmarks. Use them to validate code and KD behavior, then switch to a real dataset for the final report if required.

## MPIIGaze (real benchmark, common in papers)

MPIIGaze is a standard appearance-based gaze dataset (laptop users in the wild).

- **Landing page:** [MPIIGaze dataset](http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz) (MPI-INF)  
- **Mirror / DOI:** [DaRUS repository](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-3230) (ZIP, ~2.2 GB)  
- **License:** typically **non-commercial research** (read the dataset terms before use).

The release is **not** a single project CSV. You usually:

1. Download and unpack the archive.
2. Use the dataset’s normalized eye images (or crop faces with a detector).
3. Convert 3D gaze vectors (or 2D projections used in your protocol) to your `(gaze_x, gaze_y)` convention.
4. Export rows: `image_path,gaze_x,gaze_y` with paths relative to `--data_root`.

**Tip:** Many GitHub repos provide PyTorch loaders for MPIIGaze; reuse their label math, then dump a CSV that matches [`datasets/gaze_dataset.py`](../datasets/gaze_dataset.py).

## GazeCapture

Large mobile gaze dataset; access and preprocessing are heavier than MPIIGaze. If you use it, same idea: end with `image_path,gaze_x,gaze_y` compatible with this codebase.

## What to write in your paper

- State clearly whether results are on **synthetic** or **MPIIGaze** (or another public set).
- If synthetic: motivate as **sanity check / pipeline validation** and cite MPIIGaze for intended real-world deployment.
- If MPIIGaze: describe preprocessing (crop size, normalization, train/val split) and cite the original paper (Zhang et al., CVPR 2015).
