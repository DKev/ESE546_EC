# Data sources for gaze estimation

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
