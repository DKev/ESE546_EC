#!/usr/bin/env python3
"""
Print a quick overview of an unpacked MPIIGaze tree (no SciPy required).

Usage (from ``gaze_kd_project``)::

    python scripts/inspect_mpiigaze_layout.py /path/to/MPIIGaze

See docs/mpiigaze_next_steps.md for what to do after download.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect MPIIGaze directory layout")
    p.add_argument("root", type=str, help="Path to unpacked MPIIGaze root folder")
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print(f"Root: {root}\n")
    print("Top-level entries:")
    for child in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        typ = "dir" if child.is_dir() else "file"
        print(f"  [{typ}] {child.name}")

    norm = root / "Data" / "Normalized"
    if not norm.is_dir():
        print("\n(No Data/Normalized found — layout may differ or path is wrong.)")
        return

    print(f"\nFound Normalized data: {norm}")
    subs = sorted([d for d in norm.iterdir() if d.is_dir()], key=lambda x: x.name)[:16]
    print(f"Participant folders (showing up to 16): {[d.name for d in subs]}")

    if not subs:
        return

    sample = subs[0]
    print(f"\nSample participant folder: {sample}")
    files = list(sample.rglob("*"))
    files = [f for f in files if f.is_file()][:30]
    print("First ~30 files under this participant (recursive):")
    for f in files:
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = f
        print(f"  {rel}")

    mats = [f for f in sample.rglob("*.mat")][:10]
    if mats:
        print("\nSample .mat files (inspect with scipy.io.loadmat):")
        for f in mats:
            print(f"  {f.relative_to(root)}")
    else:
        print("\n(No .mat under this participant — labels may live elsewhere; see docs/mpiigaze_next_steps.md)")


if __name__ == "__main__":
    main()
