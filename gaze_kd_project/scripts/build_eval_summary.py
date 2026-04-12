#!/usr/bin/env python3
"""
Merge several ``evaluate.py --export_json`` outputs into one file for plotting / tables.

Usage::

    python scripts/build_eval_summary.py --out runs/summary.json \\
        --teacher runs/eval_teacher.json \\
        --student_baseline runs/eval_student.json \\
        --student_kd runs/eval_kd.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge evaluation JSON files")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--teacher", type=str, default="", help="eval JSON for teacher")
    p.add_argument("--student_baseline", type=str, default="", help="eval JSON for student baseline")
    p.add_argument("--student_kd", type=str, default="", help="eval JSON for distilled student")
    args = p.parse_args()

    summary: dict[str, dict] = {}
    if args.teacher:
        summary["teacher"] = load_json(args.teacher)
    if args.student_baseline:
        summary["student_baseline"] = load_json(args.student_baseline)
    if args.student_kd:
        summary["student_kd"] = load_json(args.student_kd)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} with keys: {list(summary.keys())}")


if __name__ == "__main__":
    main()
