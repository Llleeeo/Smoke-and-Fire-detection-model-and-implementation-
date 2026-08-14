#!/usr/bin/env python3
"""Run one pinned YOLO26n factorial cell."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CELL_NAMES = {
    "A": "clean_original",
    "B": "clean_audited",
    "C": "leaked_original",
    "D": "leaked_audited",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=sorted(CELL_NAMES))
    parser.add_argument("seed", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    factorial_dir = Path(__file__).resolve().parent
    package = factorial_dir.parents[1]
    subprocess.run(
        [
            sys.executable,
            str(package / "scripts" / "prepare_factorial_configs.py"),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    seed_suffix = f"_seed{args.seed}" if args.cell in {"C", "D"} else ""
    data_yaml = factorial_dir / "generated" / f"{args.cell}_{CELL_NAMES[args.cell]}{seed_suffix}.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"Cell {args.cell} seed {args.seed} is not ready; inspect "
            f"{factorial_dir / 'generated' / f'MATRIX_STATUS_seed{args.seed}.json'}"
        )
    project = package / "runs" / "factorial_yolo26"
    run_name = f"{args.cell}_{CELL_NAMES[args.cell]}_seed{args.seed}_{args.epochs}e"
    run_dir = project / run_name
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse existing run directory: {run_dir}")

    settings = {
        "model": "yolo26n.pt",
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": args.batch,
        "seed": args.seed,
        "device": args.device,
        "workers": args.workers,
        "project": str(project),
        "name": run_name,
        "deterministic": True,
        "optimizer": "auto",
        "exist_ok": False,
        "plots": True,
    }
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return

    from ultralytics import YOLO

    model = YOLO(settings.pop("model"))
    model.train(**settings)


if __name__ == "__main__":
    main()
