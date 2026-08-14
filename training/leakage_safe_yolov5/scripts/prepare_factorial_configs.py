#!/usr/bin/env python3
"""Resolve and validate portable 2x2 factorial dataset specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


NAMES = ["cigarette", "flame", "smoke"]


def image_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for image in sorted(item for item in path.iterdir() if item.is_file() and not item.name.startswith(".")):
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        result[image.name] = digest
    return result


def validate_image_label_pair(images: Path) -> dict[str, int]:
    labels = images.parent / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Missing image/label directory pair: {images}")
    image_stems = {path.stem for path in images.iterdir() if path.is_file() and not path.name.startswith(".")}
    label_stems = {path.stem for path in labels.glob("*.txt")}
    if image_stems != label_stems:
        raise ValueError(f"Image/label stem mismatch: {images}")
    return {"images": len(image_stems), "labels": len(label_stems)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "matrix",
        type=Path,
        default=Path(__file__).parents[1] / "experiments" / "factorial" / "matrix.yaml",
        nargs="?",
    )
    parser.add_argument("--package-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    package = args.package_root.resolve()
    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    output_dir = (args.output_dir or args.matrix.parent / "generated").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    common = matrix["common_evaluation"]
    val_images = package / common["val_images"]
    test_images = package / common["test_images"]
    common_counts = {
        "valid": validate_image_label_pair(val_images),
        "test": validate_image_label_pair(test_images),
    }
    status: dict[str, object] = {
        "version": matrix["version"],
        "seed": args.seed,
        "common_evaluation": common_counts,
        "cells": {},
    }
    inventories: dict[str, dict[str, str]] = {}

    for cell_id, spec in matrix["cells"].items():
        is_seed_specific = "train_images_template" in spec
        relative_train = (
            spec["train_images_template"].format(seed=args.seed)
            if is_seed_specific
            else spec["train_images"]
        )
        train_images = package / relative_train
        cell_status: dict[str, object] = {
            "name": spec["name"],
            "leakage": spec["leakage"],
            "training_ontology": spec["training_ontology"],
        }
        if not train_images.is_dir():
            cell_status.update(status="pending", reason=spec.get("pending", f"missing {train_images}"))
            status["cells"][cell_id] = cell_status
            continue
        counts = validate_image_label_pair(train_images)
        inventories[cell_id] = image_inventory(train_images)
        data = {
            "path": str(package),
            "train": str(train_images),
            "val": str(val_images),
            "test": str(test_images),
            "nc": len(NAMES),
            "names": NAMES,
        }
        seed_suffix = f"_seed{args.seed}" if is_seed_specific else ""
        yaml_path = output_dir / f"{cell_id}_{spec['name']}{seed_suffix}.yaml"
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        cell_status.update(status="ready", train=counts, data_yaml=str(yaml_path))
        status["cells"][cell_id] = cell_status

    if "A" in inventories and "B" in inventories and inventories["A"] != inventories["B"]:
        raise SystemExit("A/B training image identities or bytes differ; ontology comparison is not controlled")
    if "C" in inventories and "D" in inventories and inventories["C"] != inventories["D"]:
        raise SystemExit("C/D training image identities or bytes differ; ontology comparison is not controlled")
    if "A" in inventories and "C" in inventories and len(inventories["A"]) != len(inventories["C"]):
        raise SystemExit("Controlled leakage changed training-set size")
    if "B" in inventories and "D" in inventories and len(inventories["B"]) != len(inventories["D"]):
        raise SystemExit("Controlled leakage changed training-set size")

    status_path = output_dir / f"MATRIX_STATUS_seed{args.seed}.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "MATRIX_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
