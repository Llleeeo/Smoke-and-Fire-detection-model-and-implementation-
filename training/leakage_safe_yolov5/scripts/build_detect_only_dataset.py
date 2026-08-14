#!/usr/bin/env python3
"""Convert mixed YOLO box/polygon labels to detection-only bounding boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path


SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_line(line: str) -> tuple[str, bool]:
    fields = line.split()
    class_id = int(fields[0])
    coordinates = [float(value) for value in fields[1:]]
    if len(coordinates) == 4:
        return line, False
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError(f"Unsupported YOLO label geometry: {line!r}")
    xs, ys = coordinates[0::2], coordinates[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    x_center, y_center = (left + right) / 2, (top + bottom) / 2
    width, height = right - left, bottom - top
    return " ".join([str(class_id), *(format(value, ".15g") for value in (x_center, y_center, width, height))]), True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("output_dataset", type=Path)
    args = parser.parse_args()

    source = args.source_dataset.resolve()
    output = args.output_dataset.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}_tmp_", dir=output.parent))
    counters = Counter()
    image_hashes: dict[str, str] = {}
    try:
        for split in SPLITS:
            source_images = source / split / "images"
            source_labels = source / split / "labels"
            output_images = temp / split / "images"
            output_labels = temp / split / "labels"
            output_images.mkdir(parents=True)
            output_labels.mkdir(parents=True)
            images = sorted(path for path in source_images.iterdir() if path.is_file() and not path.name.startswith("."))
            labels = sorted(source_labels.glob("*.txt"))
            if {path.stem for path in images} != {path.stem for path in labels}:
                raise SystemExit(f"Image/label mismatch in {split}")
            for image in images:
                shutil.copy2(image, output_images / image.name)
                digest = sha256(image)
                if sha256(output_images / image.name) != digest:
                    raise SystemExit(f"Image copy mismatch: {image}")
                image_hashes[f"{split}/{image.name}"] = digest
                counters[f"{split}_images"] += 1
            for label in labels:
                converted_lines = []
                for line in label.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    converted, was_polygon = convert_line(line)
                    converted_lines.append(converted)
                    counters["polygons_converted"] += int(was_polygon)
                    counters["boxes_unchanged"] += int(not was_polygon)
                    counters[f"class_{int(converted.split()[0])}"] += 1
                (output_labels / label.name).write_text(
                    "\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8"
                )
                counters[f"{split}_labels"] += 1
                counters[f"{split}_empty"] += int(not converted_lines)

        data_yaml = (source / "data.yaml").read_text(encoding="utf-8")
        lines = [
            f"path: ../{output.name}" if line.startswith("path:") else line
            for line in data_yaml.splitlines()
        ]
        (temp / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        image_tree = hashlib.sha256()
        for name, digest in sorted(image_hashes.items()):
            image_tree.update(name.encode())
            image_tree.update(bytes.fromhex(digest))
        manifest = {
            "status": "verified",
            "source_dataset": str(source),
            "output_dataset": str(output),
            "geometry": "detection_boxes_only",
            "polygons_converted_to_bounding_boxes": counters["polygons_converted"],
            "boxes_unchanged": counters["boxes_unchanged"],
            "class_counts": {str(class_id): counters[f"class_{class_id}"] for class_id in range(3)},
            "splits": {
                split: {
                    "images": counters[f"{split}_images"],
                    "label_files": counters[f"{split}_labels"],
                    "empty_label_files": counters[f"{split}_empty"],
                }
                for split in SPLITS
            },
            "images_byte_identical": True,
            "image_tree_sha256": image_tree.hexdigest(),
            "cache_files_copied": False,
        }
        (temp / "DETECT_ONLY_MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = [
            "# Detection-only geometry conversion",
            "",
            "Status: verified.",
            "",
            f"- Source: `{source}`",
            f"- Images copied byte-for-byte: {sum(counters[f'{split}_images'] for split in SPLITS)}",
            f"- Existing box rows unchanged: {counters['boxes_unchanged']}",
            f"- Polygon rows converted to tight axis-aligned bounding boxes: {counters['polygons_converted']}",
            "- Class IDs, image membership, split membership and label-file membership are unchanged.",
            "- YOLO cache files were excluded.",
            "",
        ]
        (temp / "DETECT_ONLY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
