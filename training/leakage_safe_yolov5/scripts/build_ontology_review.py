#!/usr/bin/env python3
"""Create a review queue for annotations whose historical class is ambiguous.

The CSV is intentionally a human-review artifact. This script does not infer or
rewrite labels. The optional HTML file displays the relevant boxes for review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "valid", "test")
DECISIONS = "flame | smoke | cigarette | smoking_action | invalid | ambiguous"


def find_image(image_dir: Path, stem: str) -> Path:
    matches = sorted(path for path in image_dir.glob(f"{stem}.*") if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected one image for {stem}, found {len(matches)}")
    return matches[0]


def parse_labels(label_path: Path) -> list[tuple[int, float, float, float, float, str]]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        class_id = int(fields[0])
        coordinates = list(map(float, fields[1:]))
        if len(fields) == 5:
            x_center, y_center, width, height = coordinates
            geometry = "box"
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            xs, ys = coordinates[0::2], coordinates[1::2]
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            x_center, y_center = (left + right) / 2, (top + bottom) / 2
            width, height = right - left, bottom - top
            geometry = "polygon_bbox"
        else:
            raise ValueError(f"Unsupported YOLO geometry: {label_path}:{line_number}")
        boxes.append((class_id, x_center, y_center, width, height, geometry))
    return boxes


def load_source_keys(manifest_path: Path | None) -> dict[tuple[str, str], str]:
    if manifest_path is None:
        return {}
    result = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(row["assigned_split"], row["output_image"])] = row["source_key"]
    return result


def build_queue(dataset: Path, review_class: int, source_keys: dict[tuple[str, str], str]) -> tuple[list[dict[str, object]], dict[str, list[tuple[int, float, float, float, float, str]]]]:
    rows: list[dict[str, object]] = []
    boxes_by_image: dict[str, list[tuple[int, float, float, float, float, str]]] = {}
    for split in SPLITS:
        label_dir = dataset / split / "labels"
        image_dir = dataset / split / "images"
        for label_path in sorted(label_dir.glob("*.txt")):
            boxes = parse_labels(label_path)
            relevant = [(index, box) for index, box in enumerate(boxes, 1) if box[0] == review_class]
            if not relevant:
                continue
            image_path = find_image(image_dir, label_path.stem)
            with Image.open(image_path) as opened:
                image_width, image_height = opened.size
            image_key = f"{split}/{image_path.name}"
            boxes_by_image[image_key] = boxes
            other_classes = "|".join(map(str, sorted({box[0] for box in boxes if box[0] != review_class})))
            for annotation_index, (_, x_center, y_center, width, height, geometry) in relevant:
                identity = f"{image_key}:{annotation_index}"
                rows.append(
                    {
                        "record_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                        "source_key": source_keys.get((split, image_path.name), ""),
                        "split": split,
                        "image": image_path.name,
                        "annotation_index": annotation_index,
                        "current_class": review_class,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "geometry": geometry,
                        "box_area_fraction": width * height,
                        "image_width": image_width,
                        "image_height": image_height,
                        "other_classes": other_classes,
                        "reviewer_1": "",
                        "decision_1": "",
                        "reviewer_2": "",
                        "decision_2": "",
                        "adjudicator": "",
                        "final_decision": "",
                        "notes": "",
                    }
                )
    return rows, boxes_by_image


def render_html(dataset: Path, output: Path, rows: list[dict[str, object]], boxes_by_image: dict[str, list[tuple[int, float, float, float, float, str]]], review_class: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_dataset = os.path.relpath(dataset, output.parent)
    cards = []
    for row in rows:
        image_key = f"{row['split']}/{row['image']}"
        overlays = []
        for index, (class_id, x_center, y_center, width, height, _) in enumerate(boxes_by_image[image_key], 1):
            left = (x_center - width / 2) * 100
            top = (y_center - height / 2) * 100
            kind = "target" if index == row["annotation_index"] else "context"
            overlays.append(
                f'<span class="box {kind}" style="left:{left:.5f}%;top:{top:.5f}%;width:{width * 100:.5f}%;height:{height * 100:.5f}%"><b>{class_id}:{index}</b></span>'
            )
        image_src = f"{relative_dataset}/{row['split']}/images/{row['image']}"
        cards.append(
            "<article>"
            f"<h2>{html.escape(str(row['record_id']))} - row decision: {DECISIONS}</h2>"
            f"<p>{html.escape(image_key)} | annotation {row['annotation_index']} | source {html.escape(str(row['source_key']))}</p>"
            f'<div class="image"><img loading="lazy" src="{html.escape(image_src)}">{"".join(overlays)}</div>'
            "</article>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Ontology review</title>
<style>
body{{font:14px system-ui;margin:20px;background:#f4f5f7;color:#17202a}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}
article{{background:white;border:1px solid #ccd1d1;border-radius:10px;padding:12px}}h1{{margin-bottom:4px}}h2{{font-size:14px;margin:0}}p{{overflow-wrap:anywhere}}
.image{{position:relative;display:inline-block;max-width:100%}}img{{display:block;max-width:100%;max-height:540px}}
.box{{position:absolute;box-sizing:border-box;border:2px solid #95a5a6;color:white;text-shadow:0 1px 2px black}}.box.target{{border:4px solid #e74c3c}}.box b{{background:rgba(0,0,0,.65)}}
</style></head><body><h1>Historical class {review_class} ontology review</h1>
<p>{len(rows)} annotations. Red is the row under review; grey boxes provide context. Enter decisions in the CSV, not in this HTML.</p><main>{''.join(cards)}</main></body></html>"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--review-class", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    manifest = args.manifest.resolve() if args.manifest else None
    rows, boxes_by_image = build_queue(dataset, args.review_class, load_source_keys(manifest))
    if not rows:
        raise SystemExit(f"No class {args.review_class} annotations found")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.html:
        render_html(dataset, args.html.resolve(), rows, boxes_by_image, args.review_class)
    print(f"Wrote {len(rows)} review rows to {args.output_csv}")


if __name__ == "__main__":
    main()
