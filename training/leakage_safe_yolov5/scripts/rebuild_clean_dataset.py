#!/usr/bin/env python3
"""Build a leakage-safe YOLOv5 dataset without modifying the source export.

The split unit is the pre-Roboflow source name (the part before ``.rf.``).
Sources are stratified by their set of class labels and assigned 80/10/10 to
train/valid/test. Training keeps distinct offline augmentations; validation
and test keep one deterministic representative per source. Exact image
duplicates are removed. HEIC files carrying a .jpg/.JPG suffix are converted
to real JPEG files with macOS ``sips``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("cigarette", "flame", "smoke")
HEIF_BRANDS = (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftyphevx", b"ftypmif1")
PHASH_SIZE = 32
_PHASH_POSITIONS = np.arange(PHASH_SIZE)
_PHASH_FREQUENCIES = np.arange(PHASH_SIZE)[:, None]
_DCT_MATRIX = np.cos(
    np.pi * (2 * _PHASH_POSITIONS + 1) * _PHASH_FREQUENCIES / (2 * PHASH_SIZE)
)
_DCT_MATRIX[0, :] *= 1 / np.sqrt(2)
_DCT_MATRIX *= np.sqrt(2 / PHASH_SIZE)


@dataclass(frozen=True)
class Record:
    original_split: str
    image: Path
    label: Path
    source_key: str
    image_sha256: str
    label_text: str
    label_signature: tuple[int, ...]
    annotation_count: int
    is_heif: bool
    perceptual_hash: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hamming_distance(left: int, right: int) -> int:
    """Return bit distance without requiring Python 3.10's int.bit_count()."""
    return bin(left ^ right).count("1")


def is_heif_disguised_as_jpeg(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(16)
    return any(brand in header for brand in HEIF_BRANDS)


def perceptual_hash(path: Path, is_heif: bool) -> int:
    readable_path = path
    temporary_path: str | None = None
    if is_heif:
        if shutil.which("sips") is None:
            raise RuntimeError("Perceptual hashing of HEIC input requires macOS 'sips'")
        descriptor, temporary_path = tempfile.mkstemp(suffix=".jpg")
        os.close(descriptor)
        completed = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(path), "--out", temporary_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"sips failed for {path}: {completed.stderr.strip()}")
        readable_path = Path(temporary_path)
    try:
        with Image.open(readable_path) as opened:
            pixels = np.asarray(
                ImageOps.exif_transpose(opened)
                .convert("L")
                .resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS),
                dtype=np.float64,
            )
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        low_frequency = (_DCT_MATRIX @ pixels @ _DCT_MATRIX.T)[:8, :8]
    if not np.isfinite(low_frequency).all():
        raise ValueError(f"non-finite perceptual hash coefficients for {path}")
    median = np.median(low_frequency.flat[1:])
    value = 0
    for bit in (low_frequency > median).flat:
        value = (value << 1) | int(bit)
    return value


def source_key(path: Path) -> str:
    return path.stem.split(".rf.", 1)[0]


def validate_label(path: Path) -> tuple[str, tuple[int, ...], int, Counter[int], list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    classes: set[int] = set()
    class_counts: Counter[int] = Counter()
    errors: list[str] = []
    annotations = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except (IndexError, ValueError):
            errors.append(f"{path.name}:{line_number}: cannot parse label")
            continue
        if class_id not in range(len(CLASS_NAMES)):
            errors.append(f"{path.name}:{line_number}: class {class_id} outside 0..2")
        is_box = len(fields) == 5
        is_polygon = len(coordinates) >= 6 and len(coordinates) % 2 == 0
        if not (is_box or is_polygon):
            errors.append(f"{path.name}:{line_number}: unsupported field count {len(fields)}")
        if any(value < 0 or value > 1 for value in coordinates):
            errors.append(f"{path.name}:{line_number}: coordinate outside [0, 1]")
        if is_box and (coordinates[2] <= 0 or coordinates[3] <= 0):
            errors.append(f"{path.name}:{line_number}: non-positive box size")
        classes.add(class_id)
        class_counts[class_id] += 1
        annotations += 1
    return text, tuple(sorted(classes)), annotations, class_counts, errors


def inventory(root: Path) -> tuple[dict[str, list[Record]], dict[str, object]]:
    records_by_source: dict[str, list[Record]] = defaultdict(list)
    errors: list[str] = []
    split_counts: dict[str, dict[str, int]] = {}
    label_kind_counts: Counter[str] = Counter()
    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.is_file() and not path.name.startswith("."))
        labels = {path.stem: path for path in label_dir.glob("*.txt") if not path.name.startswith(".")}
        missing = [path.name for path in images if path.stem not in labels]
        orphaned = sorted(set(labels) - {path.stem for path in images})
        if missing:
            errors.append(f"{split}: images without labels: {missing[:10]}")
        if orphaned:
            errors.append(f"{split}: labels without images: {orphaned[:10]}")
        empty_count = 0
        annotations = 0
        split_classes: Counter[int] = Counter()
        for image in images:
            label = labels.get(image.stem)
            if label is None:
                continue
            text, signature, count, class_counts, label_errors = validate_label(label)
            errors.extend(label_errors)
            empty_count += int(not text)
            annotations += count
            split_classes.update(class_counts)
            for line in text.splitlines():
                label_kind_counts["box" if len(line.split()) == 5 else "polygon"] += 1
            heif = is_heif_disguised_as_jpeg(image)
            records_by_source[source_key(image)].append(
                Record(
                    original_split=split,
                    image=image,
                    label=label,
                    source_key=source_key(image),
                    image_sha256=sha256_file(image),
                    label_text=text,
                    label_signature=signature,
                    annotation_count=count,
                    is_heif=heif,
                    perceptual_hash=perceptual_hash(image, heif),
                )
            )
        split_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_count,
            "annotations": annotations,
            **{f"class_{class_id}": split_classes[class_id] for class_id in range(len(CLASS_NAMES))},
        }
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors[:100]))
    profile = {
        "split_counts": split_counts,
        "unique_source_groups": len(records_by_source),
        "label_kinds": dict(label_kind_counts),
    }
    return records_by_source, profile


def merge_near_duplicate_sources(
    records_by_source: dict[str, list[Record]], threshold: int = 4
) -> tuple[dict[str, list[Record]], dict[str, object]]:
    keys = sorted(records_by_source)
    parent = {key: key for key in keys}
    sizes = {key: 1 for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]
        return True

    sources_by_hash: dict[int, set[str]] = defaultdict(set)
    for key, records in records_by_source.items():
        for record in records:
            sources_by_hash[record.perceptual_hash].add(key)
    unique_hashes = list(sources_by_hash)
    union_count = 0
    for source_set in sources_by_hash.values():
        ordered_sources = sorted(source_set)
        for other in ordered_sources[1:]:
            union_count += int(union(ordered_sources[0], other))
    for index, left_hash in enumerate(unique_hashes):
        for right_hash in unique_hashes[index + 1 :]:
            if hamming_distance(left_hash, right_hash) <= threshold:
                for left_source in sources_by_hash[left_hash]:
                    for right_source in sources_by_hash[right_hash]:
                        union_count += int(union(left_source, right_source))

    source_components: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        source_components[find(key)].append(key)
    records_by_component: dict[str, list[Record]] = {}
    component_sizes: Counter[int] = Counter()
    for source_keys in source_components.values():
        ordered_keys = sorted(source_keys)
        component_id = hashlib.sha256("\n".join(ordered_keys).encode()).hexdigest()[:16]
        records_by_component[component_id] = [
            record for key in ordered_keys for record in records_by_source[key]
        ]
        component_sizes[len(ordered_keys)] += 1
    profile = {
        "perceptual_hash_threshold": threshold,
        "perceptual_components": len(records_by_component),
        "source_groups_merged": len(keys) - len(records_by_component),
        "union_operations": union_count,
        "largest_component_sources": max(component_sizes, default=0),
        "component_size_distribution": {str(size): count for size, count in sorted(component_sizes.items())},
    }
    return records_by_component, profile


def assign_sources(records_by_source: dict[str, list[Record]], seed: str) -> dict[str, str]:
    strata: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for key, records in records_by_source.items():
        combined_signature = tuple(sorted({class_id for record in records for class_id in record.label_signature}))
        strata[combined_signature].append(key)
    assignment: dict[str, str] = {}
    for signature, keys in sorted(strata.items()):
        ordered = sorted(keys, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
        size = len(ordered)
        test_count = round(size * 0.10)
        valid_count = round(size * 0.10)
        if size >= 3:
            test_count = max(1, test_count)
            valid_count = max(1, valid_count)
        if test_count + valid_count > size:
            valid_count = max(0, size - test_count)
        for index, key in enumerate(ordered):
            if index < test_count:
                assignment[key] = "test"
            elif index < test_count + valid_count:
                assignment[key] = "valid"
            else:
                assignment[key] = "train"
    return assignment


def choose_records(records: list[Record], target_split: str) -> tuple[list[Record], list[tuple[Record, str]]]:
    preferred_order = {
        "train": ["train", "valid", "test"],
        "valid": ["valid", "test", "train"],
        "test": ["test", "valid", "train"],
    }[target_split]
    ordered = sorted(
        records,
        key=lambda item: (
            -item.annotation_count,
            preferred_order.index(item.original_split),
            item.image.name,
        ),
    )
    selected = ordered[:1]
    hashes = Counter(record.image_sha256 for record in records)
    removed = []
    for record in ordered[1:]:
        if hashes[record.image_sha256] > 1:
            same_hash_labels = {candidate.label_text for candidate in records if candidate.image_sha256 == record.image_sha256}
            reason = "exact_duplicate_label_conflict" if len(same_hash_labels) > 1 else "exact_duplicate_same_label"
        else:
            reason = f"extra_variant_or_near_duplicate_for_{target_split}_component"
        removed.append((record, reason))
    return selected, removed


def unique_output_name(record: Record, target_split: str, used: set[str]) -> str:
    suffix = ".jpg" if record.is_heif else record.image.suffix
    candidate = record.image.stem + suffix
    if candidate not in used:
        used.add(candidate)
        return candidate
    candidate = f"{record.original_split}__{record.image.stem}{suffix}"
    counter = 2
    while candidate in used:
        candidate = f"{record.original_split}_{counter}__{record.image.stem}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def materialize_image(record: Record, destination: Path) -> str:
    if record.is_heif:
        if shutil.which("sips") is None:
            raise RuntimeError("HEIC-to-JPEG conversion requires macOS 'sips'; install an equivalent converter on other systems")
        completed = subprocess.run(
            ["sips", "-s", "format", "jpeg", str(record.image), "--out", str(destination)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"sips failed for {record.image}: {completed.stderr.strip()}")
        return "converted_heic_to_jpeg"
    shutil.copy2(record.image, destination)
    return "copied"


def build(source_root: Path, output_root: Path, seed: str) -> dict[str, object]:
    if (output_root / "dataset").exists() or (output_root / "audit").exists():
        raise FileExistsError(f"Output already contains dataset/audit results; choose a fresh package path: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records_by_source, input_profile = inventory(source_root)
    records_by_component, perceptual_profile = merge_near_duplicate_sources(records_by_source)
    input_profile["perceptual_grouping"] = perceptual_profile
    assignment = assign_sources(records_by_component, seed)
    for split in SPLITS:
        (output_root / "dataset" / split / "images").mkdir(parents=True, exist_ok=False)
        (output_root / "dataset" / split / "labels").mkdir(parents=True, exist_ok=False)
    audit_dir = output_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=False)

    selected_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []
    used_names: dict[str, set[str]] = {split: set() for split in SPLITS}
    output_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    output_hashes: dict[str, set[str]] = {split: set() for split in SPLITS}
    output_sources: dict[str, set[str]] = {split: set() for split in SPLITS}

    for key in sorted(records_by_component):
        target_split = assignment[key]
        selected, removed = choose_records(records_by_component[key], target_split)
        for record, reason in removed:
            removed_rows.append(
                {
                    "source_key": key,
                    "original_source_key": record.source_key,
                    "original_split": record.original_split,
                    "assigned_split": target_split,
                    "image_name": record.image.name,
                    "image_sha256": record.image_sha256,
                    "reason": reason,
                }
            )
        for record in selected:
            image_name = unique_output_name(record, target_split, used_names[target_split])
            output_image = output_root / "dataset" / target_split / "images" / image_name
            output_label = output_root / "dataset" / target_split / "labels" / (Path(image_name).stem + ".txt")
            action = materialize_image(record, output_image)
            shutil.copy2(record.label, output_label)
            output_sha256 = sha256_file(output_image)
            output_hashes[target_split].add(output_sha256)
            output_sources[target_split].add(key)
            output_counts[target_split]["images"] += 1
            output_counts[target_split]["empty_labels"] += int(not record.label_text)
            output_counts[target_split]["heic_converted"] += int(record.is_heif)
            for line in record.label_text.splitlines():
                output_counts[target_split]["annotations"] += 1
                output_counts[target_split][f"class_{int(line.split()[0])}"] += 1
                output_counts[target_split]["box"] += int(len(line.split()) == 5)
                output_counts[target_split]["polygon"] += int(len(line.split()) != 5)
            selected_rows.append(
                {
                    "source_key": key,
                    "original_source_key": record.source_key,
                    "original_split": record.original_split,
                    "assigned_split": target_split,
                    "original_image": record.image.name,
                    "output_image": image_name,
                    "input_sha256": record.image_sha256,
                    "output_sha256": output_sha256,
                    "action": action,
                    "empty_label": int(not record.label_text),
                    "label_signature": "|".join(map(str, record.label_signature)),
                }
            )

    source_overlaps = {}
    hash_overlaps = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        source_overlaps[f"{left}_{right}"] = len(output_sources[left] & output_sources[right])
        hash_overlaps[f"{left}_{right}"] = len(output_hashes[left] & output_hashes[right])
    if any(source_overlaps.values()) or any(hash_overlaps.values()):
        raise AssertionError(f"Leakage check failed: sources={source_overlaps}, hashes={hash_overlaps}")

    data_yaml = """# Leakage-safe YOLOv5 dataset generated by rebuild_clean_dataset.py
path: ../dataset
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['cigarette', 'flame', 'smoke']
"""
    (output_root / "dataset" / "data.yaml").write_text(data_yaml, encoding="utf-8")

    with (audit_dir / "selected_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    with (audit_dir / "removed_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(removed_rows[0]))
        writer.writeheader()
        writer.writerows(removed_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_root.name,
        "split_seed": seed,
        "strategy": {
            "split_unit": "connected component of pre-Roboflow source name and perceptual hash distance <= 4",
            "ratios_by_source": {"train": 0.8, "valid": 0.1, "test": 0.1},
            "stratification": "set of class IDs present; empty labels form a negative-image stratum",
            "all_split_policy": "keep one deterministic, best-annotated representative per perceptual component",
            "exact_duplicate_policy": "remove repeated SHA-256 images; record label conflicts in the removal manifest",
        },
        "input": input_profile,
        "output": {split: dict(output_counts[split]) | {"sources": len(output_sources[split])} for split in SPLITS},
        "removed": dict(Counter(row["reason"] for row in removed_rows)),
        "checks": {
            "source_overlap": source_overlaps,
            "exact_hash_overlap": hash_overlaps,
            "selected_records": len(selected_rows),
            "removed_records": len(removed_rows),
        },
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Original YOLO dataset root")
    parser.add_argument("output", type=Path, help="New output package root (must not exist)")
    parser.add_argument("--seed", default="uris-yolo5-clean-v1")
    args = parser.parse_args()
    summary = build(args.source.resolve(), args.output.resolve(), args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
