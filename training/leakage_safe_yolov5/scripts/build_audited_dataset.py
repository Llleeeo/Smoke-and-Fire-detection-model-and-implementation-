#!/usr/bin/env python3
"""Build and verify an independent YOLO dataset from frozen ontology decisions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = {0: "cigarette", 1: "flame", 2: "smoke"}
CANONICAL_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
EXCLUDED = {"smoking_action", "invalid"}
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path, relative_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=str):
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(root / relative)))
    return digest.hexdigest()


def parse_geometry(fields: list[str]) -> tuple[float, float, float, float, str]:
    values = [float(value) for value in fields[1:]]
    if len(values) == 4:
        return values[0], values[1], values[2], values[3], "box"
    if len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        return (left + right) / 2, (top + bottom) / 2, right - left, bottom - top, "polygon_bbox"
    raise ValueError(f"Unsupported YOLO geometry with {len(values)} coordinates")


def label_class_counts(root: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for split in SPLITS:
        for path in (root / split / "labels").glob("*.txt"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dataset", type=Path)
    parser.add_argument("final_audit_csv", type=Path)
    parser.add_argument("output_dataset", type=Path)
    args = parser.parse_args()

    source = args.source_dataset.resolve()
    output = args.output_dataset.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source dataset does not exist: {source}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    with args.final_audit_csv.open(newline="", encoding="utf-8-sig") as handle:
        audit = list(csv.DictReader(handle))
    if len(audit) != 498 or len({row["record_id"] for row in audit}) != 498:
        raise SystemExit("Final audit must contain 498 unique records")
    if any(row["final_decision"] == "ambiguous" or not row["final_decision"] for row in audit):
        raise SystemExit("Final audit contains unresolved decisions")

    operations: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    planned = Counter()
    for row in audit:
        decision = row["final_decision"]
        if decision in CANONICAL_IDS:
            target_class = CANONICAL_IDS[decision]
            action = "keep" if target_class == int(row["current_class"]) else "relabel"
        elif decision in EXCLUDED:
            target_class = None
            action = "drop"
        else:
            raise SystemExit(f"Unknown final decision: {decision!r}")
        key = (row["split"], Path(row["image"]).stem)
        annotation_index = int(row["annotation_index"])
        if annotation_index in operations[key]:
            raise SystemExit(f"Duplicate annotation operation: {key} line {annotation_index}")
        row = dict(row)
        row["computed_action"] = action
        row["target_class"] = "" if target_class is None else str(target_class)
        operations[key][annotation_index] = row
        planned[action] += 1
    if planned != Counter({"keep": 371, "relabel": 95, "drop": 32}):
        raise SystemExit(f"Frozen action counts changed unexpectedly: {dict(planned)}")

    source_image_paths: list[Path] = []
    source_label_paths: list[Path] = []
    for split in SPLITS:
        image_dir, label_dir = source / split / "images", source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise SystemExit(f"Missing split directories for {split}")
        source_image_paths.extend(path.relative_to(source) for path in image_dir.iterdir() if path.is_file())
        source_label_paths.extend(path.relative_to(source) for path in label_dir.glob("*.txt"))
    if {path.with_suffix("") for path in source_image_paths} != {
        Path(path.parts[0], "images", path.stem) for path in source_label_paths
    }:
        raise SystemExit("Image and label stem sets differ in the source dataset")

    source_image_fingerprint = tree_fingerprint(source, source_image_paths)
    before_counts = label_class_counts(source)
    source_empty = sum(not (source / path).read_text(encoding="utf-8").strip() for path in source_label_paths)

    temp_root = Path(tempfile.mkdtemp(prefix=f".{output.name}_tmp_", dir=output.parent))
    applied = Counter()
    changed_records: list[dict[str, str]] = []
    try:
        for split in SPLITS:
            (temp_root / split / "images").mkdir(parents=True)
            (temp_root / split / "labels").mkdir(parents=True)
        for relative in source_image_paths:
            destination = temp_root / relative
            shutil.copy2(source / relative, destination)

        seen_operations: set[str] = set()
        for relative in source_label_paths:
            split, _, filename = relative.parts
            key = (split, Path(filename).stem)
            source_lines = (source / relative).read_text(encoding="utf-8").splitlines()
            output_lines: list[str] = []
            for annotation_index, line in enumerate(source_lines, 1):
                fields = line.split()
                operation = operations.get(key, {}).get(annotation_index)
                if operation is None:
                    output_lines.append(line)
                    continue
                if int(fields[0]) != int(operation["current_class"]):
                    raise SystemExit(f"Class mismatch: {relative}:{annotation_index}")
                x, y, width, height, geometry = parse_geometry(fields)
                expected = tuple(float(operation[name]) for name in ("x_center", "y_center", "width", "height"))
                actual = (x, y, width, height)
                if geometry != operation["geometry"] or any(abs(a - b) > 1e-9 for a, b in zip(actual, expected)):
                    raise SystemExit(f"Geometry mismatch: {relative}:{annotation_index}")
                action = operation["computed_action"]
                if action == "keep":
                    output_lines.append(line)
                elif action == "relabel":
                    output_lines.append(" ".join([operation["target_class"], *fields[1:]]))
                elif action != "drop":
                    raise AssertionError(action)
                applied[action] += 1
                seen_operations.add(operation["record_id"])
                changed_records.append(
                    {
                        "record_id": operation["record_id"],
                        "split": split,
                        "image": operation["image"],
                        "annotation_index": str(annotation_index),
                        "original_class_id": operation["current_class"],
                        "final_decision": operation["final_decision"],
                        "canonical_class_id": operation["target_class"],
                        "detector_action": action,
                    }
                )
            destination = temp_root / relative
            destination.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

        if len(seen_operations) != 498 or applied != planned:
            missing = {row["record_id"] for row in audit} - seen_operations
            raise SystemExit(f"Operation coverage mismatch; applied={dict(applied)}, missing={sorted(missing)[:5]}")

        data_yaml = (source / "data.yaml").read_text(encoding="utf-8")
        data_yaml = data_yaml.replace("path: ../dataset\n", "path: ../dataset_audited\n", 1)
        (temp_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

        output_image_paths = [path.relative_to(temp_root) for split in SPLITS for path in (temp_root / split / "images").iterdir() if path.is_file()]
        output_label_paths = [path.relative_to(temp_root) for split in SPLITS for path in (temp_root / split / "labels").glob("*.txt")]
        if set(output_image_paths) != set(source_image_paths):
            raise SystemExit("Output image set differs from source")
        if set(output_label_paths) != set(source_label_paths):
            raise SystemExit("Output label-file set differs from source")
        output_image_fingerprint = tree_fingerprint(temp_root, output_image_paths)
        if output_image_fingerprint != source_image_fingerprint:
            raise SystemExit("Copied image content differs from source")

        after_counts = label_class_counts(temp_root)
        output_empty = sum(not (temp_root / path).read_text(encoding="utf-8").strip() for path in output_label_paths)
        before_total, after_total = sum(before_counts.values()), sum(after_counts.values())
        if before_total - after_total != planned["drop"]:
            raise SystemExit("Total label count did not decrease by the drop count")
        expected_counts = Counter(before_counts)
        expected_counts[1] -= planned["relabel"] + planned["drop"]
        expected_counts[0] += planned["relabel"]
        if after_counts != expected_counts:
            raise SystemExit(f"Final class counts differ from expected: {dict(after_counts)} vs {dict(expected_counts)}")

        manifest = {
            "status": "verified",
            "source_dataset": str(source),
            "output_dataset": str(output),
            "final_audit_csv": str(args.final_audit_csv.resolve()),
            "splits": {
                split: {
                    "images": sum(path.parts[0] == split for path in output_image_paths),
                    "label_files": sum(path.parts[0] == split for path in output_label_paths),
                }
                for split in SPLITS
            },
            "operations": dict(planned),
            "class_counts_before": {CLASS_NAMES[key]: before_counts[key] for key in sorted(CLASS_NAMES)},
            "class_counts_after": {CLASS_NAMES[key]: after_counts[key] for key in sorted(CLASS_NAMES)},
            "total_boxes_before": before_total,
            "total_boxes_after": after_total,
            "empty_label_files_before": source_empty,
            "empty_label_files_after": output_empty,
            "image_tree_sha256": source_image_fingerprint,
            "images_byte_identical": True,
            "source_cache_files_copied": False,
        }
        (temp_root / "AUDIT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log_fields = list(changed_records[0])
        with (temp_root / "ONTOLOGY_TRANSFORM_LOG.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=log_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(changed_records)
        report = [
            "# Audited dataset transformation report",
            "",
            "Status: verified.",
            "",
            f"- Images: {len(output_image_paths)} copied byte-for-byte; split membership unchanged.",
            f"- Label files: {len(output_label_paths)} retained, including empty negative-label files.",
            f"- Frozen operations: keep {planned['keep']}, relabel {planned['relabel']}, drop {planned['drop']}.",
            f"- Total boxes: {before_total} -> {after_total}.",
            f"- Empty label files: {source_empty} -> {output_empty}.",
            "- Existing class-0 and class-2 annotations were not otherwise modified.",
            "- YOLO cache files were intentionally excluded and will be regenerated by training.",
            "",
            "| Class | Before | After | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
        for class_id, name in CLASS_NAMES.items():
            report.append(f"| {name} ({class_id}) | {before_counts[class_id]} | {after_counts[class_id]} | {after_counts[class_id] - before_counts[class_id]:+d} |")
        report.extend(["", f"Image tree SHA-256: `{source_image_fingerprint}`", ""])
        (temp_root / "TRANSFORMATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
        temp_root.rename(output)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
