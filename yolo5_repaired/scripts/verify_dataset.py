#!/usr/bin/env python3
"""Verify pairing, label validity, source isolation, hashes, and image decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "valid", "test")
PHASH_SIZE = 32
_POSITIONS = np.arange(PHASH_SIZE)
_FREQUENCIES = np.arange(PHASH_SIZE)[:, None]
_DCT = np.cos(np.pi * (2 * _POSITIONS + 1) * _FREQUENCIES / (2 * PHASH_SIZE))
_DCT[0, :] *= 1 / np.sqrt(2)
_DCT *= np.sqrt(2 / PHASH_SIZE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_key(path: Path) -> str:
    return path.stem.split(".rf.", 1)[0]


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as opened:
        pixels = np.asarray(
            ImageOps.exif_transpose(opened)
            .convert("L")
            .resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS),
            dtype=float,
        )
    low_frequency = (_DCT @ pixels @ _DCT.T)[:8, :8]
    median = np.median(low_frequency.flat[1:])
    value = 0
    for bit in (low_frequency > median).flat:
        value = (value << 1) | int(bit)
    return value


def verify(root: Path) -> dict[str, object]:
    result: dict[str, object] = {"splits": {}, "overlaps": {}, "passed": False}
    sources: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    perceptual_hashes: dict[str, list[tuple[str, int]]] = {}
    failures: list[str] = []
    for split in SPLITS:
        images = sorted(path for path in (root / split / "images").iterdir() if path.is_file() and not path.name.startswith("."))
        labels = {path.stem: path for path in (root / split / "labels").glob("*.txt")}
        image_stems = {path.stem for path in images}
        if image_stems != set(labels):
            failures.append(f"{split}: image/label stem mismatch")
        counts = Counter(images=len(images), labels=len(labels))
        split_sources: set[str] = set()
        split_hashes: set[str] = set()
        split_perceptual_hashes: list[tuple[str, int]] = []
        for image in images:
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as error:
                failures.append(f"{split}: cannot decode {image.name}: {error}")
            digest = sha256_file(image)
            if digest in split_hashes:
                failures.append(f"{split}: repeated exact image hash {digest}")
            split_hashes.add(digest)
            split_perceptual_hashes.append((image.name, perceptual_hash(image)))
            split_sources.add(source_key(image))
            text = labels[image.stem].read_text(encoding="utf-8", errors="replace").strip()
            counts["empty_labels"] += int(not text)
            for line_number, line in enumerate(text.splitlines(), 1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except (IndexError, ValueError):
                    failures.append(f"{split}: malformed {labels[image.stem].name}:{line_number}")
                    continue
                is_box = len(fields) == 5
                is_polygon = len(coordinates) >= 6 and len(coordinates) % 2 == 0
                if class_id not in range(3) or not (is_box or is_polygon) or any(value < 0 or value > 1 for value in coordinates):
                    failures.append(f"{split}: invalid {labels[image.stem].name}:{line_number}")
                counts["annotations"] += 1
                counts[f"class_{class_id}"] += 1
                counts["box"] += int(is_box)
                counts["polygon"] += int(is_polygon)
        sources[split] = split_sources
        hashes[split] = split_hashes
        perceptual_hashes[split] = split_perceptual_hashes
        result["splits"][split] = dict(counts) | {"sources": len(split_sources)}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        result["overlaps"][f"{left}_{right}"] = {
            "source_names": len(sources[left] & sources[right]),
            "exact_image_hashes": len(hashes[left] & hashes[right]),
            "perceptual_hash_distance_le_4": sum(
                (left_hash ^ right_hash).bit_count() <= 4
                for _, left_hash in perceptual_hashes[left]
                for _, right_hash in perceptual_hashes[right]
            ),
        }
        if (
            sources[left] & sources[right]
            or hashes[left] & hashes[right]
            or result["overlaps"][f"{left}_{right}"]["perceptual_hash_distance_le_4"]
        ):
            failures.append(f"{left}/{right}: leakage remains")
    result["failures"] = failures
    result["passed"] = not failures
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = verify(args.dataset.resolve())
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
