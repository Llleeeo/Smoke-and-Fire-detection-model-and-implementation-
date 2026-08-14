#!/usr/bin/env python3
"""Build seed-specific C/D training sets with controlled offline-variant leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from rebuild_clean_dataset import hamming_distance, is_heif_disguised_as_jpeg, perceptual_hash


CANONICAL_IDS = {"cigarette": 0, "flame": 1, "smoke": 2}
EXCLUDED = {"smoking_action", "invalid"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def label_path(root: Path, split: str, image: str) -> Path:
    return root / split / "labels" / f"{Path(image).stem}.txt"


def image_path(root: Path, split: str, image: str) -> Path:
    return root / split / "images" / image


def raw_class_sequence(raw: Path, row: dict[str, str]) -> tuple[int, ...]:
    image = row.get("image_name") or row.get("original_image")
    return tuple(
        int(line.split()[0])
        for line in label_path(raw, row["original_split"], image).read_text(encoding="utf-8").splitlines()
    )


def signature(path: Path) -> tuple[int, ...]:
    return tuple(sorted({int(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines()}))


def convert_geometry(fields: list[str]) -> list[str]:
    values = [float(value) for value in fields[1:]]
    if len(values) == 4:
        return fields[1:]
    if len(values) < 6 or len(values) % 2:
        raise ValueError(f"Unsupported YOLO geometry: {' '.join(fields)}")
    xs, ys = values[0::2], values[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    box = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    return [format(value, ".15g") for value in box]


def render_injected_label(
    raw_label: Path,
    representative_image: str,
    decisions: dict[tuple[str, int], str],
    audited: bool,
) -> str:
    output: list[str] = []
    for annotation_index, line in enumerate(raw_label.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        class_id = int(fields[0])
        if audited and class_id == 1:
            decision = decisions.get((representative_image, annotation_index))
            if decision is None:
                raise ValueError(f"Missing audit decision: {representative_image}:{annotation_index}")
            if decision in EXCLUDED:
                continue
            class_id = CANONICAL_IDS[decision]
        output.append(" ".join([str(class_id), *convert_geometry(fields)]))
    return "\n".join(output) + ("\n" if output else "")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dataset", type=Path)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    raw = args.raw_dataset.resolve()
    package = args.package_root.resolve()
    original = package / "dataset_detect_original"
    audited = package / "dataset_detect_audited"
    selected = read_csv(package / "audit" / "selected_manifest.csv")
    removed = read_csv(package / "audit" / "removed_manifest.csv")
    audit = read_csv(package / "audit" / "ontology_review_final.csv")
    decisions = {
        (row["image"], int(row["annotation_index"])): row["final_decision"]
        for row in audit
        if row["split"] == "test"
    }

    selected_test = {row["source_key"]: row for row in selected if row["assigned_split"] == "test"}
    removed_by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in removed:
        if (
            row["assigned_split"] == "test"
            and row["reason"] == "extra_variant_or_near_duplicate_for_test_component"
            and row["source_key"] in selected_test
        ):
            removed_by_component[row["source_key"]].append(row)

    eligible: dict[str, list[dict[str, object]]] = {}
    for component, candidates in removed_by_component.items():
        representative = selected_test[component]
        representative_sequence = raw_class_sequence(raw, representative)
        representative_hash = perceptual_hash(
            image_path(audited, "test", representative["output_image"]), False
        )
        accepted: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_image = image_path(raw, candidate["original_split"], candidate["image_name"])
            if (
                candidate["original_source_key"] != representative["original_source_key"]
                or raw_class_sequence(raw, candidate) != representative_sequence
                or candidate["image_sha256"] == representative["output_sha256"]
                or is_heif_disguised_as_jpeg(candidate_image)
            ):
                continue
            accepted.append(
                {
                    "row": candidate,
                    "phash_distance": hamming_distance(
                        representative_hash, perceptual_hash(candidate_image, False)
                    ),
                }
            )
        if accepted:
            eligible[component] = accepted
    if len(eligible) != 41:
        raise SystemExit(f"Expected 41 strictly eligible test components, found {len(eligible)}")

    train_rows = [row for row in selected if row["assigned_split"] == "train"]
    train_hashes = {
        file_sha256(image_path(original, "train", row["output_image"])) for row in train_rows
    }
    replacement_strata: dict[tuple[tuple[int, ...], tuple[int, ...]], list[dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        pair = (
            signature(label_path(original, "train", row["output_image"])),
            signature(label_path(audited, "train", row["output_image"])),
        )
        replacement_strata[pair].append(row)

    for seed in args.seeds:
        final_original = package / f"dataset_leaked_original_seed{seed}"
        final_audited = package / f"dataset_leaked_audited_seed{seed}"
        if final_original.exists() or final_audited.exists():
            raise SystemExit(f"Refusing to overwrite seed {seed} leakage output")

        injections: list[dict[str, object]] = []
        for component, candidates in sorted(eligible.items()):
            selected_candidate = min(
                candidates,
                key=lambda item: hashlib.sha256(
                    f"{seed}:inject:{component}:{item['row']['image_name']}".encode()
                ).hexdigest(),
            )
            representative = selected_test[component]
            candidate = selected_candidate["row"]
            raw_label = label_path(raw, candidate["original_split"], candidate["image_name"])
            original_text = render_injected_label(raw_label, representative["output_image"], decisions, False)
            audited_text = render_injected_label(raw_label, representative["output_image"], decisions, True)
            pair = (
                tuple(sorted({int(line.split()[0]) for line in original_text.splitlines()})),
                tuple(sorted({int(line.split()[0]) for line in audited_text.splitlines()})),
            )
            if candidate["image_sha256"] in train_hashes:
                raise SystemExit(f"Injected candidate already exists byte-identically in clean train: {candidate['image_name']}")
            injections.append(
                {
                    "component": component,
                    "representative": representative,
                    "candidate": candidate,
                    "phash_distance": selected_candidate["phash_distance"],
                    "original_text": original_text,
                    "audited_text": audited_text,
                    "signature_pair": pair,
                }
            )

        needed = Counter(item["signature_pair"] for item in injections)
        replacements: dict[tuple[tuple[int, ...], tuple[int, ...]], list[dict[str, str]]] = {}
        for pair, count in needed.items():
            ranked = sorted(
                replacement_strata[pair],
                key=lambda row: hashlib.sha256(
                    f"{seed}:replace:{pair}:{row['source_key']}:{row['output_image']}".encode()
                ).hexdigest(),
            )
            if len(ranked) < count:
                raise SystemExit(f"Insufficient replacement candidates for {pair}: {len(ranked)} < {count}")
            replacements[pair] = ranked[:count]
        replacement_queue = {pair: list(rows) for pair, rows in replacements.items()}
        for item in injections:
            item["replacement"] = replacement_queue[item["signature_pair"]].pop()

        temp_root = Path(tempfile.mkdtemp(prefix=f".leakage_seed{seed}_", dir=package))
        temp_original, temp_audited = temp_root / "original", temp_root / "audited"
        try:
            for root in (temp_original, temp_audited):
                (root / "train" / "images").mkdir(parents=True)
                (root / "train" / "labels").mkdir(parents=True)
            removed_names = {item["replacement"]["output_image"] for item in injections}
            if len(removed_names) != len(injections):
                raise SystemExit("Replacement image selected more than once")
            for source_root, destination_root in ((original, temp_original), (audited, temp_audited)):
                for image in (source_root / "train" / "images").iterdir():
                    if image.is_file() and not image.name.startswith(".") and image.name not in removed_names:
                        shutil.copy2(image, destination_root / "train" / "images" / image.name)
                        shutil.copy2(
                            label_path(source_root, "train", image.name),
                            label_path(destination_root, "train", image.name),
                        )

            manifest_rows: list[dict[str, str]] = []
            injected_names: set[str] = set()
            for item in injections:
                candidate = item["candidate"]
                representative = item["representative"]
                injected_name = f"leak_{item['component']}_{candidate['image_name']}"
                if injected_name in injected_names:
                    raise SystemExit(f"Injected filename collision: {injected_name}")
                injected_names.add(injected_name)
                source_image = image_path(raw, candidate["original_split"], candidate["image_name"])
                for destination_root, text in (
                    (temp_original, item["original_text"]),
                    (temp_audited, item["audited_text"]),
                ):
                    shutil.copy2(source_image, destination_root / "train" / "images" / injected_name)
                    label_path(destination_root, "train", injected_name).write_text(text, encoding="utf-8")
                replacement = item["replacement"]
                manifest_rows.append(
                    {
                        "seed": str(seed),
                        "test_component": item["component"],
                        "test_representative": representative["output_image"],
                        "injected_raw_split": candidate["original_split"],
                        "injected_raw_image": candidate["image_name"],
                        "injected_output_image": injected_name,
                        "injected_sha256": candidate["image_sha256"],
                        "phash_distance_to_test_representative": str(item["phash_distance"]),
                        "original_signature": "|".join(map(str, item["signature_pair"][0])),
                        "audited_signature": "|".join(map(str, item["signature_pair"][1])),
                        "replacement_component": replacement["source_key"],
                        "replacement_image": replacement["output_image"],
                    }
                )

            original_images = {
                path.name: file_sha256(path)
                for path in (temp_original / "train" / "images").iterdir()
                if path.is_file()
            }
            audited_images = {
                path.name: file_sha256(path)
                for path in (temp_audited / "train" / "images").iterdir()
                if path.is_file()
            }
            if original_images != audited_images or len(original_images) != 1511:
                raise SystemExit("C/D image identities differ or training size is not 1511")
            test_hashes = {
                file_sha256(path)
                for path in (audited / "test" / "images").iterdir()
                if path.is_file() and not path.name.startswith(".")
            }
            if set(original_images.values()) & test_hashes:
                raise SystemExit("Exact test image bytes entered controlled-leakage training set")
            for root in (temp_original, temp_audited):
                images = {path.stem for path in (root / "train" / "images").iterdir() if path.is_file()}
                labels = {path.stem for path in (root / "train" / "labels").glob("*.txt")}
                if images != labels:
                    raise SystemExit("Generated leakage dataset has image/label mismatch")
                if any(len(line.split()) != 5 for path in (root / "train" / "labels").glob("*.txt") for line in path.read_text().splitlines()):
                    raise SystemExit("Generated leakage dataset is not detection-only")

            for root in (temp_original, temp_audited):
                with (root / "CONTROLLED_LEAKAGE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(manifest_rows)
                summary = {
                    "status": "verified",
                    "seed": seed,
                    "dose": "100% of strictly eligible test components",
                    "eligible_test_components": len(injections),
                    "all_test_components": len(selected_test),
                    "injected_images": len(injections),
                    "replacement_images_removed": len(removed_names),
                    "training_images": len(original_images),
                    "exact_test_hashes_in_train": 0,
                    "image_pool_identical_between_C_and_D": True,
                    "selection_unit": "same original source, same label sequence, distinct image hash",
                    "replacement_stratum": "paired original/audited class signature",
                }
                (root / "CONTROLLED_LEAKAGE_SUMMARY.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            temp_original.rename(final_original)
            temp_audited.rename(final_audited)
            temp_root.rmdir()
        except BaseException:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise
        print(f"seed {seed}: built C/D with {len(injections)} injections and {len(removed_names)} replacements")


if __name__ == "__main__":
    main()
