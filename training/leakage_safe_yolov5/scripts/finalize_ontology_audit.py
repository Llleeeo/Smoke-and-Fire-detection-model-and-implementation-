#!/usr/bin/env python3
"""Validate adjudication outputs and freeze final ontology decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


FINAL_DECISIONS = {"flame", "smoke", "cigarette", "smoking_action", "invalid"}
CANONICAL_CLASS_IDS = {"cigarette": 0, "flame": 1, "smoke": 2}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preliminary_csv", type=Path)
    parser.add_argument("queue_csv", type=Path)
    parser.add_argument("adjudication_csv", type=Path)
    parser.add_argument("adjudication_backup", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    preliminary_fields, preliminary = read_csv(args.preliminary_csv)
    _, queue = read_csv(args.queue_csv)
    adjudication_fields, adjudications = read_csv(args.adjudication_csv)
    required = {"record_id", "adjudicator", "final_decision", "notes", "adjudicated_at"}
    if not required.issubset(adjudication_fields):
        raise SystemExit(f"Adjudication CSV is missing: {sorted(required - set(adjudication_fields))}")

    def unique_map(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            record_id = row["record_id"]
            if record_id in result:
                raise SystemExit(f"Duplicate record_id in {label}: {record_id}")
            result[record_id] = row
        return result

    preliminary_by_id = unique_map(preliminary, "preliminary CSV")
    queue_by_id = unique_map(queue, "adjudication queue")
    adjudication_by_id = unique_map(adjudications, "adjudication CSV")
    if set(queue_by_id) != set(adjudication_by_id):
        raise SystemExit("Adjudication results do not exactly match the frozen queue")
    if not set(adjudication_by_id).issubset(preliminary_by_id):
        raise SystemExit("Adjudication results contain IDs outside the preliminary audit")

    for record_id, row in adjudication_by_id.items():
        if row["final_decision"] not in FINAL_DECISIONS:
            raise SystemExit(f"Invalid final decision for {record_id}: {row['final_decision']!r}")
        for field in ("adjudicator", "notes", "adjudicated_at"):
            if not row[field].strip():
                raise SystemExit(f"Blank {field} for {record_id}")

    backup = json.loads(args.adjudication_backup.read_text(encoding="utf-8"))
    state = backup.get("state", {})
    answers = state.get("answers", {})
    if set(backup.get("recordIds", [])) != set(queue_by_id) or set(answers) != set(queue_by_id):
        raise SystemExit("JSON backup IDs do not exactly match the frozen queue")
    for record_id, row in adjudication_by_id.items():
        answer = answers[record_id]
        actual = (answer.get("decision", ""), answer.get("notes", ""), answer.get("adjudicatedAt", ""))
        expected = (row["final_decision"], row["notes"], row["adjudicated_at"])
        if actual != expected:
            raise SystemExit(f"CSV/JSON mismatch for {record_id}")
    if any(row["adjudicator"] != state.get("adjudicator", "") for row in adjudications):
        raise SystemExit("Adjudicator differs between CSV and JSON")

    final_rows: list[dict[str, str]] = []
    for source in preliminary:
        row = dict(source)
        record_id = row["record_id"]
        if record_id in adjudication_by_id:
            decision = adjudication_by_id[record_id]
            row["adjudicator"] = decision["adjudicator"]
            row["final_decision"] = decision["final_decision"]
            row["notes"] = decision["notes"]
            row["adjudicated_at"] = decision["adjudicated_at"]
            row["final_source"] = "adjudicated"
        else:
            if row["final_decision"] not in FINAL_DECISIONS:
                raise SystemExit(f"Unresolved preliminary decision for {record_id}: {row['final_decision']!r}")
            row["adjudicated_at"] = ""
            row["final_source"] = "reviewer1_accepted"
        final_rows.append(row)

    if len(final_rows) != 498 or len({row["record_id"] for row in final_rows}) != 498:
        raise SystemExit("Final audit must contain 498 unique records")
    if any(row["final_decision"] not in FINAL_DECISIONS for row in final_rows):
        raise SystemExit("Final audit contains an unresolved decision")

    final_fields = list(preliminary_fields)
    for field in ("adjudicated_at", "final_source"):
        if field not in final_fields:
            final_fields.append(field)

    canonical_rows: list[dict[str, str]] = []
    for row in final_rows:
        decision = row["final_decision"]
        class_id = CANONICAL_CLASS_IDS.get(decision)
        if class_id is None:
            action = "drop"
            class_value = ""
        elif int(row["current_class"]) == class_id:
            action = "keep"
            class_value = str(class_id)
        else:
            action = "relabel"
            class_value = str(class_id)
        canonical_rows.append(
            {
                "record_id": row["record_id"],
                "source_key": row["source_key"],
                "split": row["split"],
                "image": row["image"],
                "annotation_index": row["annotation_index"],
                "original_class_id": row["current_class"],
                "final_decision": decision,
                "canonical_class_id": class_value,
                "detector_action": action,
                "final_source": row["final_source"],
                "adjudicator": row["adjudicator"],
                "notes": row["notes"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "ontology_review_final.csv"
    canonical_path = args.output_dir / "ontology_canonical_labels.csv"
    summary_path = args.output_dir / "ONTOLOGY_AUDIT_FINAL_REPORT.md"
    write_csv(final_path, final_fields, final_rows)
    write_csv(canonical_path, list(canonical_rows[0]), canonical_rows)

    decision_counts = Counter(row["final_decision"] for row in final_rows)
    action_counts = Counter(row["detector_action"] for row in canonical_rows)
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in final_rows:
        split_counts[row["split"]][row["final_decision"]] += 1
    decisions = ("flame", "smoke", "cigarette", "smoking_action", "invalid")
    report = [
        "# Final ontology audit report",
        "",
        "Status: frozen; all required adjudications are complete.",
        "",
        "## Integrity gates",
        "",
        "- Final audit: 498/498 unique records with no blank or ambiguous final decisions.",
        f"- Adjudication: {len(adjudications)}/{len(queue)} frozen queue records complete.",
        "- Adjudication CSV and JSON backup match exactly.",
        "- Reviewer 1: yhm; Reviewer 2: hzp; adjudicator recorded as `两人一起`.",
        "- Prespecified agreement sample: 96/100 raw agreement; Cohen's kappa = 0.903.",
        "",
        "## Final decision distribution",
        "",
        "| Decision | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    report.extend(
        f"| {decision} | {decision_counts[decision]} | {decision_counts[decision] / len(final_rows):.1%} |"
        for decision in decisions
    )
    report.extend(
        [
            "",
            "## Detector actions for the audited historical class-1 boxes",
            "",
            "| Action | Count | Meaning |",
            "| --- | ---: | --- |",
            f"| keep | {action_counts['keep']} | Retain the original class ID. |",
            f"| relabel | {action_counts['relabel']} | Change to the canonical class ID. |",
            f"| drop | {action_counts['drop']} | Exclude `smoking_action` and `invalid` boxes. |",
            "",
            "Canonical mapping: `cigarette` = 0, `flame` = 1, `smoke` = 2. The image pool must remain unchanged when dropped boxes make an image empty.",
            "",
            "## Split-by-decision counts",
            "",
            "| Split | flame | smoke | cigarette | smoking_action | invalid | Total |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ("train", "valid", "test"):
        counts = split_counts[split]
        report.append(
            f"| {split} | {counts['flame']} | {counts['smoke']} | {counts['cigarette']} | "
            f"{counts['smoking_action']} | {counts['invalid']} | {sum(counts.values())} |"
        )
    report.extend(
        [
            "",
            "## Scope note",
            "",
            "These 498 records are the audited historical class-1 annotations. Existing class-0 and class-2 annotations are outside this queue and remain unchanged unless separately audited.",
            "",
        ]
    )
    summary_path.write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({
        "final_records": len(final_rows),
        "adjudicated_records": len(adjudications),
        "final_decisions": dict(decision_counts),
        "detector_actions": dict(action_counts),
        "outputs": [str(final_path), str(canonical_path), str(summary_path)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
