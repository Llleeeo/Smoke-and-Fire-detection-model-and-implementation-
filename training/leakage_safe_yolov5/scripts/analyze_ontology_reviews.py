#!/usr/bin/env python3
"""Validate, merge, and summarize two ontology-review result files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


DECISIONS = ("flame", "smoke", "cigarette", "smoking_action", "invalid", "ambiguous")
REQUIRED_RESULT_FIELDS = {
    "record_id",
    "reviewer_slot",
    "reviewer",
    "decision",
    "notes",
    "reviewed_at",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_results(path: Path, expected_slot: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing_fields = REQUIRED_RESULT_FIELDS - set(rows[0])
    if missing_fields:
        raise ValueError(f"Missing fields in {path}: {sorted(missing_fields)}")
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate record_id in {path}")
    for row in rows:
        if row["reviewer_slot"] != expected_slot:
            raise ValueError(f"Unexpected reviewer slot in {path}: {row['reviewer_slot']}")
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Invalid decision in {path}: {row['record_id']}={row['decision']!r}")
        if not row["reviewer"].strip():
            raise ValueError(f"Blank reviewer in {path}: {row['record_id']}")
        if not row["reviewed_at"].strip():
            raise ValueError(f"Blank reviewed_at in {path}: {row['record_id']}")
    return rows


def validate_backup(path: Path, results: list[dict[str, str]], expected_slot: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("slot") != expected_slot:
        raise ValueError(f"Unexpected backup slot in {path}: {payload.get('slot')}")
    answers = payload.get("state", {}).get("answers", {})
    result_map = {row["record_id"]: row for row in results}
    if set(answers) != set(result_map):
        raise ValueError(f"Backup/result record mismatch in {path}")
    for record_id, row in result_map.items():
        answer = answers[record_id]
        observed = (answer.get("decision", ""), answer.get("notes", ""), answer.get("reviewedAt", ""))
        expected = (row["decision"], row["notes"], row["reviewed_at"])
        if observed != expected:
            raise ValueError(f"Backup/result content mismatch in {path}: {record_id}")


def elapsed_seconds(rows: list[dict[str, str]]) -> float:
    times = [datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00")) for row in rows]
    return (max(times) - min(times)).total_seconds()


def cohen_kappa(r1: dict[str, dict[str, str]], r2: dict[str, dict[str, str]], common: list[str]) -> tuple[float, float, float]:
    total = len(common)
    observed = sum(r1[key]["decision"] == r2[key]["decision"] for key in common) / total
    counts_1 = Counter(r1[key]["decision"] for key in common)
    counts_2 = Counter(r2[key]["decision"] for key in common)
    expected = sum((counts_1[label] / total) * (counts_2[label] / total) for label in DECISIONS)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return observed, expected, kappa


def markdown_table(matrix: Counter[tuple[str, str]]) -> str:
    header = "| Reviewer 1 \\ Reviewer 2 | " + " | ".join(DECISIONS) + " |\n"
    divider = "| --- | " + " | ".join("---:" for _ in DECISIONS) + " |\n"
    body = ""
    for left in DECISIONS:
        body += f"| {left} | " + " | ".join(str(matrix[left, right]) for right in DECISIONS) + " |\n"
    return header + divider + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("master_csv", type=Path)
    parser.add_argument("reviewer1_csv", type=Path)
    parser.add_argument("reviewer2_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reviewer1-name", default="Reviewer 1")
    parser.add_argument("--reviewer2-name", default="Reviewer 2")
    parser.add_argument("--reviewer2-queue", type=Path)
    parser.add_argument("--reviewer1-backup", type=Path)
    parser.add_argument("--reviewer2-backup", type=Path)
    args = parser.parse_args()

    master = read_csv(args.master_csv)
    if len(master) != 498:
        raise ValueError(f"Expected 498 master rows, found {len(master)}")
    master_map = {row["record_id"]: row for row in master}
    if len(master_map) != len(master):
        raise ValueError("Master record_id is not unique")

    rows_1 = validate_results(args.reviewer1_csv, "reviewer1")
    rows_2 = validate_results(args.reviewer2_csv, "reviewer2")
    map_1 = {row["record_id"]: row for row in rows_1}
    map_2 = {row["record_id"]: row for row in rows_2}
    if set(map_1) != set(master_map):
        raise ValueError("Reviewer 1 records do not exactly match the master queue")
    if not set(map_2) <= set(master_map):
        raise ValueError("Reviewer 2 contains unknown records")
    if args.reviewer2_queue:
        expected_2 = {row["record_id"] for row in read_csv(args.reviewer2_queue)}
        if set(map_2) != expected_2:
            raise ValueError("Reviewer 2 results do not match the prespecified blind queue")
    if args.reviewer1_backup:
        validate_backup(args.reviewer1_backup, rows_1, "reviewer1")
    if args.reviewer2_backup:
        validate_backup(args.reviewer2_backup, rows_2, "reviewer2")

    common = sorted(map_2)
    observed, expected, kappa = cohen_kappa(map_1, map_2, common)
    confusion = Counter((map_1[key]["decision"], map_2[key]["decision"]) for key in common)

    merged = []
    disagreements = []
    adjudication = []
    for source in master:
        record_id = source["record_id"]
        one = map_1[record_id]
        two = map_2.get(record_id)
        reasons = []
        if two and one["decision"] != two["decision"]:
            reasons.append("reviewer_disagreement")
        if one["decision"] == "ambiguous" or (two and two["decision"] == "ambiguous"):
            reasons.append("ambiguous")
        if one["decision"] in {"invalid", "ambiguous"} and not one["notes"].strip():
            reasons.append("reviewer1_missing_required_note")
        if two and two["decision"] in {"invalid", "ambiguous"} and not two["notes"].strip():
            reasons.append("reviewer2_missing_required_note")
        combined_notes = " | ".join(
            item for item in (
                f"R1: {one['notes'].strip()}" if one["notes"].strip() else "",
                f"R2: {two['notes'].strip()}" if two and two["notes"].strip() else "",
            ) if item
        )
        final_decision = "" if reasons else one["decision"]
        row = dict(source)
        row.update(
            {
                "reviewer_1": args.reviewer1_name,
                "decision_1": one["decision"],
                "reviewed_at_1": one["reviewed_at"],
                "notes_1": one["notes"],
                "reviewer_2": args.reviewer2_name if two else "",
                "decision_2": two["decision"] if two else "",
                "reviewed_at_2": two["reviewed_at"] if two else "",
                "notes_2": two["notes"] if two else "",
                "adjudication_reason": "|".join(reasons),
                "adjudicator": "",
                "final_decision": final_decision,
                "notes": combined_notes,
            }
        )
        merged.append(row)
        if two and one["decision"] != two["decision"]:
            disagreements.append(row)
        if reasons:
            adjudication.append(row)

    fields = list(master[0])
    for field in (
        "reviewed_at_1",
        "notes_1",
        "reviewed_at_2",
        "notes_2",
        "adjudication_reason",
    ):
        if field not in fields:
            fields.append(field)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ontology_review_merged_preliminary.csv", merged, fields)
    write_csv(args.output_dir / "ontology_review_disagreements.csv", disagreements, fields)
    write_csv(args.output_dir / "ontology_adjudication_queue.csv", adjudication, fields)

    split_counts = Counter((master_map[key]["split"], map_1[key]["decision"]) for key in master_map)
    decision_counts = Counter(row["decision"] for row in rows_1)
    missing_notes_1 = sum(row["decision"] in {"invalid", "ambiguous"} and not row["notes"].strip() for row in rows_1)
    missing_notes_2 = sum(row["decision"] in {"invalid", "ambiguous"} and not row["notes"].strip() for row in rows_2)
    summary = {
        "master_rows": len(master),
        "reviewer1_rows": len(rows_1),
        "reviewer2_rows": len(rows_2),
        "reviewer1_name": args.reviewer1_name,
        "reviewer2_name": args.reviewer2_name,
        "reviewer1_decisions": dict(decision_counts),
        "reviewer1_missing_required_notes": missing_notes_1,
        "reviewer2_missing_required_notes": missing_notes_2,
        "agreement_count": round(observed * len(common)),
        "agreement_total": len(common),
        "raw_agreement": observed,
        "chance_agreement": expected,
        "cohen_kappa": kappa,
        "disagreement_count": len(disagreements),
        "adjudication_queue_count": len(adjudication),
        "reviewer1_elapsed_seconds": elapsed_seconds(rows_1),
        "reviewer2_elapsed_seconds": elapsed_seconds(rows_2),
    }
    (args.output_dir / "ontology_review_agreement.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Ontology review quality report",
        "",
        "Status: preliminary; adjudication is still required.",
        "",
        "## Completeness and integrity",
        "",
        f"- Master queue: {len(master)} unique annotation records.",
        f"- {args.reviewer1_name}: {len(rows_1)}/{len(master)} complete; all decisions use the allowed vocabulary.",
        f"- {args.reviewer2_name}: {len(rows_2)}/{len(rows_2)} complete on the prespecified blind sample; all record IDs match the master queue.",
        "- Both JSON backups exactly match their exported CSV files." if args.reviewer1_backup and args.reviewer2_backup else "- Backup equivalence was not checked.",
        f"- Missing mandatory notes: Reviewer 1 = {missing_notes_1}; Reviewer 2 = {missing_notes_2}.",
        "",
        "## Reviewer 1 outcome distribution",
        "",
        "| Decision | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for decision in DECISIONS:
        count = decision_counts[decision]
        lines.append(f"| {decision} | {count} | {count / len(rows_1):.1%} |")
    lines.extend(
        [
            "",
            "## Agreement",
            "",
            f"- Raw agreement: {round(observed * len(common))}/{len(common)} = {observed:.1%}.",
            f"- Cohen's kappa: {kappa:.3f} (chance agreement {expected:.3f}).",
            f"- Disagreements requiring adjudication: {len(disagreements)}.",
            f"- Total adjudication queue after adding ambiguous and missing-note cases: {len(adjudication)}.",
            "",
            markdown_table(confusion),
            "## Split-by-decision counts",
            "",
            "| Split | " + " | ".join(DECISIONS) + " |",
            "| --- | " + " | ".join("---:" for _ in DECISIONS) + " |",
        ]
    )
    for split in ("train", "valid", "test"):
        lines.append(f"| {split} | " + " | ".join(str(split_counts[split, decision]) for decision in DECISIONS) + " |")
    lines.extend(
        [
            "",
            "## Interpretation and remaining risk",
            "",
            "Agreement is high, but it does not replace adjudication. The canonical labels must not be generated until every row in `ontology_adjudication_queue.csv` has a final decision and rationale.",
            "",
            "The absence of any `smoke` decision in the historical class-1 queue is an empirical finding about this queue, not evidence that the full dataset has no smoke annotations; smoke may already exist under another class.",
            "",
            "Review timestamps indicate rapid keyboard-based review. This is not automatically invalid, but the adjudicator should inspect every disagreement/ambiguous case at full image resolution and verify a small random sample of agreements before freezing labels.",
        ]
    )
    (args.output_dir / "ONTOLOGY_REVIEW_QC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
