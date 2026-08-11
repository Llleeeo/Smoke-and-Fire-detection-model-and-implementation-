#!/usr/bin/env python3
"""Build the canonical Data Analytics report artifact from saved audit evidence."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    audit_root = package_root / "audit"
    summary = json.loads((audit_root / "summary.json").read_text(encoding="utf-8"))
    verification = json.loads((audit_root / "verification.json").read_text(encoding="utf-8"))
    generated_at = summary["generated_at"]

    split_rows = []
    for split in ("train", "valid", "test"):
        before = summary["input"]["split_counts"][split]
        after = summary["output"][split]
        split_rows.append(
            {
                "split": split,
                "before_images": before["images"],
                "after_images": after["images"],
                "after_sources": after["sources"],
                "after_annotations": after["annotations"],
                "empty_negative_images": after["empty_labels"],
                "cigarette_annotations": after.get("class_0", 0),
                "flame_annotations": after.get("class_1", 0),
                "smoke_annotations": after.get("class_2", 0),
            }
        )
    removed_rows = []
    reason_labels = {
        "exact_duplicate_same_label": "Exact duplicate",
        "exact_duplicate_label_conflict": "Exact duplicate with label conflict",
        "extra_variant_or_near_duplicate_for_train_component": "Train near-duplicate / extra variant",
        "extra_variant_or_near_duplicate_for_valid_component": "Validation near-duplicate / extra variant",
        "extra_variant_or_near_duplicate_for_test_component": "Test near-duplicate / extra variant",
    }
    risk_labels = {
        "exact_duplicate_same_label": "Overweighting and leakage risk",
        "exact_duplicate_label_conflict": "Inconsistent supervision",
        "extra_variant_or_near_duplicate_for_train_component": "Overweighting one visual source",
        "extra_variant_or_near_duplicate_for_valid_component": "Correlated validation evidence",
        "extra_variant_or_near_duplicate_for_test_component": "Correlated test evidence",
    }
    removed_total = summary["checks"]["removed_records"]
    for reason, count in sorted(summary["removed"].items(), key=lambda item: -item[1]):
        removed_rows.append(
            {
                "reason": reason_labels.get(reason, reason),
                "count": count,
                "share": count / removed_total,
                "risk": risk_labels.get(reason, "Data quality risk"),
                "action": "Removed; original export retained",
            }
        )
    overlap_rows = []
    for pair, values in verification["overlaps"].items():
        overlap_rows.append(
            {
                "split_pair": pair.replace("_", " / "),
                "source_name_overlap": values["source_names"],
                "exact_hash_overlap": values["exact_image_hashes"],
                "perceptual_near_duplicate_overlap": values["perceptual_hash_distance_le_4"],
            }
        )

    audit_source = {
        "id": "audit",
        "label": "Generated YOLOv5 cleaning audit",
        "path": "summary.json",
        "query": {
            "language": "python",
            "query": "python scripts/verify_dataset.py dataset",
            "description": "Deterministic dataset inventory, source grouping, exact and perceptual duplicate checks, label validation, and image decode verification.",
            "filters": [
                "Classes limited to cigarette, flame, and smoke",
                "Perceptual duplicate threshold: 64-bit pHash Hamming distance <= 4",
                "Split ratios: 80% train, 10% validation, 10% test by perceptual component",
            ],
            "metric_definitions": [
                "Source overlap counts shared pre-Roboflow source names across split pairs.",
                "Perceptual overlap counts image pairs across splits with pHash Hamming distance <= 4.",
                "Removed records equal original image records minus one selected representative per perceptual component.",
            ],
        },
    }

    def sql_string(value: object) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    removal_values = ",\n  ".join(
        "(" + ", ".join(
            [
                sql_string(row["reason"]),
                str(row["count"]),
                f"{row['share']:.12f}",
                sql_string(row["risk"]),
                sql_string(row["action"]),
            ]
        ) + ")"
        for row in removed_rows
    )
    removal_source = {
        "id": "removed_source",
        "label": "Reviewed removal-reason rows",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": f"SELECT * FROM (VALUES\n  {removal_values}\n) AS t(reason, count, share, risk, action)",
            "description": "Reviewed counts of records excluded by the deterministic cleaning script.",
            "metric_definitions": ["Count is the number of original image records excluded for each mutually exclusive reason."],
        },
    }
    split_values = ",\n  ".join(
        "(" + ", ".join(
            [
                sql_string(row["split"]),
                str(row["before_images"]),
                str(row["after_images"]),
                str(row["after_sources"]),
                str(row["after_annotations"]),
                str(row["empty_negative_images"]),
                str(row["cigarette_annotations"]),
                str(row["flame_annotations"]),
                str(row["smoke_annotations"]),
            ]
        ) + ")"
        for row in split_rows
    )
    split_source = {
        "id": "split_source",
        "label": "Reviewed split-summary rows",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": f"SELECT * FROM (VALUES\n  {split_values}\n) AS t(split, before_images, after_images, after_sources, after_annotations, empty_negative_images, cigarette_annotations, flame_annotations, smoke_annotations)",
            "description": "Before/after image counts and final annotation counts by split.",
            "metric_definitions": ["After images count one retained representative per perceptual component."],
        },
    }
    overlap_values = ",\n  ".join(
        "(" + ", ".join(
            [
                sql_string(row["split_pair"]),
                str(row["source_name_overlap"]),
                str(row["exact_hash_overlap"]),
                str(row["perceptual_near_duplicate_overlap"]),
            ]
        ) + ")"
        for row in overlap_rows
    )
    overlap_source = {
        "id": "overlap_source",
        "label": "Reviewed cross-split leakage checks",
        "query": {
            "engine": "DuckDB",
            "language": "sql",
            "sql": f"SELECT * FROM (VALUES\n  {overlap_values}\n) AS t(split_pair, source_name_overlap, exact_hash_overlap, perceptual_near_duplicate_overlap)",
            "description": "Cross-split overlap counts from the independent verifier.",
            "metric_definitions": ["Perceptual overlap uses a 64-bit pHash Hamming-distance threshold of 4."],
        },
    }
    report_sources = [audit_source, removal_source, split_source, overlap_source]

    title = "YOLOv5 数据修复报告"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "吸烟、火焰与烟雾检测数据的去重、重分组和可复现性修复结果。",
            "generatedAt": generated_at,
            "sources": report_sources,
            "charts": [
                {
                    "id": "removed_reasons_chart",
                    "title": "按原因移除的图片记录",
                    "subtitle": f"共移除 {removed_total:,} 条；原始导出仍完整保留",
                    "showDescription": True,
                    "intent": "comparison",
                    "question": "清洗主要移除了哪些类型的重复或相关记录？",
                    "rationale": "四类移除原因适合用排序水平条形图比较绝对数量。",
                    "type": "horizontalBar",
                    "dataset": "removed_reasons",
                    "sourceId": "removed_source",
                    "encodings": {
                        "x": {"field": "reason", "type": "nominal", "label": "移除原因"},
                        "y": {"field": "count", "type": "quantitative", "aggregate": "none", "label": "图片记录", "format": "number"},
                        "tooltip": [
                            {"field": "count", "type": "quantitative", "label": "数量", "format": "number"},
                            {"field": "share", "type": "quantitative", "label": "占全部移除记录", "format": "percent"},
                            {"field": "risk", "type": "text", "label": "风险"},
                        ],
                    },
                    "xAxisTitle": "原因",
                    "yAxisTitle": "移除的图片记录数",
                    "valueFormat": "number",
                    "layout": "full",
                    "labels": {"values": "all"},
                    "palette": {"kind": "sequential", "name": "blue"},
                    "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True, "categoryLabelPolicy": "wrap"},
                    "surface": {"surface": "card", "viewMode": "both", "showControls": False},
                    "maxRows": 10,
                }
            ],
            "tables": [
                {
                    "id": "split_table",
                    "title": "清洗前后分组规模",
                    "subtitle": "清洗后按感知相似组计数；训练集、验证集、测试集分别约占 80%/10%/10%",
                    "showDescription": True,
                    "dataset": "split_counts",
                    "defaultSort": {"field": "after_images", "direction": "desc"},
                    "density": "spacious",
                    "sourceId": "split_source",
                    "layout": "full",
                    "columns": [
                        {"field": "split", "label": "分组", "type": "text"},
                        {"field": "before_images", "label": "清洗前图片", "type": "number", "format": "number"},
                        {"field": "after_images", "label": "清洗后图片", "type": "number", "format": "number"},
                        {"field": "after_sources", "label": "独立感知组", "type": "number", "format": "number"},
                        {"field": "after_annotations", "label": "标注实例", "type": "number", "format": "number"},
                        {"field": "empty_negative_images", "label": "负样本", "type": "number", "format": "number"},
                    ],
                },
                {
                    "id": "overlap_table",
                    "title": "跨分组泄漏检查",
                    "subtitle": "逐对比较 train、valid 与 test；三种重叠口径均应为 0",
                    "showDescription": True,
                    "dataset": "overlap_checks",
                    "defaultSort": {"field": "split_pair", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": "overlap_source",
                    "layout": "full",
                    "columns": [
                        {"field": "split_pair", "label": "分组对", "type": "text"},
                        {"field": "source_name_overlap", "label": "同源文件名重叠", "type": "number", "format": "number"},
                        {"field": "exact_hash_overlap", "label": "完全相同图片", "type": "number", "format": "number"},
                        {"field": "perceptual_near_duplicate_overlap", "label": "感知近重复", "type": "number", "format": "number"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}", "layout": "full"},
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "sourceId": "audit",
                    "layout": "full",
                    "body": (
                        "## Executive Summary\n\n"
                        "- **数据泄漏已经修复。** 清洗后的 train、valid、test 之间，同源文件名、逐字节重复和感知近重复的交叉数量均为 0。\n"
                        f"- **5,068 张导出图片被归并为 1,887 个独立感知组。** 最终保留 1,511 张训练图、188 张验证图和 188 张测试图；其余 {removed_total:,} 条重复、增强版本或连拍记录已从清洗副本移除。\n"
                        "- **数据包现在可直接交给 YOLOv5。** 路径配置已修正，标签与图像一一配对，所有输出图片均可解码，伪装成 JPEG 的 HEIC 文件已转换。\n"
                        "- **旧模型不能作为清洗后的最终结论。** 原 `best.pt` 和旧 mAP 仅保留作参考；需要重新训练并在新测试集上评估。"
                    ),
                },
                {
                    "id": "removal_finding",
                    "type": "markdown",
                    "sourceId": "audit",
                    "layout": "full",
                    "body": (
                        "## 大部分冗余来自重复增强和近似画面\n\n"
                        "**严格清洗不是简单删除那 114 张跨组完全重复图片。** 它还把相同原图的 Roboflow 增强版本、不同文件名的近重复图片和连续拍摄画面归入同一个感知组。每组只保留标注最完整的一张，避免同一场景反复影响训练或评估。"
                    ),
                },
                {"id": "removed_chart_block", "type": "chart", "chartId": "removed_reasons_chart", "layout": "full"},
                {
                    "id": "split_finding",
                    "type": "markdown",
                    "sourceId": "audit",
                    "layout": "full",
                    "body": (
                        "## 评估集变小，但每张图提供更独立的证据\n\n"
                        "验证集和测试集各保留 188 个独立感知组。与旧划分相比，数量下降是预期结果：旧导出把增强版本和连拍当作独立样本，而新划分按真实来源计数。数据量看起来更小，但测试分数更接近模型对新场景的泛化表现。"
                    ),
                },
                {"id": "split_table_block", "type": "table", "tableId": "split_table", "layout": "full"},
                {
                    "id": "verification_finding",
                    "type": "markdown",
                    "sourceId": "audit",
                    "layout": "full",
                    "body": (
                        "## 三种泄漏检查全部通过\n\n"
                        "检查同时覆盖文件名来源、SHA-256 完全重复和 64 位感知哈希近重复。三个分组对在所有口径上都为 0；图像解码、标签字段、类别范围和归一化坐标检查也没有失败项。"
                    ),
                },
                {"id": "overlap_table_block", "type": "table", "tableId": "overlap_table", "layout": "full"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 下一步：重新训练后再报告指标\n\n"
                        "1. 按修复包内的 YOLOv5 依赖建立训练环境。\n"
                        "2. 运行 `train_clean.sh`，可从旧 `best.pt` 继续微调，但训练数据改用新的 `dataset/data.yaml`。\n"
                        "3. 训练完成后，在新 test 分组上单独运行评估，并以新的 per-class AP、mAP@0.5 和 mAP@0.5:0.95 替换旧指标。\n"
                        "4. 再用真实部署视频做一次场景外测试，尤其关注烟雾漏检和背景误报。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 仍需同伴确认的两点\n\n"
                        "- 空标签图片是否明确设计为笔、铅笔等香烟相似物负样本？抽样结果支持这一判断，但最好把它写进标注规范。\n"
                        "- 最终部署场景是固定摄像头、手机照片还是公开视频？这会决定下一轮数据应优先补哪些光照、距离和遮挡条件。"
                    ),
                },
                {
                    "id": "caveats",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Caveats and Assumptions\n\n"
                        "- 感知哈希阈值采用汉明距离不超过 4，重点捕捉高度相似的连拍和重复画面；它不是完整的视频身份识别。\n"
                        "- 多边形标签保留原样，YOLOv5 检测训练会将其转换为框；换用其他框架时需再次检查兼容性。\n"
                        "- 报告只说明数据包已经通过质量检查，不代表模型已在清洗数据上重新训练。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "split_counts": split_rows,
                "removed_reasons": removed_rows,
                "overlap_checks": overlap_rows,
            },
        },
        "sources": report_sources,
    }
    (audit_root / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = """# Report source notes

- Audience: product stakeholders (executive report specification).
- Delivery mode: portable HTML fallback because the MCP report renderer is not callable in this desktop session.
- Required structure mapping: title, Executive Summary, three evidence findings, recommended next steps, further questions, and caveats are all visible sections.
- Chart map: `removed_reasons_chart` answers why records were removed; horizontal bar; reason/count; single-root blue palette; rendered in `report.html`.
- The split and leakage evidence uses tables because exact lookup is the main job and only three split pairs exist.
- No trend chart is used because the evidence is a one-time before/after audit rather than a temporal series.
- Source evidence: `summary.json`, `verification.json`, manifests, and the executed `data_quality_audit.ipynb`.
- Portable report validation and structural verification passed; browser-level responsive/source-dialog QA did not run because no compatible Chromium executable was available.
"""
    (audit_root / "report_notes.md").write_text(notes, encoding="utf-8")
    print(audit_root / "artifact.json")


if __name__ == "__main__":
    main()
