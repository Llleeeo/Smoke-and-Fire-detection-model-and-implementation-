# URIS Completion Report source map

This file maps the official Completion Report sections to the evidence retained in `main`.

## Section A - Project and project team

Use the previous report for the project title, ID, work programme, student, and supervisor details. Update the actual completion date, duration, and any approved change-request dates.

Source: `URIS_Progress_Report_2025.pdf`.

## Section B.4 - Summary of objectives achieved

| Proposal objective | Final evidence | Reporting position |
| --- | --- | --- |
| Collect and label images for machine learning | 5,068 raw records audited; 1,887 perceptual components; leakage-safe 1,511/188/188 split; 498 class-1 annotations adjudicated | Completed with substantial quality-control extension |
| Deploy and train YOLOv5 | 12/12 formal YOLOv5s runs and 12/12 internal evaluations completed across four conditions and three seeds | Completed |
| Generate escape routes using building and population variables | Standalone C++ evacuation simulation retained in `Escape_Route_Prototype.pdf` | Prototype completed; tested detector-to-routing integration not completed |

The student and supervisor should confirm the percentages entered in the official form.

## Section B.5 - Research activities

Recommended narrative order:

1. Project motivation and original objectives.
2. Initial object-detection prototype and data collection.
3. Identification of duplicate/leakage and ontology risks.
4. Leakage-safe dataset reconstruction.
5. Two-reviewer annotation-ontology audit and adjudication.
6. Four-condition factorial experiment.
7. Three-seed formal training and fixed internal testing.
8. Internal results and descriptive contrasts.
9. Detection implementation and escape-route prototype.
10. Difficulties, limitations, conclusion, and future work.

### Dataset-integrity evidence

- `training/leakage_safe_yolov5/audit/report.html`
- `training/leakage_safe_yolov5/audit/summary.json`
- `training/leakage_safe_yolov5/audit/verification.json`
- `training/leakage_safe_yolov5/audit/selected_manifest.csv`
- `training/leakage_safe_yolov5/audit/removed_manifest.csv`

### Ontology-audit evidence

- `training/leakage_safe_yolov5/audit/ONTOLOGY_AUDIT_FINAL_REPORT.md`
- `training/leakage_safe_yolov5/audit/ontology_review_final.csv`
- `training/leakage_safe_yolov5/audit/ontology_canonical_labels.csv`
- `training/leakage_safe_yolov5/audit/ontology_review_agreement.json`

### Formal experiment evidence

- `research/final_report_evidence/METRICS_SUMMARY.md`
- `research/final_report_evidence/RUN_REGISTRY.csv`
- `research/final_report_evidence/INTERNAL_RESULTS.csv`
- `research/final_report_evidence/REPRODUCIBILITY.md`
- `research/final_report_evidence/figures/internal_results_by_condition.png`
- `research/final_report_evidence/figures/factorial_effects.png`
- `research/final_report_evidence/figures/seed_variance.png`

### Required interpretation boundaries

- Treat the former one-seed clean-split experiment as development history, not final evidence.
- Use the four-condition, three-seed internal result table as the final quantitative result.
- Describe factorial contrasts as descriptive; do not claim statistical significance or causality.
- Do not claim robust cross-domain generalization.
- Describe the escape-route module as a standalone prototype, not an integrated real-time evacuation system.

## Section B.6 - Project outputs

### Publications and scholarly outputs

No accepted or published paper is evidenced in the repository. Enter `N/A` unless an external publication record is supplied.

### Competitions or awards

No competition or award record is evidenced in the repository. Enter `N/A` unless an external record is supplied.

### Conferences or presentations

The final URIS Showcase ePoster is under `docs/generated-output/eposter/`. Confirm the event date, organiser, presentation title, and official event link before completing the table.

### Innovation and technology development

List:

1. Public research repository.
2. Leakage-safe YOLOv5 workflow.
3. Duplicate and data-integrity audit.
4. Human annotation-ontology review and adjudication workflow.
5. Controlled four-condition factorial experiment.
6. Three-seed internal result and reproducibility archive.
7. Camera/inference implementation described in the previous report and archive branch.
8. Standalone C++ escape-route simulation.
9. URIS Showcase ePoster.

The Completion Report must retain outputs already reported in the Progress Report.

## Section C - Student declaration

Update the name, signature, and completion-report date. Retain the official originality, academic-integrity, non-duplicate-submission, and FYP declarations.

## Sections D-F - Endorsement and approval

Section D is completed by the Chief Supervisor, including rating, comments, project-grant balance, declaration, signature, and date. Section E is for the Departmental/School Research Committee. Section F is for URIC/Graduate School use.

Only a project rated Outstanding is eligible for the Best URIS Research Project Award.
