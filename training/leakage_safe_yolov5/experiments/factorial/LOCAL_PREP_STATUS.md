# Local factorial preparation status

Status date: 2026-08-15

## Completed

- The frozen ontology audit contains 498 resolved class-1 records.
- The audited three-class dataset retains all 1,887 images and applies 95 relabel and 32 drop operations.
- Detection-only original and audited variants convert historical polygon rows to tight axis-aligned boxes so YOLOv5 and modern Ultralytics loaders consume the same samples.
- Cell A (clean/original ontology) and Cell B (clean/audited ontology) each contain 1,511 identical training images.
- A and B share the same audited validation set (188 images) and audited internal test set (188 images).
- Pinned YOLOv5 loads one batch from every A/B split at 640 px with zero corrupt samples.
- Ultralytics 8.4.120 loads YOLO26n and all 1,511 A/B training samples with zero corrupt samples.
- Pinned YOLOv5 and Ultralytics 8.4.120 also load all 1,511 C/D seed-0 training samples with zero corrupt samples.
- YOLOv5s and YOLO26n run commands pass dry-run validation.

## Controlled leakage datasets

The private companion repository was restored and verified on 2026-08-15. Its immutable raw export at `handoff/yolo 5/` contains 5,068 images plus 5,068 paired labels. All 10,137 entries in `manifests/raw-yolo5.sha256` passed verification. A new machine can reproduce this state with `restore_factorial_data.sh`; no workstation-specific source path is required.

Do not create C/D by copying the fixed test images directly. The preregistered design requires unused variants from eligible test components plus size-matched clean training replacements drawn using the restored raw export.

The raw-data blocker is resolved. Seed-specific C/D datasets for seeds 0, 1, and 2 have been generated. Each seed injects one unused offline variant from each of 41 strictly eligible test components and removes 41 clean training images from paired original/audited class-signature strata. This is 100% of the strictly eligible components, or 41/188 = 21.8% of the complete internal test set. C and D contain identical 1,511-image pools within each seed, and no fixed test representative is copied byte-identically into training.
