# Experiment log

## 2026-08-14 - environment and data restoration

- Restored the private raw handoff and clean dataset into Git-ignored local paths.
- Verified the private SHA-256 manifests for the raw and cleaned datasets.
- Rebuilt the pinned YOLOv5 environment at commit `3fb11111c6a8088fbc91430a1f99d207c16f0620`.
- Verified the restored clean split: 1,511 train / 188 validation / 188 test; zero source-name, SHA-256, or pHash-distance-at-most-4 overlap across splits.

## 2026-08-14 - ontology audit queue

- Generated 498 class-1 review rows covering 494 images.
- Split counts: 399 train / 49 validation / 50 test annotations.
- Geometry: 489 boxes / 9 polygons represented by their bounding rectangle in the review HTML.
- Manual inspection confirmed that historical class 1 contains both visible flames and smoking/face-cigarette regions. The existing `flame` rename is therefore not an annotation-level correction.
- Final review artifacts: `training/leakage_safe_yolov5/audit/ontology_review_final.csv` and `ontology_canonical_labels.csv`.

## 2026-08-14 - one-epoch pipeline smoke test

- Purpose: execution validation only; excluded from final performance evidence.
- Initialisation: generic `yolov5s.pt`, not the semantically mixed legacy checkpoint.
- Configuration: 1 epoch, 640 px, batch 8, MPS, workers 2, seed 0.
- Completed successfully in about 0.036 hours of reported epoch time.
- Validation result after one epoch: precision 0.696, recall 0.062, AP@0.5 0.0183, AP@0.5:0.95 0.0041. These values are expected to be poor and are not scientifically interpreted.
- Observed reproducibility risks: repeated Torch AMP deprecation warnings and a transient numerical warning in the historical YOLOv5 augmentation matrix. The run did not abort.
- Output: Git-ignored `training/leakage_safe_yolov5/runs/protocol_smoke_1e/`.

## 2026-08-14 - ablation dataset builder validation

- Extended the dataset builder with `--train-policy one|all` and `--drop-train-negatives` while preserving the historical `one` default.
- Rebuilding with the default reproduced 1,511 / 188 / 188 images and the same selected and removed manifest content; the only manifest difference was LF versus CRLF line endings.
- The `all` training-variant condition produced 3,399 train / 188 validation / 188 test images across the identical 1,511 / 188 / 188 source components.
- The `all` condition passed source-name, exact-hash, and pHash-distance-at-most-4 cross-split checks.
- The `--drop-train-negatives` condition produced 1,235 train images with all 276 empty-label training images excluded while leaving validation/test fixed; verification passed. This condition is a diagnostic only unless total optimizer steps/image exposures are controlled.
- These are dataset-construction checks, not model results.

## 2026-08-15 - ontology audit frozen

- Reviewer 1: `yhm`; reviewer 2: `hzp`.
- Completed 498 ontology decisions covering the historical class-1 annotations.
- The 100-row independent overlap achieved 96% raw agreement and Cohen's kappa 0.903.
- All 21 rows requiring adjudication were resolved.
- Frozen decisions: 371 flame, 95 cigarette, 16 smoking-action exclusions, and 16 invalid exclusions.
- The audited dataset retains all 1,887 images, relabels 95 boxes, drops 32 boxes, and keeps newly empty-label images as negatives.

## 2026-08-15 - primary factorial datasets prepared

- Restored and SHA-256 verified the private 5,068-image raw export.
- Converted historical polygon annotations to tight axis-aligned detection boxes for consistent YOLOv5 and modern Ultralytics loading.
- Prepared clean/original cell A and clean/audited cell B with identical 1,511-image training pools.
- For seeds 0, 1, and 2, prepared controlled-leakage cells C/D by injecting one unused offline variant from each of 41 strictly eligible test components and removing 41 class-signature-matched clean training images.
- The leakage endpoint therefore covers 41/188 (21.8%) of all internal test components and 100% of the strictly eligible components; it must not be described simply as 100% test leakage.
- C/D image identities are identical within each seed, training size remains 1,511, and no exact test-representative hash enters training.
- Pinned YOLOv5 and Ultralytics 8.4.120 loaded the seed-0 C/D datasets with 0 corrupt samples. Runner dry-runs passed for both architectures.
- These are construction and loading checks, not model results.

## 2026-08-18 - formal factorial training and internal evaluation complete

- Completed A/B/C/D for seeds 0, 1, and 2 at 45 epochs: 12/12 formal YOLOv5s runs.
- Evaluated every selected `best.pt` on the common audited internal test: 12/12 evaluations complete.
- Internal test size: 188 images / 167 instances per run.
- Internal mAP@0.5:0.95 means ranged from 0.4587 to 0.4767 across conditions.
- Audited-label conditions B and D produced the strongest internal mAP@0.5:0.95 means.
- Canonical tables and figures are under `research/final_report_evidence/`.
