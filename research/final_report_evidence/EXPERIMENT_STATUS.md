# Experiment status for the URIS Completion Report

## Completed work

| Stage | Scope | Status | Report use |
| --- | --- | --- | --- |
| Data-integrity audit | source-name, SHA-256, and perceptual near-duplicate checks | Complete | Dataset methodology |
| Leakage-safe split | 1,511 train / 188 validation / 188 internal test | Complete | Dataset methodology |
| Ontology audit | 498 historical class-1 annotations | Complete and frozen | Annotation methodology |
| Phase 1 | A/B/C/D, seed 0, 5 epochs | Complete | Pipeline validation only |
| Screening | A/B/C/D, seed 0, 25 epochs | Complete | Non-confirmatory development only |
| Formal matrix | A/B/C/D, seeds 0-2, 45 epochs | 12/12 complete | Primary experiment |
| Internal test | all 12 best checkpoints on 188 images / 167 instances | 12/12 complete | Primary results |

## Primary conclusions

- Audited-label conditions B and D produced the strongest mean validation mAP@0.5:0.95.
- Internal-test mAP@0.5:0.95 ranged from 0.4587 to 0.4767 across conditions.
- The ontology contrasts were positive in both split conditions.
- Controlled contamination did not produce a stable performance benefit across validation and internal testing.
- Results are descriptive across three seeds and are not significance tests or causal estimates.

## Reporting restrictions

- Five-epoch smoke tests and 25-epoch screening runs are execution/development evidence, not final performance evidence.
- The internal test belongs to the project's collection context and does not establish performance in substantially different environments.
- One A/seed2 attempt failed because of a DataLoader worker exit; the preserved protocol reran it successfully with workers=0.
- Per-run internal-test numeric stdout was not persisted. Verified condition mean +/- sample SD is retained, while unavailable per-run cells remain `NA`.
