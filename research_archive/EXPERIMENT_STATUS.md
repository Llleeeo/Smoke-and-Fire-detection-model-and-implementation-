# Experiment Status

## Current state

- **Phase 1 smoke test: complete.** A/B/C/D each passed the five-epoch smoke gate and produced required artifacts.
- **Formal factorial training: complete.** Four conditions × three seeds × 45 epochs; 12/12 final runs completed.
- **Internal test: complete.** All 12 `best.pt` checkpoints were evaluated on the fixed 188-image, 167-instance internal set.
- **External generalization: weak.** D-Fire proxy, TACO cigarette-only, and the preliminary manual cigarette pilot all performed far below the internal results.
- **External work: stopped.** No further TACO, D-Fire, cigarette/external evaluation, training, or downloading is authorized.
- **Performance-improvement experiments: not started.**

## Completed stages

| Stage | Scope | Status | Main output |
|---|---|---|---|
| Phase 1 | A/B/C/D, seed 0, 5 epochs | Complete / smoke gate passed | Pipeline diagnostics only |
| Screening | A/B/C/D, seed 0, 25 epochs | Complete | Non-confirmatory screening |
| Formal matrix | A/B/C/D, seeds 0–2, 45 epochs | Complete | 12 checkpoints |
| Internal test | 12 best checkpoints | Complete | Condition mean ± sample SD |
| D-Fire preparation | Kaggle mirror, 470-image pilot | Complete with rights/pHash caveats | Audit and manifests |
| D-Fire unchanged evaluation | Two-class YAML vs three-class checkpoint | Blocked before inference | No metrics |
| D-Fire proxy evaluation | fire→flame proxy and smoke, A seed0 | Complete, exploratory | Weak external AP |
| TACO audit/evaluation | cigarette-only, A seed0 | Complete, supplementary | Weak cigarette AP |
| Manual cigarette pilot | preliminary manual boxes, A seed0 | Complete, preliminary | Weak cigarette AP |

## Phase 1 smoke-test metrics

All four cells passed the smoke gate with seed 0, five epochs, image size 640 and batch 8. These values are execution diagnostics only.

| Cell | Precision | Recall | mAP50 | mAP50:95 | Approx. wall time |
|---|---:|---:|---:|---:|---:|
| A | 0.5355 | 0.5538 | 0.5247 | 0.2288 | 2m43s |
| B | 0.6457 | 0.5397 | 0.5263 | 0.2229 | 2m24s |
| C | 0.4695 | 0.3983 | 0.4192 | 0.1685 | 2m23s |
| D | 0.4766 | 0.4119 | 0.3909 | 0.1911 | 2m30s |

The later seed-0 25-epoch screening also completed for A/B/C/D, but remains non-confirmatory and is not mixed with the formal 45-epoch matrix.

## Conclusions that may be retained

B and D produced the strongest mean validation results. Audited-label conditions have a positive descriptive association with validation mAP50:95. Internal-test mAP50:95 was approximately 0.459–0.477 across conditions with moderate seed variation. The Windows GPU workflow is reproducible enough to rerun from the recorded code/data commits and manifests.

## Claims that cannot be made

The factorial contrasts are not significance tests or causal estimates. Controlled leakage did not produce a stable benefit across validation and internal test. Internal performance does not establish external generalization. D-Fire is not a strict flame benchmark, TACO is not a three-class benchmark, and the manual cigarette pilot is not publication-grade external evidence.

## Failed, blocked, and incomplete work

- One A/seed2 formal attempt failed with a PyTorch DataLoader worker exit. The failed directory was preserved; the condition was rerun successfully with workers=0.
- A harness `git fetch origin` timeout, a one-time PowerShell warning, and recurring AMP deprecation warnings were non-fatal.
- The first unchanged D-Fire evaluation was blocked by a 2-class/3-class mismatch before image inference.
- The first TACO scan found derived labels named `*.jpg.txt`; that invalid attempt was stopped and corrected before the valid supplementary evaluation.
- D-Fire official-source intake was previously blocked by licensing/independence uncertainty; the later Kaggle mirror remains a provisional source with third-party rights risk.
- Perceptual-hash overlap was not completed for external pilots.
- Per-run internal-test numeric stdout was not persisted in machine-readable form. Verified condition mean ± SD is retained; run-level internal cells are `NA`.
- External licensing and third-party image rights remain unresolved enough to prevent a strict public benchmark claim.
