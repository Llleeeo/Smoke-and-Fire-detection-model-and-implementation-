# Metrics Summary

All statistics are descriptive, non-confirmatory diagnostics. Sample SD uses n=3 seeds.

## Validation final epoch (mean ± sample SD)

| Condition | Precision | Recall | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|
| A | 0.8042 ± 0.0543 | 0.6833 ± 0.0238 | 0.7237 ± 0.0252 | 0.4208 ± 0.0094 |
| B | 0.8190 ± 0.0270 | 0.7156 ± 0.0151 | 0.7517 ± 0.0145 | 0.4446 ± 0.0108 |
| C | 0.8044 ± 0.0261 | 0.7071 ± 0.0251 | 0.7389 ± 0.0104 | 0.4310 ± 0.0129 |
| D | 0.8248 ± 0.0301 | 0.7092 ± 0.0217 | 0.7587 ± 0.0193 | 0.4511 ± 0.0107 |

## Internal test (mean ± sample SD)

| Condition | Precision | Recall | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|
| A | 0.7970 ± 0.0576 | 0.7280 ± 0.0219 | 0.7700 ± 0.0223 | 0.4630 ± 0.0193 |
| B | 0.8487 ± 0.0245 | 0.7420 ± 0.0187 | 0.8070 ± 0.0295 | 0.4767 ± 0.0200 |
| C | 0.8130 ± 0.0308 | 0.6957 ± 0.0049 | 0.7647 ± 0.0212 | 0.4587 ± 0.0110 |
| D | 0.7953 ± 0.0560 | 0.7597 ± 0.0185 | 0.8013 ± 0.0274 | 0.4753 ± 0.0203 |

## Factorial contrasts on mAP50:95

| Metric | B-A ontology | C-A leakage/original | D-B leakage/audited | D-C ontology under leakage | Interaction |
|---|---:|---:|---:|---:|---:|
| Validation final | +0.0238 | +0.0102 | +0.0065 | +0.0201 | -0.0037 |
| Validation best observed | +0.0268 | +0.0160 | +0.0050 | +0.0158 | -0.0110 |
| Internal test | +0.0137 | -0.0043 | -0.0014 | +0.0166 | +0.0029 |

## External AP and internal-external gap

Gap uses verified A-condition internal mean 0.4630 because exact A-seed0 internal stdout was not persisted. Semantic and cohort differences make these gaps descriptive only.

| External dataset | External AP50:95 | A internal mean | Gap |
|---|---:|---:|---:|
| D-Fire proxy overall | 0.010300 | 0.4630 | 0.452700 |
| TACO cigarette-only | 0.001440 | 0.4630 | 0.461560 |
| Manual cigarette pilot | 0.000166 | 0.4630 | 0.462834 |

## Sample size and missingness

- Formal training: 12/12 runs complete, 45 epochs each; validation 188 images per run.
- Internal test: 12/12 complete, 188 images/167 instances per run; 12 run-level metric rows are NA because numeric stdout was not persisted.
- D-Fire proxy: 470 images/678 instances.
- TACO: 227 images/667 instances.
- Manual cigarette pilot: 111 usable images/125 instances.
- Missing values are written as NA, never 0.
