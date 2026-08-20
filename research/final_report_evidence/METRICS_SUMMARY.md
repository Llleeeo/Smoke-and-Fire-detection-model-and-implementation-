# Internal metrics summary

All statistics are descriptive. Sample SD uses n=3 training seeds.

## Validation final epoch, mean +/- sample SD

| Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.8042 +/- 0.0543 | 0.6833 +/- 0.0238 | 0.7237 +/- 0.0252 | 0.4208 +/- 0.0094 |
| B | 0.8190 +/- 0.0270 | 0.7156 +/- 0.0151 | 0.7517 +/- 0.0145 | 0.4446 +/- 0.0108 |
| C | 0.8044 +/- 0.0261 | 0.7071 +/- 0.0251 | 0.7389 +/- 0.0104 | 0.4310 +/- 0.0129 |
| D | 0.8248 +/- 0.0301 | 0.7092 +/- 0.0217 | 0.7587 +/- 0.0193 | 0.4511 +/- 0.0107 |

## Internal test, mean +/- sample SD

| Condition | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.7970 +/- 0.0576 | 0.7280 +/- 0.0219 | 0.7700 +/- 0.0223 | 0.4630 +/- 0.0193 |
| B | 0.8487 +/- 0.0245 | 0.7420 +/- 0.0187 | 0.8070 +/- 0.0295 | 0.4767 +/- 0.0200 |
| C | 0.8130 +/- 0.0308 | 0.6957 +/- 0.0049 | 0.7647 +/- 0.0212 | 0.4587 +/- 0.0110 |
| D | 0.7953 +/- 0.0560 | 0.7597 +/- 0.0185 | 0.8013 +/- 0.0274 | 0.4753 +/- 0.0203 |

## Factorial contrasts on mAP@0.5:0.95

| Metric | B-A ontology | C-A contamination/original | D-B contamination/audited | D-C ontology under contamination | Interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation final | +0.0238 | +0.0102 | +0.0065 | +0.0201 | -0.0037 |
| Validation best observed | +0.0268 | +0.0160 | +0.0050 | +0.0158 | -0.0110 |
| Internal test | +0.0137 | -0.0043 | -0.0014 | +0.0166 | +0.0029 |

## Sample size and missingness

- Formal training: 12/12 runs complete, 45 epochs each; validation used 188 images per run.
- Internal test: 12/12 evaluations complete, using 188 images and 167 instances per run.
- Run-level internal numeric stdout was not retained, so the 12 per-run metric rows are `NA`.
- Verified condition mean +/- sample SD is retained and used for reporting.
- Missing values are written as `NA`, never as zero.
