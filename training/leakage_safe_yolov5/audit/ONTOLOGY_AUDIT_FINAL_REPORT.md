# Final ontology audit report

Status: frozen; all required adjudications are complete.

## Integrity gates

- Final audit: 498/498 unique records with no blank or ambiguous final decisions.
- Adjudication: 21/21 frozen queue records complete.
- Adjudication CSV and JSON backup match exactly.
- Reviewer 1: yhm; Reviewer 2: hzp; adjudicator recorded as `两人一起`.
- Prespecified agreement sample: 96/100 raw agreement; Cohen's kappa = 0.903.

## Final decision distribution

| Decision | Count | Share |
| --- | ---: | ---: |
| flame | 371 | 74.5% |
| smoke | 0 | 0.0% |
| cigarette | 95 | 19.1% |
| smoking_action | 16 | 3.2% |
| invalid | 16 | 3.2% |

## Detector actions for the audited historical class-1 boxes

| Action | Count | Meaning |
| --- | ---: | --- |
| keep | 371 | Retain the original class ID. |
| relabel | 95 | Change to the canonical class ID. |
| drop | 32 | Exclude `smoking_action` and `invalid` boxes. |

Canonical mapping: `cigarette` = 0, `flame` = 1, `smoke` = 2. The image pool must remain unchanged when dropped boxes make an image empty.

## Split-by-decision counts

| Split | flame | smoke | cigarette | smoking_action | invalid | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 293 | 0 | 80 | 14 | 12 | 399 |
| valid | 39 | 0 | 8 | 0 | 2 | 49 |
| test | 39 | 0 | 7 | 2 | 2 | 50 |

## Scope note

These 498 records are the audited historical class-1 annotations. Existing class-0 and class-2 annotations are outside this queue and remain unchanged unless separately audited.
