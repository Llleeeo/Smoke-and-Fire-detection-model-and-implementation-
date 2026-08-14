# Ontology review quality report

Status: preliminary; adjudication is still required.

## Completeness and integrity

- Master queue: 498 unique annotation records.
- yhm: 498/498 complete; all decisions use the allowed vocabulary.
- hzp: 100/100 complete on the prespecified blind sample; all record IDs match the master queue.
- Both JSON backups exactly match their exported CSV files.
- Missing mandatory notes: Reviewer 1 = 5; Reviewer 2 = 0.

## Reviewer 1 outcome distribution

| Decision | Count | Share |
| --- | ---: | ---: |
| flame | 368 | 73.9% |
| smoke | 0 | 0.0% |
| cigarette | 97 | 19.5% |
| smoking_action | 15 | 3.0% |
| invalid | 5 | 1.0% |
| ambiguous | 13 | 2.6% |

## Agreement

- Raw agreement: 96/100 = 96.0%.
- Cohen's kappa: 0.903 (chance agreement 0.587).
- Disagreements requiring adjudication: 4.
- Total adjudication queue after adding ambiguous and missing-note cases: 21.

| Reviewer 1 \ Reviewer 2 | flame | smoke | cigarette | smoking_action | invalid | ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| flame | 73 | 0 | 0 | 0 | 0 | 0 |
| smoke | 0 | 0 | 0 | 0 | 0 | 0 |
| cigarette | 2 | 0 | 18 | 1 | 0 | 0 |
| smoking_action | 0 | 0 | 0 | 2 | 0 | 0 |
| invalid | 0 | 0 | 0 | 0 | 0 | 1 |
| ambiguous | 0 | 0 | 0 | 0 | 0 | 3 |

## Split-by-decision counts

| Split | flame | smoke | cigarette | smoking_action | invalid | ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 293 | 0 | 80 | 13 | 4 | 9 |
| valid | 36 | 0 | 10 | 0 | 1 | 2 |
| test | 39 | 0 | 7 | 2 | 0 | 2 |

## Interpretation and remaining risk

Agreement is high, but it does not replace adjudication. The canonical labels must not be generated until every row in `ontology_adjudication_queue.csv` has a final decision and rationale.

The absence of any `smoke` decision in the historical class-1 queue is an empirical finding about this queue, not evidence that the full dataset has no smoke annotations; smoke may already exist under another class.

Review timestamps indicate rapid keyboard-based review. This is not automatically invalid, but the adjudicator should inspect every disagreement/ambiguous case at full image resolution and verify a small random sample of agreements before freezing labels.
