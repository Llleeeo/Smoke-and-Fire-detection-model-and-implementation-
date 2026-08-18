# Limitations

1. Factorial statistics are descriptive with only three seeds. No confidence intervals, hypothesis tests, multiplicity correction, or causal identification were performed.
2. The internal test belongs to the project’s collection context and is not an independent external benchmark.
3. Per-run internal metrics were not saved as text/CSV; only verified condition mean ± sample SD survives.
4. External datasets have mismatched class coverage and ontology. D-Fire covers fire/smoke; TACO and the manual Kaggle pilot cover cigarette only.
5. **D-Fire fire is used as a proxy label for flame; this is exploratory, not a strict flame benchmark.**
6. D-Fire Kaggle is a mirror with unverified page licensing; TACO images retain source-owner rights; the Kaggle cigarette source has third-party rights risk.
7. Exact SHA-256 overlap was zero for audited pilots, but perceptual-hash overlap was not completed.
8. External pilots are small or provisional and differ in source domain, annotation conventions, target scale, image quality, and class prevalence.
9. The internal–external gap compares the A-condition internal mean with external evaluations from A seed0 because exact A-seed0 internal stdout was not persisted.
10. Checkpoint binaries and private datasets are intentionally excluded from GitHub, so the public archive is a results/provenance package rather than a fully self-contained rerun bundle.

