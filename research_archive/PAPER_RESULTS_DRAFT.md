# Paper Results Draft

## Technical summary

The 3-seed × 4-condition factorial matrix completed for all 12 planned YOLOv5 runs. Audited-label conditions B and D achieved the strongest validation mAP50:95 means (0.4446 and 0.4511), while internal-test means were tightly grouped from 0.4587 to 0.4767. External pilot performance was substantially lower, indicating that strong internal performance did not transfer to the tested external sources.

## Factorial findings

At the final validation epoch, the ontology contrast B−A was +0.0238 mAP50:95, leakage under original labels C−A was +0.0102, and leakage under audited labels D−B was +0.0065. The interaction `(D−C)−(B−A)` was −0.0037. On the internal test, the corresponding contrasts were +0.0137, −0.0043, −0.0013, and +0.0030. These are descriptive diagnostics and do not establish causal effects.

## Internal performance

Across conditions, internal mAP50 ranged from 0.7647 to 0.8070 and mAP50:95 from 0.4587 to 0.4767. Seed-level SD for internal mAP50:95 ranged from 0.0110 to 0.0203. The fixed internal test contained 188 images and 167 instances.

## External pilots

The A-seed0 checkpoint obtained mAP50:95 0.0103 on the D-Fire fire→flame/smoke proxy pilot, 0.00144 on TACO cigarette-only, and 0.000166 on the preliminary manually labelled cigarette pilot. **D-Fire fire is used as a proxy label for flame; this is exploratory, not a strict flame benchmark.** TACO and the manual pilot do not evaluate flame or smoke. These values cannot be pooled or presented as strict three-class external AP.

## Interpretation

The evidence supports completion and internal reproducibility of the factorial workflow, plus a directional association between ontology auditing and validation performance. It does not support strong external generalization, a causal leakage claim, or a publication-grade external benchmark claim. Performance-improvement experiments have not begun.

## Recommended next step

Use this archive to write the methods and results sections without initiating new experiments. Any future performance-improvement study should be pre-specified separately, use a legally cleared external benchmark, and avoid tuning on the current external pilots.

## Further questions

- Can the original per-run internal stdout be recovered from terminal history or archived logs?
- Can external image rights and perceptual overlap be resolved sufficiently for a strict benchmark?
- Would class-specific error analysis explain the internal–external gap without additional training?

