# Paper outline

## Working title

Disentangling Data Leakage and Label-Semantic Bias in Fire-Safety Object Detection: A Controlled Factorial Study

## One-sentence contribution

We estimate how near-duplicate leakage and semantic-ontology inconsistency individually and jointly distort internal fire-safety detector evaluation, then test whether those conclusions survive a locked source-disjoint external evaluation.

## Abstract skeleton

1. **Problem:** Fire-prone object detectors are commonly evaluated on web-sourced, augmented datasets where related images and inconsistent class meanings can cross dataset boundaries.
2. **Method:** We introduce a controlled 2 x 2 factorial evaluation combining source grouping, exact/perceptual duplicate analysis, annotation-ontology adjudication, controlled leakage injection, and a common canonical test ontology.
3. **Evidence:** `[N groups / N images / N video hours]`, two detector families, three training seeds, frozen internal and external tests.
4. **Results:** Leakage changes `[metric]` by `[effect and 95% CI]`; ontology correction changes `[class finding]`; their interaction is `[effect]`; internal and external effects `[agree/disagree]`.
5. **Conclusion:** Standard benchmark AP `[overstates/does not predict]` generalisation when duplicate structure and class semantics are not audited.

## Claimed contributions

1. A controlled factorial study separating the individual and interaction effects of near-duplicate leakage and semantic ontology inconsistency in fire-safety object detection.
2. A matched internal-versus-external evaluation testing whether those effects survive on locked unseen buildings/cameras, with source-cluster confidence intervals.
3. Reproducible source groups, duplicate manifests, ontology decisions, and experiment recipes for auditing the result.

## Planned figures

1. Primary 2 x 2 interaction plot for internal and external evaluation, with 95% confidence intervals.
2. Dataset lineage from raw exports to source/pHash components, ontology adjudication, and external holdout.
3. Optional leakage dose-response plot under the audited ontology.

## Planned tables

1. Dataset composition, source groups, annotations, negative duration, provenance, and overlap checks.
2. Four-cell factorial results for the primary detector, including interaction estimates.
3. Replication-architecture results and internal-versus-external effect estimates.

## Required limitations

- The web-sourced historical dataset may not be redistributable in full.
- The current internal test was used during development and is not a never-seen final holdout.
- Fire occurrence is rare; the negative-video estimate may still have wide confidence intervals.
- The work evaluates warning support and must not be presented as a replacement for certified fire-safety systems.
- The escape-route prototype is outside the detector evaluation unless a real, tested detection-to-routing interface is implemented.

## Claims that are currently prohibited

- "Class 1 was corrected to flame" before annotation-level adjudication is complete.
- "Leakage repair caused a +0.671 AP improvement"; the historical comparison also changes training and ontology.
- "Deployment ready" based on 188 still images or one inference example.
- "Real-time escape routing" while routing remains a disconnected prototype.
- "State of the art" without comparable public datasets, baselines, and protocols.

## Venue decision gate

- **ICASSP 2027:** use only if the 2 x 2 interaction, external holdout, and concise four-page story are complete by early September 2026.
- **IEEE ICCE 2027:** pragmatic applied-systems option if the strongest result is the trustworthy monitoring/false-alarm evaluation and edge-device evidence.
- **Fire Safety Journal:** no fixed conference deadline; preferred longer-form route if external video, operational metrics, and fire-safety implications require a full article.

Do not submit the same manuscript to more than one venue at the same time.
