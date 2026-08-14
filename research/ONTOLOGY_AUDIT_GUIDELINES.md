# Ontology audit guidelines

Status: protocol v1.0, freeze before review

## Unit of review

Each row represents one historical class-1 annotation, not one image. The red box is the annotation being reviewed. Grey boxes are context only. If one image contains two red-box records, decide each annotation separately.

The reviewer decides what the **red box itself is intended to localise**. Do not infer a label from the filename, the historical class, another reviewer's decision, or what would make the dataset more balanced.

## Allowed semantic decisions

| Decision | Use when | Do not use when |
| --- | --- | --- |
| `flame` | Visible combustion flame is present and the box principally localises the flame region. Flame colour may be yellow, orange, red or blue. | The box principally covers a person, face, hand, cigarette, smoke plume, lamp, reflection or sunset. |
| `smoke` | A visible smoke plume is present and the box principally localises smoke rather than flame. | The region is cloud, fog, steam, blur, or smoke is only inferred from context. |
| `cigarette` | A cigarette/cigar or lit tip is visibly identifiable and the box is object-level: it is centred on the object and does not principally describe the person's face/body or the act of smoking. | The cigarette is not visually resolvable, or the box covers a face, hand-to-mouth interaction, or most of a person. |
| `smoking_action` | The annotation describes a person/face/hand-to-mouth interaction or a broad smoking region rather than a localisable cigarette object. | A cigarette is tightly and visibly localised. |
| `invalid` | No canonical target is localised: unrelated background/object, cloud/fog/steam, annotation is grossly misplaced, or the target is absent. | The evidence is merely difficult to see; use `ambiguous` when a reasonable reviewer cannot decide. |
| `ambiguous` | Image evidence is genuinely insufficient, two target concepts are inseparable in the box, or a confident decision would require guessing. | Do not use simply to save time. Add a short note explaining the ambiguity. |

Canonical targets are `cigarette`, `flame`, and `smoke`. `smoking_action` is retained as an audit outcome but excluded from the canonical three-class detector unless the study is explicitly redesigned.

## Boundary rules

1. Judge visible pixels, not presumed intent. A filename containing `smoking` is not evidence by itself.
2. When a box contains both flame and smoke, choose the concept that the box principally localises. Use `ambiguous` if neither dominates.
3. A large face/person box with a tiny cigarette is `smoking_action`, not `cigarette`.
4. A tight box around a visibly identifiable cigarette plus a small amount of finger/lip context may still be `cigarette`.
5. Clouds, fog, steam, bright lamps, reflections and sunsets are `invalid` unless genuine smoke/flame is visibly present inside the box.
6. If the semantic class is clear but the box is moderately loose, keep the semantic decision and write `box_loose` in notes. If the box is unusably displaced and does not localise the target, use `invalid` and write `bad_box`.
7. Offline augmentations or near-duplicates are reviewed independently in Reviewer 1's full queue. Do not copy a prior answer without looking at the displayed pixels.

## Reviewer workflow

### Reviewer 1

- Reviews all 498 annotations.
- Uses only the six allowed values.
- Adds notes for every `ambiguous` or `invalid` decision and for material box-quality problems.
- Exports a CSV backup after every review session.

### Reviewer 2

- Reviews the fixed 100-record sample independently.
- Must not see Reviewer 1's decisions before exporting their own results.
- Uses the same protocol and decision values.

### Adjudication

Adjudication is required for:

- every Reviewer 1 / Reviewer 2 disagreement;
- every `ambiguous` decision from either reviewer;
- every record marked `bad_box` or otherwise unusable.

The adjudicator records the final decision and a short rationale. Do not silently overwrite either original review.

## Quality gates

The audit is complete only when:

- Reviewer 1 has 498/498 valid decisions;
- Reviewer 2 has 100/100 valid blind decisions;
- raw agreement and Cohen's kappa are reported on the prespecified 100-record sample;
- every disagreement/ambiguous case has a final adjudicated decision;
- no final decision is blank;
- record IDs remain unique and the source CSV is preserved unchanged;
- counts are reported by split, final decision, and source group.

Kappa is descriptive evidence, not a pass/fail truth certificate. Report the confusion matrix and disagreement types alongside it.
