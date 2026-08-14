# Audit-first fire-prone object detection: experiment protocol

Status: preregistration draft, 2026-08-14

## Intended paper claim

This study asks how near-duplicate leakage and label-ontology mismatch individually and jointly change the apparent performance of fire-safety object detectors, and whether those conclusions survive source-disjoint external evaluation. The paper is an evaluation and data-quality study, not a claim of a new detector architecture.

Provisional title:

> Disentangling Data Leakage and Label-Semantic Bias in Fire-Safety Object Detection: A Controlled Factorial Study

## Research questions

- **RQ1 - joint distortion:** To what extent do near-duplicate leakage and ontology inconsistency jointly distort reported fire-safety object-detection performance?
- **RQ2 - interaction:** Does leakage mask or exaggerate the measured benefit of ontology correction?
- **RQ3 - external validity:** Do effects measured internally persist on a locked source-disjoint external test?
- **RQ4 - architectural robustness:** Are the direction and approximate magnitude of the effects consistent across two detector families?

Within-training duplication and hard-negative/video evaluation remain secondary analyses for a longer paper. They must not displace the factorial core or external holdout in the ICASSP submission.

## Phase 0: blockers that must be resolved before new headline training

### 0.1 Ontology adjudication

The historical class 1 contains at least two different concepts: flame boxes and smoking/face-cigarette boxes. Renaming the class to `flame` is not a correction. Every selected class-1 annotation must receive one of these decisions:

- `flame`: a visible flame with a box that reasonably localises the flame;
- `smoke`: visible smoke with a box that reasonably localises the smoke plume;
- `cigarette`: a tight cigarette box that can be mapped to class 0;
- `smoking_action`: a person/face/hand interaction box rather than a flame or tight cigarette;
- `invalid`: annotation does not match a target concept;
- `ambiguous`: evidence is insufficient and requires adjudication.

Two people should independently review at least a stratified 20% sample. Report Cohen's kappa and raw agreement. All disagreements and all `ambiguous` rows require adjudication. For the final three-class detector, `smoking_action` and `invalid` boxes are excluded rather than silently mapped to `flame` or `cigarette`.

Run:

```bash
cd training/leakage_safe_yolov5
.venv/bin/python scripts/build_ontology_review.py \
  dataset \
  audit/ontology_review.csv \
  --html audit/ontology_review.html
```

### 0.2 Provenance and licence audit

For every source group, record the originating dataset/page, licence, retrieval date, and whether redistribution is permitted. Google Images is a search mechanism, not a licence. If raw images cannot be redistributed, publish code, hashes, split manifests, annotation decisions, and acquisition instructions, and state the limitation explicitly.

### 0.3 New external holdout

The current 188-image test has already influenced project decisions and should be treated as a fixed development benchmark, not a never-seen final test. Before fitting the paper models, create and lock an external holdout from unseen buildings, cameras, or licensed datasets.

Minimum target:

- at least 100 independently sourced positive images or clips per class;
- at least three unseen scenes/cameras;
- at least six hours of negative video from three or more scenes;
- hard negatives including lights, sunset, reflection, fog, steam, clouds, pens/pencils, and unlit cigarettes;
- grouping by video/source so adjacent frames can never cross splits.

If people are identifiable, obtain the required consent/ethics approval or use licensed/controlled footage without identifiable subjects.

## Experimental datasets

All conditions use the same frozen development test and external holdout. Hyperparameters are selected using validation only.

| ID | Split | Train variants | Ontology | Empty-label train negatives | Purpose |
| --- | --- | --- | --- | --- | --- |
| D0 | historical image-level split | all | historical mixed | included | reproduce the optimistic historical condition |
| D1 | source/pHash-group split | all variants from train groups | historical mixed | included | isolate split integrity from within-train deduplication |
| D2 | same groups as D1 | one representative per group | historical mixed | included | isolate within-train deduplication |
| D3 | same records as D2 | one representative | adjudicated three-class | included | isolate ontology correction |
| D4 | same records as D3 | one representative | adjudicated three-class | excluded from train only | isolate hard-negative training |

D0 versus D1 is a robustness comparison, not a clean causal estimate, because the composition of the training data changes with the split. Likewise, D3 versus D4 is a practical diagnostic but changes the number of training images. The controlled experiments below provide the primary causal evidence.

## Controlled leakage experiment

1. Start with D3 and keep the fixed test representative from every test component.
2. Identify test components that contain at least one unused offline or near-duplicate variant.
3. For contamination doses 0%, 25%, 50%, and 100%, place one unused variant from the corresponding proportion of test components into training.
4. For every injected variant, remove one clean training image from the same label-signature stratum. This keeps training-set size constant.
5. Select contaminated components and replacements deterministically for each seed.
6. Evaluate every condition on the identical frozen representatives and on the uncontaminated external holdout.

This yields a dose-response estimate of leakage inflation while controlling training size and class stratum. The external holdout checks whether apparent gains transfer beyond contaminated sources.

## Primary factorial experiment

The primary ICASSP analysis crosses training ontology with leakage status:

| Training condition | Original mixed ontology | Audited ontology |
| --- | ---: | ---: |
| Leakage-free source/pHash-group split | A | B |
| Controlled near-duplicate contamination | C | D |

All four cells must be evaluated against the **same adjudicated canonical internal test labels** and the same adjudicated external holdout. Changing evaluation labels between cells would make AP values non-comparable. Keep the training image pool constant across the ontology factor; reassigned, excluded, and newly empty-label images are retained and logged. C and D receive identical contaminated components and size-matched replacements.

Report `B-A`, `D-C`, `C-A`, `D-B`, and the interaction `(D-C)-(B-A)` with source-group paired bootstrap intervals. The 25% and 50% doses are secondary; the four factorial cells, three seeds, and external test have priority.

## Controlled hard-negative experiment

Use the adjudicated positive training pool and draw a fixed number of total image exposures/optimizer steps in every condition. Compare fixed hard-negative sampling fractions such as 0%, 10%, and 20%; fill the remaining exposures by deterministic sampling from the same positive pool. Keep augmentation and all other training settings identical. Evaluate on the same positive test and negative image/video sets. This separates the effect of negative sampling from simply running more batches because a larger dataset was supplied.

## Models and training controls

Use one architecture for causal isolation and one second architecture to test whether the conclusion generalises.

- Primary: pinned YOLOv5s used by the existing project.
- Replication architecture: a pinned small modern detector supported on the available hardware. Freeze the choice and version before the confirmatory runs.

For every compared condition, keep initial weights, image size, batch size, optimiser/hyperparameters, augmentation, early-stopping policy, maximum epochs, and evaluation thresholds identical. Do not initialise paper models from the semantically mixed legacy checkpoint; initialise all confirmatory models from the same generic pretrained checkpoint.

Use seeds `0`, `1`, and `2` for confirmatory comparisons. A single seed may be used for screening only and must not appear as confirmatory evidence.

## Metrics

### Image metrics

- precision and recall at the declared operating point;
- AP@0.5 and AP@0.5:0.95, overall and per class;
- false positives per negative image;
- model parameters, file size, inference latency, and peak memory.

### Video and alert metrics

- false alarms per hour;
- event recall, where a positive event is a contiguous annotated fire/smoke/smoking-risk interval;
- time-to-first-detection from event onset;
- alert precision after a fixed temporal rule, for example at least 3 positive frames in a 5-frame window;
- end-to-end latency on the target device.

The confidence threshold and temporal rule must be selected on validation video and frozen before the external test.

## Statistical analysis

- Report mean and standard deviation across three training seeds.
- Compute paired 95% bootstrap confidence intervals on metric differences using source groups, not individual images, as the resampling unit.
- For video, bootstrap by video or camera session, never by frame.
- Report effect sizes and confidence intervals as primary evidence; p-values are secondary.
- Do not claim that a metric change was caused by leakage unless it comes from the controlled contamination experiment.
- Keep the external holdout locked until code, hyperparameters, thresholds, and exclusion rules are frozen.

## Minimum publishable evidence

The ICASSP paper should not be submitted until it contains:

- a completed ontology audit and adjudicated label set;
- a provenance/licence statement;
- the four-cell ontology x leakage factorial experiment with three seeds for the primary architecture;
- a second architecture on at least the four factorial endpoints, or the clean/contaminated endpoints under the audited ontology if compute is constrained;
- the untouched external holdout;
- paired confidence intervals and complete run manifests.

Negative-video false alarms per hour and the full leakage dose-response strengthen the work but are secondary for the four-page submission.

## Compute-aware execution order

1. Complete ontology review and external-data collection in parallel.
2. Run five-epoch smoke tests for dataset/config validation only.
3. Run one-seed, 25-epoch screening for D1-D4 and leakage endpoints.
4. Freeze the final experiment matrix and analysis code.
5. Run three-seed, 45-epoch confirmatory experiments.
6. Evaluate all saved checkpoints once on the locked external holdout.
7. Generate tables/figures directly from run manifests; do not hand-copy metrics.

On the current Apple M4, the historical 45-epoch YOLOv5s run took about two hours. The confirmatory core of roughly 12-18 YOLOv5s runs therefore requires about 24-36 serial GPU-hours, excluding ontology review and the second architecture.
