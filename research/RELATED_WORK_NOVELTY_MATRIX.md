# Related work and novelty matrix

Status: literature-positioning draft, 2026-08-14

## Bottom line

The paper should **not** claim that it is the first to show that train/test leakage inflates object-detection metrics, the first to use perceptual hashes for near-duplicate auditing, the first to relabel a fire/smoke dataset, or the first to use external/cross-dataset evaluation. Each of those ideas has direct prior work.

The defensible research gap is narrower and more interesting:

> Existing work studies leakage, annotation quality, and fire/smoke benchmark generalisation mostly in isolation. This study tests whether near-duplicate leakage changes the apparent benefit of ontology correction, and whether either conclusion survives source-disjoint external evaluation in a safety-oriented detector.

This is an **interaction and evaluation-validity** paper, not a new-detector paper.

## Search scope

This is a rapid primary-source positioning review, not a formal systematic review. The screening covered 31 closely related works across four lines:

1. visual train/test leakage and near-duplicate auditing;
2. fire/smoke datasets and cross-scene evaluation;
3. object-detection label-error detection/correction;
4. out-of-distribution and source-disjoint evaluation.

The matrix below retains the closest works. A check means that the paper makes the dimension a substantive part of its design, not that it merely mentions it.

## Closest-work matrix

| Work | Fire/smoke | Controlled leakage | Near-duplicate audit | Semantic/label audit | Source-disjoint or external evaluation | Tests leakage x ontology interaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Babu et al., 2024, *Improving Image Data Leakage Detection in Automotive Software* | No | Yes, 0--100% | Yes, pHash | No | Cross-dataset method check, not a locked deployment holdout | No |
| Ramos et al., ICCVW 2025, *Data Leakage in Visual Datasets* | No | Not this design | Yes, hard/soft leakage | Discusses conflicting leaked labels | Multiple visual datasets | No |
| Lilja et al., CVPR 2024, *Localization Is All You Evaluate* | No | No | Spatial/sequence leakage | No | Geographically separated mapping data | No |
| Adimoolam et al., CVPR 2026, *Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets* | No | No | Yes | Dataset-quality audit | Cross-split/OOD consequences | No |
| Zhang et al., IEEE 2022, *Unified Smoke and Fire Detection in an Evolutionary Framework* | Yes, classification | No | Samples/removes duplicate frames | Relabels smoke/flame and removes ambiguity | Two surveillance/Internet test sets | No |
| Han et al., PRCV 2024, *Benchmarking Multi-Scene Fire and Smoke Detection* | Yes | No | Removes redundancy | Relabels and standardises | Independent test benchmark | No |
| Sharobiddinov et al., Sensors 2025, *Edge-Based Autonomous Fire and Smoke Detection Using MobileNetV2* | Yes, classification | No | MD5/pHash audit | No ontology intervention | Cross-dataset test | No |
| Liu et al., 2025, *DetectiumFire* | Yes | No | Redundancy reduction | Expert/multimodal annotation | Cross-dataset task evaluation | No |
| Ma et al., CVPRW 2022, *The Effect of Improving Annotation Quality on Object Detection Datasets* | No | No | No | Professional reannotation | No dedicated external holdout | No |
| Tkachenko et al., ICML DMLR 2023, *ObjectLab* | No | No | No | Automated missing/bad-box/class-error triage | Multiple datasets/models | No |
| Schubert et al., WACV 2024, *Identifying Label Errors in Object Detection Datasets by Loss Inspection* | No | No | No | Four error types, synthetic and real | Multiple datasets | No |
| Penquitt et al., 2025/2026, *From Label Error Detection to Correction (REC✓D)* | No | No | No | Human-validated correction and ambiguity | KITTI case study | No |
| Hasan et al., CVPR 2021, *Generalizable Pedestrian Detection* | No | No | No | No | Direct cross-dataset detection | No |
| Mao et al., ICCV 2023, *COCO-O* | No | No | No | No | Six natural distribution shifts | No |
| Dewangan et al., Remote Sensing 2022, *FIgLib & SmokeyNet* | Wildfire smoke | No | Video/camera grouping is relevant | Human labels | Fixed-camera real-world sequences | No |
| **Proposed study** | **Yes** | **Yes** | **Yes** | **Yes, semantic ontology adjudication** | **Yes, locked unseen sources** | **Yes, primary estimand** |

## What is already occupied

### Controlled object-detection leakage

Babu et al. already ask how incremental leakage affects object detection. Their YOLOv7 experiment introduces leakage from 0% to 100%, reports precision, recall, mAP and F1, and uses pHash/manual inspection to diagnose visually similar automotive frames. Therefore, a contribution phrased as “we quantify how leakage increases mAP” is not novel enough on its own.

### Fire/smoke dataset cleaning and benchmark construction

Zhang et al. already sample video frames, remove duplicate or highly ambiguous examples, relabel smoke/flame, construct difficult negatives and evaluate on additional surveillance/Internet videos, although their task is multi-label image classification rather than bounding-box detection. Han et al. explicitly identify irregular construction, redundant video-derived imagery, inconsistent labels and weak evaluation in public fire/smoke datasets. They collect, screen, relabel and standardise data and establish an independent test benchmark. The Sensors 2025 MobileNetV2 study also performs MD5/pHash duplicate auditing and cross-dataset classification. FASDD, FIgLib/SmokeyNet, Nemo, FireAndSmoke, DetectiumFire, GWFP and other recent datasets emphasise diversity, realistic negatives or cross-domain evaluation. Therefore, “we build a cleaner fire dataset” is insufficient by itself.

### Object-detection annotation audit

Ma et al., ObjectLab, Schubert et al. and REC✓D already study incorrect classes, missing objects, bad localisation, ambiguity and human correction workflows. The novelty here cannot be merely that manual review found incorrect annotations.

### External/OOD evaluation

Generalizable Pedestrian Detection, COCO-O, RWDS and camera-trap work demonstrate that in-domain detector rankings may fail under unseen environments. External testing is necessary evidence, but not a standalone novelty claim.

## Proposed research questions

- **RQ1 -- joint distortion:** To what extent do near-duplicate leakage and ontology inconsistency jointly distort the reported performance of fire-safety object detectors?
- **RQ2 -- interaction:** Does leakage mask or exaggerate the measured benefit of ontology correction?
- **RQ3 -- external validity:** Do effects measured on the internal benchmark persist on a locked, source-disjoint external test?
- **RQ4 -- architectural robustness:** Are the direction and approximate magnitude of the effects consistent across two detector families?

## Primary 2 x 2 factorial design

Use a single, adjudicated canonical test ontology for **all** four cells. Otherwise, changing test labels at the same time as training labels makes AP values non-comparable.

| Training condition | Original mixed training ontology | Audited training ontology |
| --- | ---: | ---: |
| Leakage-free group split | A | B |
| Controlled near-duplicate contamination | C | D |

Controls:

- A--D use the same training image identities before controlled leakage injection;
- audited training may reassign/drop boxes, but must retain the same image pool, including images that become empty-label negatives;
- C and D receive the same contaminated source components and size-matched replacements;
- all models use the same generic initialisation, schedule, augmentation and seeds;
- all four cells are evaluated on the same adjudicated internal test and the same adjudicated external holdout;
- the external holdout is never used for model/threshold selection.

Primary effects:

- ontology effect without leakage: `B - A`;
- ontology effect with leakage: `D - C`;
- leakage effect under original ontology: `C - A`;
- leakage effect under audited ontology: `D - B`;
- interaction: `(D - C) - (B - A)`.

Estimate each difference with source-group paired bootstrap intervals. The interaction, not the raw 0--100% curve, is the paper's primary estimand.

The 25% and 50% leakage doses can remain a secondary dose-response analysis under the audited ontology. They are useful but should be dropped before the 2 x 2 core, three seeds, or external holdout if time becomes constrained.

## Three defensible contributions

1. **Joint causal audit.** A controlled factorial study that separates and estimates the individual and interaction effects of near-duplicate leakage and semantic ontology inconsistency in fire-safety object detection.
2. **External-validity diagnosis.** A matched comparison showing whether internal benchmark changes transfer to a locked, source-disjoint test from unseen buildings/cameras, with source-cluster uncertainty estimates.
3. **Reproducible audit artifact.** Versioned source groups, exact/perceptual duplicate manifests, adjudicated ontology decisions and fixed experiment recipes that make the evaluation auditable even when all raw web imagery cannot be redistributed.

Do not claim a new pHash algorithm, new label-error detector, new fire detector architecture, or the first cleaned fire dataset.

## Recommended title and hook

Preferred title:

> **Disentangling Data Leakage and Label-Semantic Bias in Fire-Safety Object Detection: A Controlled Factorial Study**

More attention-grabbing alternative:

> **When High AP Lies: Leakage and Label Semantics in Fire-Safety Object Detection**

The first figure should show the 2 x 2 interaction for both internal and external evaluation. The ideal result pattern is not assumed in advance, but the figure should make this question visually immediate: does ontology correction appear small under a contaminated benchmark yet become substantial under leakage-free/external evaluation?

## Draft related-work text

### Leakage and duplicate-aware evaluation

Visual benchmark leakage ranges from exact overlap to perceptually or semantically near-identical samples. Recent studies show that hard and soft leakage compromise evaluation across visual datasets, while domain-specific work demonstrates similar risks in sequential automotive, mapping and geospatial imagery. Most directly, Babu et al. inject increasing proportions of test imagery into object-detector training and use perceptual hashing to analyse near-duplicate automotive frames. Our work therefore does not treat leakage-induced AP inflation as a new phenomenon; instead, it asks whether leakage changes the conclusions drawn from a concurrent semantic-ontology audit.

### Annotation quality and ontology consistency

Object-detection annotations contain missing objects, class swaps, localisation errors and irreducible ambiguity. Prior work proposes model-based triage, loss-based detection, professional reannotation and crowd-assisted correction. These studies focus primarily on finding or fixing annotation errors. We study a complementary evaluation question: when a safety-related class conflates visually and operationally different concepts, how does that semantic inconsistency interact with near-duplicate leakage, and which apparent gains survive evaluation against a common adjudicated ontology?

### Fire/smoke benchmarks and external validity

Fire and smoke detection research increasingly recognises that redundant video frames, inconsistent dataset construction and narrow scene coverage make reported performance difficult to compare. Multi-scene benchmarks relabel and standardise public resources, while recent fixed-camera, remote-sensing and cross-domain datasets stress realistic scene diversity. Our study does not propose another detector benchmark in isolation. It uses a locked source-disjoint holdout to test whether conclusions produced by contaminated and ontology-inconsistent internal evaluations generalise to unseen safety-monitoring scenes.

## Screened bibliography by theme

### Leakage, duplication and evaluation validity

1. Babu et al. (2024), *Improving Image Data Leakage Detection in Automotive Software*.
2. Ramos et al. (ICCVW 2025), *Data Leakage in Visual Datasets*.
3. Lilja et al. (CVPR 2024), *Localization Is All You Evaluate: Data Leakage in Online Mapping Datasets and How to Fix It*.
4. Adimoolam et al. (CVPR 2026), *Data Leakage Detection and De-duplication in Large Scale Geospatial Image Datasets*.
5. Sun et al. (ICCV 2017), *Revisiting Unreasonable Effectiveness of Data in Deep Learning Era* (supplementary near-duplicate analysis).

### Label quality and correction

6. Northcutt et al. (NeurIPS Datasets and Benchmarks 2021), *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks*.
7. Ma et al. (CVPRW 2022), *The Effect of Improving Annotation Quality on Object Detection Datasets: A Preliminary Study*.
8. Tkachenko et al. (ICML DMLR 2023), *ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data*.
9. Schubert et al. (WACV 2024), *Identifying Label Errors in Object Detection Datasets by Loss Inspection*.
10. Bär et al. (CVPRW 2023), *A Novel Benchmark for Refinement of Noisy Localization Labels in Autolabeled Datasets for Object Detection*.
11. Koksal et al. (CVPRW 2020), *Effect of Annotation Errors on Drone Detection With YOLOv3*.
12. Penquitt et al. (2025/2026), *From Label Error Detection to Correction: A Modular Framework and Benchmark for Object Detection Datasets*.
13. Tschirschwitz and Rodehorst (WACV 2025), *Label Convergence: Defining an Upper Performance Bound in Object Recognition through Contradictory Annotations*.
14. Llerena et al. (WACV 2025), *Noise-Aware Evaluation of Object Detectors*.
15. Goswami et al. (NeurIPS 2023), *AQuA: A Benchmarking Tool for Label Quality Assessment*.

### Fire/smoke data and evaluation

16. Han et al. (PRCV 2024), *Benchmarking Multi-Scene Fire and Smoke Detection*.
17. Zhang et al. (IEEE 2022), *Unified Smoke and Fire Detection in an Evolutionary Framework with Self-Supervised Progressive Data Augment*.
18. Cao et al. (2024), *An Open Flame and Smoke Detection Dataset for Deep Learning in Remote Sensing Based Fire Detection* (FASDD).
19. Dewangan et al. (Remote Sensing 2022), *FIgLib & SmokeyNet*.
20. Govil et al. (Remote Sensing 2022), *Nemo: An Open-Source Transformer-Supercharged Benchmark for Fine-Grained Wildfire Smoke Detection*.
21. Yar et al. (Sensors 2024), *A Comparative Performance Evaluation of YOLO-Type Detectors on a New Open Fire and Smoke Dataset*.
22. Sharobiddinov et al. (Sensors 2025), *Edge-Based Autonomous Fire and Smoke Detection Using MobileNetV2*.
23. Liu et al. (2025), *DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding*.
24. Hamdan et al. (2026), *A Large Scale Open-Source Image and Video Dataset for Robust Wildfire Detection and Classification* (GWFP).
25. *Cross-Domain Fire Detection Across Indoor and Outdoor Scenes* (Sensors 2026).

### External and distribution-shift evaluation

26. Beery et al. (ECCV 2018), *Recognition in Terra Incognita*.
27. Hasan et al. (CVPR 2021), *Generalizable Pedestrian Detection: The Elephant in the Room*.
28. Mao et al. (ICCV 2023), *COCO-O: A Benchmark for Object Detectors under Natural Distribution Shifts*.
29. Al-Emadi et al. (CVPR 2025), *Benchmarking Object Detectors under Real-World Distribution Shifts in Satellite Imagery*.
30. Zhou et al. (CVPR 2022), *Simple Multi-Dataset Detection*.
31. Humblot-Renaux et al. (CVPR 2024), *A Noisy Elephant in the Room: Is Your Out-of-Distribution Detector Robust to Label Noise?*

## Decision for ICASSP 2027

The paper is viable only if the factorial interaction and external-validity evidence are completed. If the final result is merely that contamination increases internal AP, the work is too close to Babu et al. If the result is merely that relabelling improves a fire detector, it is too close to the broader annotation-quality and fire-benchmark literature. A null interaction can still be publishable if estimated precisely and accompanied by a meaningful internal-versus-external discrepancy, but it must be reported honestly.
