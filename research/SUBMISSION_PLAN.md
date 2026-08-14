# Submission plan as of 2026-08-14

## Recommendation

Prepare the experiments to an ICASSP-level standard, but use a result-based decision gate rather than choosing the venue now.

### Route A: ICASSP 2027 - stretch target

- Deadline: 2026-09-16.
- Format: four technical pages plus an optional references-only fifth page.
- Fit: computer vision, image/video processing, machine learning, and applied signal-processing systems.
- Required story: a concise controlled-leakage result with a clean external holdout and evidence that the effect generalises across two detector families.
- Risk: the deadline leaves about one month for 498 annotation decisions, external data, confirmatory runs, and writing.
- Official call: <https://2027.ieeeicassp.org/call-for-papers/>

### Route B: IEEE ICCE 2027 - pragmatic conference target

- Deadline: 2026-09-15.
- Format: 2-page or up-to-6-page initial submission.
- Fit: Audio/Video Systems and Signal Processing, Internet of Things and Edge AI, or Advanced AI for Consumer Electronics.
- Required story: trustworthy camera monitoring, edge-device performance, and false-alarm evaluation. The audit contribution should support the applied system story.
- Official call: <https://icce.org/2027/call-for-papers-2/>
- Official guidelines: <https://icce.org/2027/submission-guidelines/>

### Route C: Fire Safety Journal - longer-form fallback/preferred quality route

- Rolling journal submission rather than a September conference deadline.
- Fit: active fire protection systems including detection, quantitative fire-risk and fire-safety engineering.
- Required story: stronger external video evidence, operational false-alarm and time-to-detection metrics, and a clear fire-safety implication rather than only computer-vision AP.
- Official scope: <https://www.sciencedirect.com/journal/fire-safety-journal>

## Decision gates

### 2026-08-21

- Ontology pilot complete for at least 100 class-1 rows.
- Second reviewer confirmed.
- External holdout sources and permissions confirmed.
- Five-epoch smoke tests pass.

If any item fails, stop aiming at the September conference deadlines and use the journal route.

### 2026-09-02

- Full ontology adjudication complete.
- One-seed screening experiments complete.
- External test locked.
- Draft figures show a clear and defensible effect.

Choose ICASSP only if the controlled leakage experiment is the dominant result and transfers across architectures. Choose ICCE if the operational system/false-alarm result is stronger. Otherwise continue toward Fire Safety Journal.

### 2026-09-09

- Three-seed confirmatory runs complete.
- Bootstrap analysis complete.
- All numbers generated from manifests.
- Full draft complete and reviewed by the supervisor.

No simultaneous submission is permitted. Conference attendance and full-author registration costs must be approved before submission.

## Four-week execution calendar

| Dates | Experimental work | Paper work |
| --- | --- | --- |
| Aug 14-21 | ontology pilot, provenance audit, external data lock, smoke tests | related-work matrix, methods draft |
| Aug 22-Sep 2 | complete adjudication, D1-D4 screening, leakage endpoints | introduction, figures and table generators |
| Sep 3-9 | three-seed confirmatory runs, second architecture, bootstrap | full draft, limitations, supervisor review |
| Sep 10-15/16 | final verification only; no test-driven tuning | venue format, compliance checks, submission |
