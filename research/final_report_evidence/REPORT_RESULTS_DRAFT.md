# Completion Report results draft

## Technical summary

The 3-seed x 4-condition factorial matrix completed all 12 planned YOLOv5s runs. Audited-label conditions B and D achieved the strongest validation mAP@0.5:0.95 means, while internal-test means ranged from 0.4587 to 0.4767.

## Factorial findings

At the final validation epoch, the ontology contrast B-A was +0.0238 mAP@0.5:0.95. The contamination contrasts C-A and D-B were +0.0102 and +0.0065, and the interaction was -0.0037. On the internal test, B-A was +0.0137, C-A was -0.0043, D-B was -0.0014, and the interaction was +0.0029.

These are descriptive comparisons. They show a consistent positive association between audited labels and internal performance, but they do not establish statistical significance or a causal effect.

## Internal performance

Across conditions, internal mAP@0.5 ranged from 0.7647 to 0.8070 and mAP@0.5:0.95 ranged from 0.4587 to 0.4767. Sample SD for internal mAP@0.5:0.95 ranged from 0.0110 to 0.0203 across the three seeds. The fixed internal test contained 188 images and 167 instances.

Condition B achieved the highest mean precision, mAP@0.5, and mAP@0.5:0.95. Condition D achieved the highest mean recall. Controlled near-duplicate contamination did not provide a stable benefit across validation and internal testing.

## Interpretation

The evidence supports completion and internal reproducibility of the factorial workflow. It also supports treating annotation-ontology auditing as an important part of trustworthy detector development. Because the retained evaluation is internal to the project collection context, the report does not claim robust performance in substantially different environments.
