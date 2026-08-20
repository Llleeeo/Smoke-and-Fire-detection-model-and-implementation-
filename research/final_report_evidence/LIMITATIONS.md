# Limitations

1. The factorial statistics are descriptive across three seeds. No confidence intervals, hypothesis tests, multiplicity correction, or causal identification were performed.
2. The fixed internal test belongs to the project's collection context and is not an independent cross-domain benchmark.
3. Per-run internal metrics were not saved as text or CSV. Only the verified condition mean +/- sample SD survives.
4. The current audit fully adjudicates the 498 historical class-1 records; existing class-0 and class-2 annotations were not independently re-audited at the same level.
5. Checkpoint binaries and private datasets are intentionally excluded from the public repository. The public evidence package therefore relies on code, manifests, hashes, summaries, and locally retained checkpoints.
6. The escape-route simulation remains a standalone prototype. A tested detection-to-routing interface was not implemented.
7. The internal results do not establish robust performance under substantially different cameras, lighting, viewpoints, object scales, or background conditions.
