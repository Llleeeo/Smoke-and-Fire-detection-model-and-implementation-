# Report source notes

- Audience: product stakeholders (executive report specification).
- Delivery mode: portable HTML fallback because the MCP report renderer is not callable in this desktop session.
- Required structure mapping: title, Executive Summary, three evidence findings, recommended next steps, further questions, and caveats are all visible sections.
- Chart map: `removed_reasons_chart` answers why records were removed; horizontal bar; reason/count; single-root blue palette; rendered in `report.html`.
- The split and leakage evidence uses tables because exact lookup is the main job and only three split pairs exist.
- No trend chart is used because the evidence is a one-time before/after audit rather than a temporal series.
- Source evidence: `summary.json`, `verification.json`, manifests, and the executed `data_quality_audit.ipynb`.
- Portable report validation and structural verification passed; browser-level responsive/source-dialog QA did not run because no compatible Chromium executable was available.
