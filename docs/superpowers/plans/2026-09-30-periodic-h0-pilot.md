# Periodic H0 pilot implementation plan

**Goal:** Execute the published planar experiment protocol and assemble a readable source-linked exposition, then audit this delivery and C32/C33.

**Spec:** [Published experiment](../../research-translation/20260930/EXPERIMENT.md), continued under Dylan's explicit implementation authorization. The substantive protocol and scientific boundaries are unchanged.

**Architecture:** A small standalone experiment folder separates Fourier synthesis, periodic persistence/oracle, reporting and the pilot runner. Exact mathematical sources remain in the existing source manifest. A pinned optional environment keeps research dependencies out of the stdlib verification suite.

**Tech stack:** Python, NumPy, GUDHI; exact versions pinned after checking compatible published packages. Standard-library tests where possible; actual GUDHI integration tests for topology.

## Constraints and review focus

- d=2, L=24, variance-one parent normalization approximated by a declared deterministic Fourier denominator; no per-field normalization.
- All source/model claims retain their existing assumptions and review scope. Numerical agreement is not proof; disagreement at finite resolution is not a refutation.
- Keep independent realizations as statistical units; preserve zero-count replicates and finite versus essential distinctions.
- Check periodic gluing, conjugate weights, Fourier/grid coupling, bin endpoints, nonfinite input and reproducibility.
- New results must report sampling, spatial and spectral effects separately; no fitted or hand-selected window described as a confirmation study.
- Final audit covers all changed files, prior C32/C33 source/custody/CI, public routing and new data/report consistency. It does not claim a reproof of every historic theorem.

## Tasks

1. Write failing tests for a literal periodic multipeak barcode, an independent elder-rule oracle, Fourier covariance/coupling, half-open histogram bins and exact bin-shape algebra.
2. Implement the generator and H0 extraction; run meaningful wrong-boundary, wrong-elder, wrong-spectrum, missing-variance and missing-volume controls. Check GUDHI against the oracle on deterministic and seeded small landscapes.
3. Pin the pilot before observing results: 32 independent seeds; grid sizes 64,128,256; square cutoffs 12,24; bins fixed in configuration. Use the same modes across grids and nested cutoffs. Covariance diagnostics use a separate larger independent sample. No confirmation fit is claimed.
4. Save per-realization bin counts, every positive finite interval, essential/zero diagnostics, runtime/version/seed/config metadata and source identities. Generate a summary from those observations with replicate-level standard errors and coupled grid/cutoff differences.
5. Write MANUSCRIPT.md and CROSSWALK.md from exact accepted source statements; label it a standalone exposition draft and acknowledge the existing V3 manuscript. Add actual pilot results and unresolved error/assembly questions.
6. Reproduce generated results, run focused negative controls and the complete applicable main suite on the stable tree. Obtain a nonauthor review of code/report and a separately exposed source review of the manuscript; repair findings.
7. Publish through one PR, verify exact current-head hosted checks before authorized integration, retain original formal artifacts, then one immutable R17 delivery and release.

**Execution:** Root implements code; a disjoint source auditor writes the manuscript; the prior-delivery auditor performs the final nonauthor engineering review. Their contributions and prior exposure are disclosed. This is not independent human review.
