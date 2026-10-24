# Coupled resolution study

Design recorded on 30 September 2026 before executing this study. The earlier
32-field pilot already revealed short-bin drift; this is a diagnostic extension
using its first eight seeds in their original order, not a fresh confirmation.

Use the same maximum-24 Fourier bank for every cutoff-24 field at grids 128,
256, 512 and 1024. Compare GUDHI periodic vertex cubical H₀ with the H₀ of two
fixed-diagonal periodic triangulated PL filtrations, using the same samples.
The one-skeleton suffices for H₀; no higher-homology claim is made. Retain every
positive finite interval, the sole essential component, zero-length accounting,
all original bins, per-field Hessian diagnostics and pairwise bottleneck distances.
Report paired changes, not fictitious extra independent fields. No slope fit,
window optimization or deletion of zero-count fields is allowed.

Before the study, check actual GUDHI barcodes against a separate union-find
implementation for both diagonal orientations and tied fixtures. A two-peak
saddle fixture must distinguish the orientations. An adapted minimum-vertex
triangulation must match cubical H₀. Exact rational bin-sandwich controls must
refuse an upper bound when a bar can disappear into the diagonal.

A floating evaluation of the analytic Hessian majorant is a **diagnostic**:
it does not enclose FFT errors, coefficient evaluation, or omitted random modes.
The deterministic statement in [APPROXIMATION.md](APPROXIMATION.md) requires
those inputs to be certified before its bound can certify this implementation.

A candidate range for a later study must survive at least two finer coupled
levels and both diagonal choices, with interpolation error small relative to
its bin endpoints/widths and sampling precision adequate. These are necessary
screening conditions, not an acceptance test or a theorem. This exploratory
sample cannot establish an asymptotic remainder window. If it supplies no
usable range, record that outcome and leave confirmation pending.
