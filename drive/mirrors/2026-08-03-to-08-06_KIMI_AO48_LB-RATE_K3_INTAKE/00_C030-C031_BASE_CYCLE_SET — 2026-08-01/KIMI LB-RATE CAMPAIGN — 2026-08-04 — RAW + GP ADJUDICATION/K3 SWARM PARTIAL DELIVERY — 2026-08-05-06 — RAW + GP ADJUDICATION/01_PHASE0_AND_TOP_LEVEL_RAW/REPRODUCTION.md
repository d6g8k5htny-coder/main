# REPRODUCTION — K3 swarm clean-room guide

## Environment

- Python 3.12+, mpmath ≥ 1.3 (no other required packages for the core certificates; numpy only where a
  certificate declares it). No network access needed. Deterministic: no wall-clock, no randomness without
  a fixed seed printed in the transcript.
- Layout: this directory (K3_SIDE24_LB/) is self-contained for the K3 artifacts; corpus inputs are read
  from the intake inventory (INPUT_MANIFEST.sha256) by RELATIVE reference recorded in each certificate's
  header; every input's whole-file sha256 is verified by the certificate before use.

## Certificate discipline (all K3 certificates)

- Fail-closed: checks call ck(cond, msg); any failure prints and raises SystemExit(1). No bare asserts,
  no swallowed exceptions.
- Two modes: `python3 <script>` and `python3 -O <script>` must both exit 0 with byte-identical PROGRAM
  output; runner labels and exit receipts are stored as SEPARATE files (never inside the transcripts).
- Transcripts: `<name>_normal.txt`, `<name>_O.txt`, receipts `<name>_exit_normal.txt`, `<name>_exit_O.txt`.
- Mutation tests (each must fail closed with nonzero exit): (m1) restore the missing WP square-root
  defect; (m2) change a determinant/type sign; (m3) perturb a pin or the normalization; (m4) omit an
  r interval; (m5) corrupt an input hash.

## Reproduce (per workstream; fill paths as artifacts land)

1. WP symbolic (frozen): `python3 W2_symbolic/W2_sanity.py` (114 checks, exit 0); second derivation in
   LEAD_SYMBOLIC_WP.md (text; no code).
2. WP rigorous numerics: `python3 W3_numerics/<cert>.py` then `python3 -O W3_numerics/<cert>.py`;
   `cmp` the program outputs; mutation driver `W3_numerics/mutation_driver.py` (all mutations exit 1).
3. WP independent implementation: `python3 W4_independent/<cert>.py` (both modes; byte-identical).
4. Uniform-in-r: `python3 W6_uniform_r/<cert>.py` (both modes).
5. γ-LOC: `python3 W7_gammaloc/verify_k3_w7_gammaloc_v1.py` (both modes) and
   `python3 W7_gammaloc/mutation_driver.py` (1 pristine accept + 8 mutated rejects + truncation).
6. Λ-side: `python3 W8_lambda/<cert>.py` (both modes) — pending W8 successor.
7. Scope audits: `python3 W9_scope_027c/verify_ridge_scope_continuation_v1.py` (73 checks, both modes,
   4 mutations); W9_scope_027a pending.
8. Dependency minimization: `python3 W10_dependency/w10_sanity.py` (S1–S7, both modes).

## Hash verification

`sha256sum -c INPUT_MANIFEST.sha256` (intake), and `sha256sum -c MANIFEST.sha256` (final, when minted).
Every consumed input must match before any certificate runs (gate G0).

## Resumption

If compute exhausts mid-campaign: each certificate's header records its completed checks; rerun any
artifact from its script; the gate register (CANONICAL_STATE.json) records which gates have passed.

## Dependency note (W13 finding F5)

W2_symbolic/W2_sanity.py additionally imports scipy (1.16.x) for special functions; all other certificates need only mpmath (and numpy where declared). scipy is not available to a bare-bones clean room; if absent, W2's script must fail loudly at import (it does) — install scipy or skip that one certificate with the reason recorded.
