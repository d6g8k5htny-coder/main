# H5 totals — VERSION FREEZE note (2026-09-15, Stage-E repair R4)

Discipline (lead, Stage-E numerics review): consumed artifacts are
version-frozen; every merge writes a NEW versioned file, never overwriting.

## Files

- `h5_totals.json` — sha256 f3e672a3cb4d4016a117624493a8ce766c5bf21e1a7697b840da82e896f98ecf.
  NOTE (honest lineage): these bytes are the DRIFTED content (I_lo =
  1.2328893552e-08, I_hi = 8.0975589252e-02), not the v1 consumed content.
  The v1 consumed content (I_lo = 1.166721190e-08, I_hi = 9.142887704e-02 =
  731.43 r^3) is preserved inside D1 v2.0's frozen body (86882dca…), in
  H5_CLOSURE.md's frozen headline (body 465c97c0553ecc0a…), and in
  H5_RECEIPTS.txt; it is reproducible from the deterministic banked records
  up to record-append timestamps (the drift direction is downward-only —
  refine-3 min-per-sector gains plus the documented sdisk 0.06→0.062 area
  delta; the v1 interval remains valid: v1 I_hi strictly exceeds every later
  version's I_hi, and v1 I_lo strictly precedes every later I_lo).
  Per the lead's instruction this file is NOT regenerated/touched further.
- `h5_totals_v2_2026-09-15.json` — sha256 f3e672a3cb4d4016a117624493a8ce766c5bf21e1a7697b840da82e896f98ecf
  (byte-identical copy of the drifted state at the freeze moment; the
  tightening amendment H5_CLOSURE_TIGHTENING_2026-09-15.md cites THIS file).
- `h5_totals_v2.json`, `h5_totals_v3.json`, … — future merges, written by
  h5_merge.py with 'x'-mode (fail-closed if the target exists;
  auto-incrementing; reproduction/display runs pass write_totals=False and
  mint nothing).

## Merger change (CODE_HASHES.txt regenerated)

h5_merge.py assemble(): versioned writes only; h5_totals.json is never
opened for writing anywhere in the tree (grep-verified at freeze).
