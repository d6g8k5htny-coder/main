# PHASE-2 SCOPE DECISION (lead mandate 19fb9293, option (a) with (c)-scoping)

## Scope: J3 certification runs on the Phase-1-certified cells only
(interior 623 + clip-1 214 = 837 of 1707 grid-of-record cells).

## Dependency justification (mandated: no silent scope reduction)
1. Every Phase-2 certificate is POINTWISE-PER-CELL: cell C's sigma-sup
   certificate closes over C's own box (its exact J3 center jets, its
   region-level S4R, its Phase-1 6-block sups). No cell's certificate
   formula references any other cell's sigma(D^2 lam)/sigma(D^3 lam) sups.
2. The only downstream consumption of per-cell sups is the certificate's
   REPORTING aggregate (max over cells, rung ratios Q) — an aggregate over
   the certified set, not a dependency of any cell's certificate. The
   v4 certificate will state the conclusion's domain as EXACTLY the
   certified cell set, with the failed-cell ledger as its boundary
   (mirroring the v2/v3 disposition language).
3. The 864 Phase-1-ledgered cells CANNOT enter the J3 route without
   further subdivision regardless: clip-mixed cells (783) have the m-range
   straddling a clip kink — the lambda formula's branch is genuinely
   ambiguous over the cell (a mathematical branch ambiguity, not
   arithmetic); denominator-straddle cells (81) have c6v = 0 inside the
   cell (the 6-block certificate itself fails). Both categories remain
   ledgered exactly as Phase 1 recorded them; the v4 certificate names
   this consumption chain explicitly: ledgered cells contribute NO sups
   and the rung-conclusion EXCLUDES their boxes.

## REFUSED-RUN RECORD (lead-mandated ledger entry)
The scalar-SV full sweep (31h projected, all-ledger outcome) was REFUSED
and never launched. Reason (receipted): the scalar chain's 'basis
straddle' ledger entries would be FALSELY ATTRIBUTED — the measured TRUE
Delta G09 over the test cells is 0.0182 (tame) while the scalar chain
certifies only 3536 (2e5x inflation through the W-congruence); an
all-ledger result would misrecord soundly-certifiable cells as failures.
Parameter changes on record: subdivision depth cap 3 -> 4 (ledger reason
strings carry '(depth 4)'); per-cell sigma^4 sup box = depth-0 cell box,
reused soundly at all depths; signed-center fixes (Phase-1 clip m0,
Phase-2 JetCache/tp_sigmas center) receipted as bug fixes 4-5.
