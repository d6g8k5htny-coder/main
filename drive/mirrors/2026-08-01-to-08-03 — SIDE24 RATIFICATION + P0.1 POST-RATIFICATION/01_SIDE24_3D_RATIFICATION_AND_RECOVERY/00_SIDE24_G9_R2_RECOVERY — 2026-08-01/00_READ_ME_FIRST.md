# SIDE24 G.9 / R2 recovery and reconstruction annex

**Date:** 2026-08-01  
**Operator:** Dylan M. Roy  
**Purpose:** recover the carriers reported missing in AO48-REC-034, preserve
their exact provenance, and reconstruct only the one later mathematical item
whose original bytes cannot be located.

## Controlling result

The reported loss was almost entirely a discovery failure.

1. The original R2 archive exists in Google Drive and has been recovered
   byte-for-byte. Its SHA-256 is
   `14eaf7d403cd8c103576807719e8ee031cad1fc5bba58f1c37574a790849e44b`,
   exactly matching its checksum companion. The archive passes `unzip -t`,
   and all 68 rows of its internal checksum ledger verify.
2. The full V5 `SIDE24_GAP_FILL_SUPPLEMENT.md` and historical
   `verifier/v8` tree exist byte-for-byte in the operator-provided
   `OKComputer_Project_Gap_Closure(1).zip`. Their hashes match the V5 package
   manifest. A fresh v8 run reports 30/30 and `ALL_ASSERTIONS_PASS`.
3. The complete ten-file C030/C031 LB-RATE set is already filed in Google
   Drive. All six JSON files parse. Each of the four markdown files with a
   located July predecessor is byte-identical to that predecessor. An
   independent recovery from `q0_machine_active.json` additionally makes four
   JSON carriers byte-identical and all six semantically identical to the
   Drive copies.
4. The later sandbox G.9.1 excerpt reviewed by AO48-AUD-033 is not present as
   a standalone byte carrier. Its exact wording and original hash are therefore
   not recoverable. Its mathematical content has been independently rebuilt in
   `recreated/G9_1_CORRECTED_RECONSTRUCTION.md`, without claiming identity to
   the absent file.

## Corrected G.9.1 result

For the planar three-dimensional Bargmann–Fock field, let
\(q=e^{-d^2/2}\), \(U=u_1^2\), and \(P=u_2^2+u_3^2\). After conditioning on
\((f,\nabla f)\) at the two axial points, define
\[
g=\frac{d^2q^2+q^2-1}{q^2-1}
\]
and
\[
v=\frac{-d^6q^2+d^4q^4+d^4q^2+4d^2q^4-4d^2q^2
+2q^4-4q^2+2}
{(-d^2q+q^2-1)(d^2q+q^2-1)}.
\]
The reconstructed exact determinant is
\[
\det\Gamma_d(u)
=(P+gU)\bigl(2gP^2+2vUP+gvU^2\bigr).
\]
It reproduces every AO48 acceptance fixture. It also gives the correct
coalescence behavior
\[
g\sim d^2/2,\qquad v\sim d^4/6,
\qquad \det\Gamma_d(u)\to0
\]
in every direction, with axial rate \(d^8U^3/24\). This confirms the
AO48-AUD-033 refutation of the discarded generic \(O(1)\) limit.

The reconstruction verifier performs 26 checks:

- Hermite derivation of the pin covariance;
- two exact adjugate identities in \(\mathbb Z[d,q]\);
- corrected Schur variances and determinant;
- transverse, axial, generic coalescence, and decoupled faces;
- explicit rejection of the historical false pin determinant and false
  \(4UP^2\) limit;
- four high-precision fixtures computed both by the closed form and by an
  independent eight-pin Gaussian-conditioning pipeline.

Normal and `python -O` transcripts are byte-identical. A forced fixture
mutation exits with status 1. The verifier contains no bare `assert`.

## Directory map

- `recovered_original/` — exact recovered carriers: R2, the operator V5 ZIP,
  full supplement, rendered supplement, and verifier v8.
- `recreated/` — the explicitly labeled corrected G.9.1 reconstruction,
  fail-closed verifier, and transcripts.
- `lower_bound_cycle/` — the ten files currently present in the C030/C031
  Drive folder.
- `provenance/` — AO48 records and the independent Drive provenance report.
- `successor/` — the V3.4 RP-C/RP-S facewise closure checkpoint and its main
  proof/status surfaces.

## Status boundary

This annex repairs preservation and discoverability. It does not silently
promote any theorem.

- Historical v8 proves only its packaging/regression criteria and explicitly
  preserves the V5 state `RP-C/RP-S OPEN`.
- The corrected planar G.9.1 identity is an exact local covariance theorem; it
  does not alone supply uniform side-24 Schur constants.
- V3.4 is the later facewise side-24 closure successor and remains
  `PROVED_IN_V3_4_AUDIT_PENDING` / package `HOLD` until independent expert
  review of the analytic Fourier/Schur, conditional-moment, and Kac–Rice
  arguments.

## Drive locations already verified

- Exact R2: Drive ID `19ThSFctrr7lz0B7YNuaqLFOQaGx8E_UR`.
- R2 checksum companion: `1l2xaXz2rGCUER6O0MSwkZhvrmSXzcFX3`.
- C030/C031 ten-file folder: `1RvyeNGThLxtGsnTLREhgw_bT5NEpw8XM`.
- AO48-AUD-033: `135ZKlWDnAuMXZlWK7gN3Scra67ShrcL9`.
- AO48-REC-034: `12pnNmP7Uznn8p6km2c5r74qhxlShC3MQ`.

See `provenance/DRIVE_PROVENANCE_RECOVERY_REPORT_2026-08-01.md` for complete
IDs, URLs, sizes, timestamps, hashes, and retrieval checks.

One unrelated archive named `research_formal_core_r2.zip` was also found and
validated during the exhaustive sweep. It is a 10,644-byte Lean/formal-core
package with SHA-256 `a6440511...`; it is not the 2,615,005-byte SIDE24 R2 and
is intentionally excluded from this annex to prevent identity drift.
