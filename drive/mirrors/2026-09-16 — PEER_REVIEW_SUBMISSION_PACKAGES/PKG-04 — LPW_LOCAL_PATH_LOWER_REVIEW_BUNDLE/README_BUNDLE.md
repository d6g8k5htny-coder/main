# IMMUTABLE REVIEW BUNDLE — LPW-CAND-20260912 KIMI independent review (2026-09-13)

Contents (whole-file sha256 in MANIFEST.sha256):

- KIMI_LPW_REVIEW_VERDICT.md — the review verdict (APPROVE, all six interfaces PASS WITHIN SCOPE).
  Body hash d399e28378d177f88d08032c81f1573f35c9af2b600b9b775ebfe40f555cbbad; whole-file
  04bcbdf2013d83b149a8933a332a078c303af347335ee40f154f37ba195ad448.
  EXTRACTION RULE for the declared body hash: the body is everything before the line
  "SHA-256 of this report body (text after this line is excluded):"; hash = SHA-256 of those body
  bytes (UTF-8, LF); the marker line and the hex line after it are excluded.
- KIMI_LPW_VERDICT_ADDENDUM.md — corrections and clarifications (body ca58855f8344c1c1; whole
  6ead3d49fb2c3d6c): Consequence 4 (two-sided) WITHDRAWN for dimension/law mismatch; numerical
  certification repairs (downward rounding; the proved endpoint floor 0.124 adopted); Schur
  planar-reference note; independence clarification (one external Kimi provider-family review
  containing five task/implementation branches with reported author-script independence — NOT five
  demonstrated mutually independent external confirmations); R3 addendum confirmed; RA's source
  residue closed byte-exactly; K3-THM-001 separateness acknowledgment with two owned defects.
- LEAD_EVIDENCE_PACK.md — the lead's independent evidence pack (interim snapshot: its final section
  predates RC's completed PASS; chronology label: interim evidence, supplemented by the final verdict).
- RA_R1_R6.md — RA's R1+R6 report (includes its 12-check transcript embedded in the report body).
  Packaging note: RA did not persist a standalone script to the shared tree; its checks are reproduced
  in the report's transcript section, and the lead's independent sympy pass (LEAD_EVIDENCE_PACK §A/B/C)
  covers the same algebra.
- RB_R2.md + RB_R2_part1_symbolic.py + RB_R2_part2_spectral.py + p1_normal.out/p1_opt.out +
  p2_normal.out/p2_opt.out — RB's R2 report, scripts, and both-mode transcripts (byte-identical).
- RC_R3.md + rc_r3_numeric.py + rc_r3_numeric_output.txt + rc_r3_numeric_output_O.txt — RC's R3
  report, script, and both-mode transcripts (byte-identical).
- RD_R4_R5.md + rd_r4r5_verify.py + out_normal.json/out_opt.json (RD's original transcripts,
  byte-identical) + rd_r4r5_out_normal.txt/rd_r4r5_out_O.txt (lead's reproduction run, byte-identical,
  hash prefix 45f4d4718d91 — matching RD's reported 45f4d471…397d, reproduction-CONFIRMED)
  + rd_r4r5_exit_normal.txt/rd_r4r5_exit_O.txt (exit receipts, both 0).
- ENVIRONMENT_LOCK.json — python 3.12.12, mpmath 1.3.0, numpy 2.2.5, sympy 1.14.0, scipy 1.16.2
  (W2_sanity only); linux x86_64; deterministic; no network.
- MANIFEST.sha256 — whole-file sha256 + bytes for every file (excluding itself).

Review mode (standing record): ordinary hostile review after reading the candidate; not blind
reconstruction. Candidate identity: 02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md, body sha256
cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5 (matches the original distribution
MANIFEST; the reconciliation bundle's inputs copy is byte-identical).

Status firewall: nothing in this bundle changes any LS-CTL Boolean, eligibility predicate, theorem
status, RP status, AO48 operator record, or q0 package status. Any status change requires separate
operator adjudication.
