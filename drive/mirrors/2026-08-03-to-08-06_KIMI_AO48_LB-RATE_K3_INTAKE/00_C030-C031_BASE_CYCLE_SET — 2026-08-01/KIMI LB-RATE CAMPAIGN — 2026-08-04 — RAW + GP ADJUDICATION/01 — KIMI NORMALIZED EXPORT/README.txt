KIMI_EXPORT_2026-08-04_LB - byte-exact addendum bundle (LB-RATE theorem-grade campaign).
Same spec as the export rounds it extends.

ENCODING: UTF-8 no BOM; LF only; ordinary spaces (no U+00A0); one trailing LF; no smart
quotes/dashes. Body-hash convention preserved: the marker line excludes itself; hashes
recomputed on the normalized bytes; whole-file hashes additionally in MANIFEST.sha256.

SELF-TEST: every section of KIMI_EXPORT_CONCAT_LB.txt re-extracted and re-hashed against
MANIFEST.sha256 before delivery: 15/15 PASS (no section fails its round trip).

PER-FILE LEDGER (ID | role | verdict/tier):
KIMI-THM-023.md | LB-5 assembly | 1 - q(r, 6/5) >= c·r^3, c = AO_lb·c_L -> 0.9144 (proved floor;
  exceeds measured-grade 0.87·r^3, stays below direct measurement C* ~= 0.96)
KIMI-AUD-023.md | LB-4 residue | full enumeration; R2 misprint recorded ((9-6.25)->(9-2.25),
  consistent reading 2.111 ~= printed 2.1); WP components consistent 0.21248·r^3
KIMI-AUD-024.md | LB-6 corollary | two-sided c·r^3 <= 1-q <= C·r^3, conditional on
  GP-DER-118-v1.10 at HOLD / NOT PROMOTED; firewall: no Boolean change, no promotion,
  status inherited verbatim, re-derives on status change, never auto-upgrades
LB1_sup_over_zone_proof.md | LB-1 proof text | PROVED (localized sup-over-zone; >=22 orders
  margin; registered correction: zone-wide kill false, full gradient+value+type intensity
  superexponentially killed throughout the pass zone; WP rigidity-zone flag)
lb1_sup_over_zone_certificate.py + lb1_t_normal.txt + lb1_t_O.txt | LB-1 certificate |
  fail-closed; normal and -O transcripts byte-identical (cmp-verified); six-rung table
  E/tol 1.7e-23 .. 2.5e-506
LB2_mean_ridge_proof.md | LB-2 proof text | PROVED at frozen rungs with r-free reduction
lb2_cert.py + lb2_transcript.txt | LB-2 certificate | fail-closed; tamper-test aborts exit 1
LB3_DISCHARGE.md | LB-3 proof text | R0 CLOSED; gamma-LOC PARTIAL ((ii-c) named conditional)
lb3_certificate.py + lb3_transcript_normal.txt + lb3_transcript_O.txt | LB-3 certificate |
  fail-closed; normal and -O transcripts byte-identical

STANDING GP ONE-LINER (for the operator's arrival processing):
On arrival of further items: land carriers, REGISTER the C031 amendment (the R2 misprint
correction: (9-6.25) -> (9-2.25) in the R2 near-term line), and REGISTER the WP re-derivation
task (Lemma WP's rigidity-zone figure ~3e-15 contradicted by certified 1.3e-5 integral over
d <= 1.5, hot spot (-0.05, -0.575) mp-verified 1.33e-4/unit-area).

OUTSTANDING LANDING: the 2026-08-03 bundle landing from HANDOFF-059 remains outstanding
(the ten-payload handoff with GP-DATA-214 source, GP-DER-197, LCR-DER-057/061, CL DQ-012,
LS-DER-026/027/030 natives; byte settlements already completed and reported separately).
