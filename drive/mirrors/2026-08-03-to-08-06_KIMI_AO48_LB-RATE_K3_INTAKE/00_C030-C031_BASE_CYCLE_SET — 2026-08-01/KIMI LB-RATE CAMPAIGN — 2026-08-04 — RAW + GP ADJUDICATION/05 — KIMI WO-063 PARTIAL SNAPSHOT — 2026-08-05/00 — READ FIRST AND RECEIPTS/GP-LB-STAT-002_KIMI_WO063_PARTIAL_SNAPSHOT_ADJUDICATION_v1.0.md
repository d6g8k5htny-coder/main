# GP-LB-STAT-002 — KIMI WO-063 Partial-Snapshot Intake and Adjudication v1.0

Date: 2026-08-05

Status: LANDED AS EVIDENCE — NONCONTROLLING

## Intake identity

- Received archive: `OKComputer_Project_Gap_Closure(5).zip`
- Archive size: 7,249,515 bytes
- Archive SHA-256: `25e4e01cde764f6fa26c3ce317fef8b23763fded325062efe98f5e372cb292f6`
- Safe extraction: PASS (`unzip -t` clean; no absolute paths, `..` traversal, or symbolic links)
- Archive members: 347
- New relative paths versus the previously landed Kimi corpus: 50
- Individual-upload policy: the opaque ZIP is not a Drive carrier. New meaningful files are landed separately. Seven compiled `.pyc` cache files and the zero-byte `mini.log` are intentionally excluded. Previously landed byte-identical corpus members are not duplicated.

## Controlling disposition

No theorem, Boolean, gate, promotion predicate, AO48-OPR-045 status, or frozen C030/C031 body changes as a result of this snapshot.

- The zone-wide *bare value-kill mechanism* used in the old C030 rigidity-zone explanation is refuted as stated: the certified mean has regions above `b`, including the previously identified ridge maxima.
- The quantitative WP upper bound remains OPEN. The new numerical true-intensity computation is useful derived-on-grid evidence of a material discrepancy, but it is not a rigorous upper or lower enclosure.
- `KIMI-THM-023 v1.1` remains an incomplete, noncontrolling HOLD draft and is not hash-frozen.
- The displayed `0.9144036` value remains a candidate measured-grade limiting anchor, not a proved theorem constant.
- Existing q0 and SIDE24 controlling states remain unchanged.

## Load-bearing audit defect

`verify_wp_witness_v1.py` and the copied WP code in `verify_thm023_v1_1_v1.py` implement

`p_grad * sqrt(E[det(H)^2 | grad=0]) * min(Cantelli, P_window)`.

The Cauchy–Schwarz step instead requires

`p_grad * sqrt(E[det(H)^2 | grad=0]) * sqrt(min(Cantelli, P_window))`.

The missing square root invalidates the reported quantity as a certified upper bound. Consequently, the claimed `I_cs`, the rung-wise “ub” table, the proposed `E_WP = 5.5e-3 r^1.6` envelope, and the downstream theorem-constant assembly do not control the stated events.

The reported numerical integral near `4.7602e-6` is not promoted to a theorem-grade inequality because the delivery supplies no rigorous quadrature-tail enclosure, spatial-integration enclosure, or certified lower bound. It records strong numerical inconsistency with the old printed `3.328125e-6` budget at `r = 0.025`, and triggers re-derivation rather than closure.

## Artifact dispositions

| Artifact | Drive-facing status |
|---|---|
| KIMI-DATA-025 | Inventory; its preliminary orphan ruling is superseded by KIMI-DATA-028 |
| KIMI-DER-025 | AMEND REQUIRED; numerical WP evidence only |
| KIMI-DER-026 | CONDITIONAL; `P-NMZ-γ` remains open |
| KIMI-DER-027a | PARTIAL; derived/F64-pad variance-channel evidence at its stated rungs |
| KIMI-DER-027b | INCOMPLETE DRAFT; pending certificate/verdict |
| KIMI-DER-027c | Verified at `r = 0.025` for the deterministic conditional mean only; not r-uniform |
| KIMI-DATA-028 | Provenance settlement; no status effect |
| KIMI-AUD-023 v1.1 | AMEND REQUIRED; noncontrolling |
| KIMI-THM-023 v1.1 | INCOMPLETE/HOLD; not hash-frozen |

## Additional transport findings

- The orphaned pre-update KIMI-THM-023 body cited as `40596829…` is ruled UNRECOVERABLE by KIMI-DATA-028. This supersedes KIMI-DATA-025's preliminary recoverability statement.
- KIMI-DATA-028 says the exact C022 body and LS-DER-030 native carrier will ship in another export. Neither is present in this delivery; they remain CLAIMED-HELD / NOT DELIVERED here.
- The delivered WP transcripts are not byte-identical: normal is 7,953 bytes / `371f308f3ea0f3e5091a88340f7cfe350ab987d928ebbfa80ab7dc584f297515`; `-O` is 7,948 bytes / `6f1a226fe1ecefeba95aa955f4bbacfc582ebee789bc454ca5248c1c674dcebc`. Their program output differs only in the appended harness label, but the delivered bytes must remain separately identified.
- `KIMI-THM-023 v1.1` contains literal pending placeholders and `PLACEHOLDER-BODY-HASH`; its verifier's `RUNGS` list omits `0.005` although the code later reads `tbl['0.005']`. The delivered normal transcript stops before PASS and no `-O` theorem transcript is present.

## Open work

1. Repair the WP Cauchy–Schwarz formula and provide a rigorous spatial/quadrature enclosure or another theorem-grade WP argument.
2. Discharge `P-NMZ-γ` or retain γ-LOC(ii-c) as conditional.
3. Complete KIMI-DER-027b, produce both normal and `-O` transcripts, and issue a hash-frozen verdict.
4. Rebuild KIMI-THM-023 v1.1 only after its dependencies and verifier are complete; re-audit the correction order (`O(r^1.6)` is not `O(r^3)`).
5. Deliver the promised exact C022 and LS-DER-030 carriers separately.

## Readability and provenance policy

Every meaningful new source, report, manifest, and transcript is stored as its own Drive file. Primary reports receive native Google Docs mirrors. Scripts and transcripts remain raw byte carriers. The original ZIP, nested ZIPs, compiled caches, and empty log are not uploaded.
