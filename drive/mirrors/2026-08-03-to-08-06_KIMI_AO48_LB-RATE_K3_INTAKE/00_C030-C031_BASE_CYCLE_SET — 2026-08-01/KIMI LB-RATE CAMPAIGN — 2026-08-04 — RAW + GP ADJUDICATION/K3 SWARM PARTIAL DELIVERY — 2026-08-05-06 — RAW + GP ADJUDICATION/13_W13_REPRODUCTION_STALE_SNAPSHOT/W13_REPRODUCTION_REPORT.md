# W13 REPRODUCTION REPORT — K3_SIDE24_LB clean-room readiness

Date: 2026-08-06. Agent: W13 (reproducibility and export). Mode: adversarial verification;
no artifact was patched, worked around, or modified.

## 0. Environment and method

- Host Python: 3.12.12 (/usr/local/bin/python3). mpmath 1.3.0, numpy 2.2.5, scipy 1.16.2.
- Every certificate was re-executed from a FRESH environment:
  `env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 python3 [-O] <script>`
  via `W13_reproduction/cleanrun2.sh`, from the certificate's own directory.
  Program stdout, stderr, exit code (`.rc`), and wall time (`.time`) were captured to
  `W13_reproduction/runs/`. Runner labels/receipts are separate files by design;
  only PROGRAM output was compared for byte-identity.
- No packages were installed. All declared imports resolved: no missing dependency.
  Note: `W2_sanity.py` imports scipy (used via scipy.stats.norm / scipy.integrate.quad);
  scipy 1.16.2 is present in this environment, matching the version recorded in
  W2_DERIVATION.md ("numpy 2.2.5, scipy 1.16.2"). REPRODUCTION.md's "no other required
  packages ... numpy only where declared" understates this: W2 requires scipy.
  Finding F5 (documentation), not a failure here.

## 1. Clean-room re-execution results

Exit codes and wall times (seconds) from `runs/*.out.rc` / `runs/*.out.time`.
"pair" = `python3` vs `python3 -O` program outputs compared with `cmp`.
"prior" = comparison against the workstream's recorded transcript.

| certificate | rc (normal/-O) | time s (normal/-O) | pair byte-identical | matches prior transcript |
|---|---|---|---|---|
| W2_symbolic/W2_sanity.py | 0 / 0 | 31.85 / 33.59 | YES | YES (see note a) |
| W6_uniform_r/w6_wp2.py | 0 / 0 | 2186.13 / 1870.61 | YES | PARTIAL (see F1) |
| W6_uniform_r/w6_cert.py | 0 / 0 | 42.28 / 49.99 | YES | YES (transcripts_cert_main.txt, cmp-clean) |
| W7_gammaloc/verify_k3_w7_gammaloc_v1.py | 0 / 0 | 2.02 / 6.59 | YES | YES (verify_t_normal.txt / verify_t_O.txt) |
| W7_gammaloc/mutation_driver.py | 0 / 0 | 10.68 / 39.43 | YES | YES (mutation_t_normal.txt; -O identical to normal) |
| W9_scope_027a/audit_scope_027a_v1.py | 0 / 0 | 2426.40 / 2421.19 | YES | YES (audit_scope_027a_v1.out.txt / .O.out.txt) |
| W9_scope_027c/verify_ridge_scope_continuation_v1.py | 0 / 0 | 516.89 / 527.24 | YES | YES (t_normal.txt / t_O.txt) |
| W10_dependency/w10_sanity.py | 0 / 0 | 0.79 / 2.05 | YES | YES (t_normal.txt; -O identical to normal) |

(W2 exit codes were observed directly from the runner shell; the v1 runner used for W2
did not yet write `.rc` files. All other exit codes are recorded in `runs/*.out.rc`.)

Program-output sha256 (normal == -O in every case):

- W2_sanity:        14b12b2243e4dd5f927b0320fa87be710f8a509a7563d93c9ba76e044ebedb2a
- w6_wp2:           ce4245af43521eacc4030cd7feacd444303adb75d2e914cd11f05874c0bdff5d
- w6_cert:          000bfb456d51614aa3f29b00f23f5e7be64ca3611fa37ef16be584ce9c1608a9
- w7 verify:        19c3c0099e0c1ca06fd1908827baeb40534e71759dbfd28b4efa244cc7864310
- w7 mutation_drv:  2a04b2872365327d3f2b4f1384dc70ac9968d0b559904d7759db54a9e2783e03
- w9a audit:        d6cfc8fb7b351ee21e24b9e941e8e5747d51f2487ee6e8bb71acab2b9d244f46
- w9c verify:       3dd113e32922321e60cec4ae36e08d13a2cc85b00cf617b7035f647db66ef99a
- w10_sanity:       20bff6dcfe2137d7b8eeef7819e44e21142f61fbd1b788dc1dd5f5bfee0ca40b

Independent confirmations:
- The fresh W9a output hash d6cfc8fb... equals the hash recorded for
  audit_scope_027a_v1.out.txt / .O.out.txt inside W9_scope_027a/MANIFEST.sha256.
- The fresh W10 output hash 20bff6dc... equals the t_normal.txt/t_O.txt hashes in
  W10_dependency/HASHES.txt.
- The fresh W2 output is byte-identical to run1.log after removing the first two lines
  of run1.log, which are a stderr RuntimeWarning (overflow in exp) that the original
  runner merged into the log; program stdout itself is identical (note a).
- The fresh w6_cert output is byte-identical to transcripts_cert_main.txt.
- Check counts observed: W9a audit prints 94 `PASS` lines + terminal "ALL CHECKS PASS"
  (matches "94 checks"); W9c prints 73 `[ok]` lines + terminal completion banner
  (matches "73 checks"); W2 prints 113 `PASS` lines + "ALL CHECKS PASSED"
  (W2_FREEZE.json and W2_DERIVATION.md say "114" — see F4).

Runtimes are wall-clock under parallel execution (wp2/w9a ran concurrently, load ~3);
they are not part of the certificate semantics. The earlier solo w6_wp2 run took 372 s.

## 2. Hash verification

### 2.1 INPUT_MANIFEST.sha256 (root intake manifest)
- `sha256sum -c` from /mnt/agents: 578 entries OK, 0 mismatches, 20 entries FAILED
  open-or-read. Log: `hash_input_manifest.txt`.
- All 20 failures share one defect: the manifest line stores `research/` prefixed to an
  ABSOLUTE path (e.g. `research//mnt/agents/output/SIDE24_gap_fill/...`). Every one of
  the 20 target files exists at its absolute path, and every one matches its recorded
  sha256 when verified manually. Log: `hash_input_manifest_recovered.txt` (20/20 HASH-OK).
- Net: 598/598 payloads verified, 0 hash mismatches; 20 malformed path lines remain a
  defect in the manifest file itself (Finding F2).

### 2.2 Workstream hash tables
- W4_independent/MANIFEST.sha256: 39/39 OK (`sha256sum -c`, rc=0). Log hash_W4_independent.txt.
- W6_uniform_r/MANIFEST.sha256: 17/17 OK (`sha256sum -c`, rc=0). Log hash_W6_uniform_r.txt.
- W9_scope_027a/MANIFEST.sha256: annotated (not machine-checkable as-is); all 4 deliverable
  hashes verified manually OK (audit_scope_027a_v1.py, .out.txt, .O.out.txt, _report.md);
  all 6 receipt hashes verified against the actual input files OK (AO48-WO-063 work order,
  C022 Observed Update.json, c022 fd.json, KIMI-DER-027a_farfield_envelope.md,
  verify_farfield_envelope_v1.py, verify_farfield_envelope_v1.out.txt). The referenced
  DER-027a transcript pair (.out.txt/.O.out.txt) is itself byte-identical as claimed.
- W10_dependency/HASHES.txt: 4/4 OK (`sha256sum -c`, rc=0).
- W2_symbolic/W2_FREEZE.json: body_sha256 of W2_DERIVATION.md (body below the freeze
  comment line) verified OK; W2_sanity.py and run1.log hashes verified OK.
- W3_numerics: no MANIFEST.sha256 and no report have landed at audit time; nothing to
  verify. Status: PENDING (per tasking, "when they land").
- W8_lambda: certificate `verify_lambda_grid_v2.py` (sha256 579ef6cd... per
  mutation_receipts.txt) is NOT present in the tree; only report, receipts and one
  transcript landed. Out of tasked re-execution scope; noted for the lead.

### 2.3 Full-tree MANIFEST.sha256 (built by W13)
- `MANIFEST.sha256` at the K3_SIDE24_LB root: every deliverable file, whole-file
  sha256 + byte count, excluding only itself and `__pycache__` contents. Built by
  `build_manifest.py`; companion machine-checkable file
  `W13_reproduction/MANIFEST.check.sha256` (strict `sha256sum -c` format).
  Caveat: the tree was LIVE during this audit (W3_numerics/strip.out and strip.txt
  landed 04:30-04:32 CST, strip.txt still growing). The manifest is a point-in-time
  snapshot taken as the last W13 action; its entry count, total bytes, and sha256
  are transmitted to the lead with this report. Any file landing after the snapshot
  requires a manifest rebuild (rerun build_manifest.py).

## 3. Mutation review (rerun, fail-closed)

W7_gammaloc/mutation_driver.py (rerun in both modes, rc=0, byte-identical to
mutation_t_normal.txt): M1 pristine ACCEPT exit 0; M2 total-above-gate, M3 gate-shrunk,
M4 station-variance corrupt, M5 station dropped, M6 sum inconsistency (proj_c1 + 1e-12),
M7 Rice constant corrupt, M8 closes-flag flip, M9 truncated JSON — all REJECTED,
exit 1. 1 pristine accept + 8 mutated rejects + truncation, as tasked.

W9_scope_027c mutations (rerun from mutations/, clean environment):
- m1_quote_gate.py: exit 1, `FAIL: m(P*) = 1.93119045755789484 (DER-027c)`
  (mutated quoted constant 1.93119035... -> 1.93119045...).
- m2_chain_gate.py: exit 1, `FAIL: chain gate P: shared-rung root inside new tube`
  (gate tightened from rho2 to 1e-30).
- m3_ladder_gate.py: exit 1, `FAIL: ladder top rung r = 0.05 certifies`
  (lmin floor relaxed 0.01 -> 3.0).
- m4_scope_gate.py: exit 1, `FAIL: P*: 0 < v < 1 (noise nondegenerate; objects differ)`
  (variance lower bound relaxed 0.01 -> 0.5).
All four fail closed with informative FAIL lines. Receipts in `runs/w9c_m*.out`.

W9_scope_027a A8 mutation tests are internal to the certificate (A8-M1 pin-value
mutation invariance, A8-M3 kernel sign-flip detection, A8-M4 envelope-coefficient
mutant rejection) and passed inside the reproduced runs above.

Mapping to the campaign's minimum mutation set: input-hash corruption is covered by
W7-M9 (truncation) plus the S0 fail-closed hash bindings exercised in every default-mode
W7 run (11 bindings verified in the reproduced transcript); determinant/type sign change
by W9a A8-M3; pin/normalization perturbation by W7-M4/M6 and W9a A8-M1; interval/gate
omission by W9c m2/m3/m4. The "missing-sqrt restoration" mutation is not implemented
by any landed certificate (Finding F6, scope note, not a failure of what exists).

## 4. Byte discipline (report only; nothing modified)

Audit: UTF-8 decodable, no BOM, LF-only, no U+00A0, exactly one trailing LF, no smart
quotes (U+2018/2019/201C/201D). 143 text artifacts in the deliverable tree
(240 including W13_reproduction's own files, which are clean). Script:
`byte_audit.py`; log: `byte_discipline.txt`. 12 violations:

1. CANONICAL_STATE.json — NO-TRAILING-LF
2. W2_symbolic/W2_FREEZE.json — NO-TRAILING-LF
3. W4_independent/probes_laws.txt — MULTI-TRAILING-LF (ends with 2+ LFs)
4-12. W7_gammaloc/mutation_workspace/M1...M9*.json — NO-TRAILING-LF
      (machine-generated by json.dump in mutation_driver.py; regenerated on every run)

No BOM, no CR, no U+00A0, no smart quotes anywhere in the tree.
Zero-length files noted (not violations): W3_numerics/scan2_land.txt, scan_out.txt.

## 5. Findings

- F1 (minor, transcript hygiene): W6_uniform_r/transcripts_wp2_grid1.txt and
  transcripts_wp2_grid2.txt are partial-grid outputs and do NOT match the current
  w6_wp2.py output (current script certifies 11 rungs 0.05..0.00625; the stored
  transcripts cover only rungs <= 0.015 / <= 0.02 and differ in the summary line).
  All overlapping rung lines match byte-for-byte, and the fresh full-grid output
  matches the rung table printed in W6_REPORT.md (C_I = {3.202,...,1.202}e-3).
  No final full-grid transcript of w6_wp2.py existed in the tree before this audit;
  the clean-room transcripts are deposited at runs/w6wp2_normal.out / w6wp2_O.out.
- F2 (manifest defect): 20 malformed `research//mnt/...` path lines in
  INPUT_MANIFEST.sha256; payloads all verified manually (section 2.1).
- F3 (byte discipline): the 12 violations in section 4 (3 hand-authored files plus
  9 regenerated mutation-workspace JSONs).
- F4 (documentation): W2_FREEZE.json and W2_DERIVATION.md claim "114 PASS"; the actual
  program prints 113 `PASS` lines plus the terminal "ALL CHECKS PASSED" line, in both
  the original run1.log and this reproduction. Reproducibility itself is exact; only
  the count label disagrees (presumably counting the terminal line).
- F5 (documentation): REPRODUCTION.md says numpy is the only extra dependency "where
  declared"; W2_sanity.py also requires scipy (present here, 1.16.2, matching the
  W2 run record).
- F6 (scope): no landed certificate implements the "missing-sqrt restoration"
  mutation; W3 hash tables/report have not landed; W8's certificate script is absent
  from the tree.

## 6. Overall verdict

CLEAN-ROOM READINESS: PASS.

Every tasked certificate re-executes from a fresh environment, in both `python3` and
`python3 -O` modes, with exit 0 and byte-identical program output; every landed hash
table verifies with zero payload mismatches; every implemented mutation test fails
closed on rerun. No failure to reproduce was found and nothing was patched.

Required fixes before archival (none blocking reproduction):
1. INPUT_MANIFEST.sha256: rewrite the 20 `research//mnt/...` lines to correct
   relative/absolute paths (F2).
2. Byte discipline: add one trailing LF to CANONICAL_STATE.json and
   W2_symbolic/W2_FREEZE.json; strip the extra trailing LF in
   W4_independent/probes_laws.txt; optionally make mutation_driver.py write a
   trailing newline in its workspace JSONs (F3).
3. Reconcile the W2 "114 PASS" label with the 113 printed PASS lines (F4).
4. Note the scipy dependency of W2_sanity.py in REPRODUCTION.md (F5).
5. Deposit a final full-grid w6_wp2 transcript in W6_uniform_r and add it to
   W6_uniform_r/MANIFEST.sha256 (F1); land W3 hash tables; land or delist the W8
   certificate script (F6).
