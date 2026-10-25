# Universal Law v21 supplement — independent verification

**Verdict: final release passes the audited custody, runner, and offline replay checks.** One output-path defect in the initial candidate was found, reproduced, repaired, and independently regression-tested. No unresolved product blocker remains within this audit's scope.

Final archive: `Universal_Law_v21_Separated_Collision_Supplement.zip`  
SHA-256: `3f598aa86889eb58c1bf4d7f17bcfbaae01a194f2fca5db21f01a946c1eb80c5`  
Size: **7,288,931 bytes**.

## Final release evidence

- Independent fresh stdlib extraction into a previously unused external directory. All **1,703** ZIP members have safe, unique casefolded paths and regular-file types; all CRCs pass. The manifest completely covers **1,702** payload files and every SHA-256 matches. Uncompressed size: 29,544,817 bytes.
- `python -I -S -B verify.py` passed all **five** fixed exact scripts. A second normal invocation also passed, created a distinct `local/run-*` directory, and left every first-run artifact hash unchanged. Each run directory has mode **0700** and **16** artifact files.
- `python -I -S -B test_integrity.py` passed **six tests**, including the full product regression with preexisting symlink and hard-link outputs. The regression verifies that external sentinels stay unchanged.
- Exact replay includes **3,576** rational Hermite controls, **seven** main classification identities, **six** independent classification identities, and both antipodal scripts. Pinned pure-Python mathematics validation covers **1,651** files and licenses.
- Execution used bwrap with a private network namespace containing only loopback, a private PID namespace/procfs, an empty `/workspace`, hidden host site packages, isolated Python startup, and disk-backed temporary storage. Child limits remain 120 CPU seconds, 150 wall seconds, and 1 GiB address space.
- `release.py` run from the fresh extraction produced a **byte-identical** ZIP. After all checks, the original final ZIP, every extracted payload, manifest, and development-source payload still match the final release. Manifest SHA-256: `9c986b11f9ad997021f953462506f5674e00d5bde8d51458eb8c39ddd53c5ee5`.
- Sealed v19 and v20 ZIP hashes were independently rechecked: `0e685444eec4f72d450012462556c1ef8a993c0daeb8d20920e8b3c55e23021a` and `71b6b21679ab970c2d876268ab29322373e0444af4ece6d248d5f5cc57e95d89` respectively.

## Initial finding and repair

Initial candidate SHA-256 `101edcb12f8080e3274f6dfaa95897f065c353caa0aadedc1c262e97d860b68b` passed its ordinary checks but followed preexisting `local/` output symlinks. An unmodified full-product clone overwrote an external disposable sentinel and still reported success. The initial failure and logs are retained in `Initial_Candidate_Report.md`, `symlink-real-product.*`, and `initial-final-integrity.json`.

The final verifier places all child artifacts, logs, and reports in a fresh private run directory; parent log/report creation is exclusive and refuses symlinks. The six-test independent run confirms the repair. The release delta is limited to the verifier, tests, README, and three repair-evidence files; mathematical notes and scripts did not change. The README explicitly states the trusted-host-account boundary.

## Source and mathematical boundaries

All three proof notes match the reviewed hashes recorded in `Verification_Report.json`; four duplicate review-copy pairs match. The Gaussian reference is exactly SHA-256 `946333fcb7985c4989d4f3030022fe35d06744d108860035633f39c98d4a6781`. The retained author-course HTML is 178,093 bytes, SHA-256 `650a12928f8fd65579f725d73db07386445370375fb0287a9ae07b3722d29ad4`, and contains Theorem 4.8's Tarski–Seidenberg projection statement. The proof body explicitly identifies that import; source inspection is not automatic admission.

This audit supplies no new mathematical approval or kernel certification. Full D5, true collisions, vanishing-separation uniformity, the Rice/normalizer interface, remaining scales, and formal admission remain open. Complete theorem certificates remain **zero**.

The portable audit-evidence ZIP includes reports, independent harnesses, raw logs, both normal final-run results, reviewed runner code, and preserved initial-failure evidence. It omits extracted vendor/source trees, release ZIP backups, and scratch storage. One corrected harness CLI-order error and a superseded exploratory fixture are explicitly documented rather than counted as product failures.
