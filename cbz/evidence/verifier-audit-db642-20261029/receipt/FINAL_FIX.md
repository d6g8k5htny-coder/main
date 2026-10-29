# Final whole-branch documentation fix

Author: OpenAI / Codex `/root/receipt_repair`, 2026-10-29. Engineering documentation correction only.

Commit: `afbf4c9ad8a6fb7a613d22c31d52aab872f304b3`.
Parent: `5a3f4cddbb7b285e62312a8e5265aaa010d11fb5`.

Addressed both P2 findings in `FINAL_REVIEW.md`: removed the unsupported source-only flag because source checking is already the default; replaced the incomplete Atlas command with the existing current-checkout fixture test and the qualified historical CLI carrying both `--root` and `--snapshot`. The registration addendum now links the existing `docs/site/README.md` historical section, enumerates its exact original manifest/source/graph/review/archive/snapshot/comparison-file prerequisites, and states that current-root refusal remains intentional. The historical test supplies archive/comparison data directly from repository evidence; the separately prepared CLI root must contain those files.

Only these three paths changed:

- `cbz/evidence/verifier-audit-db642-20261029/DECISION_ADDENDUM.md`
- `cbz/evidence/verifier-audit-db642-20261029/SHA256SUMS`
- `cbz/evidence/verifier-audit-db642-20261029/TOOL_REGISTRATION_ADDENDUM.md`

Appended a factual fix entry to the existing decision addendum and refreshed exactly its checksum and the tool-registration checksum. Every other checksum is unchanged. Production source, manifest, tests and historical evidence under `docs/site/evidence` were compared byte-for-byte with the prior HEAD and remain unchanged; the diff contains only these documentation/checksum paths. The historical-guide relative link resolves. `git diff --check` passed before commit. Post-commit membership and status were inspected.

## Verification actually executed

```sh
python3 -B -S tools/formal_gate_check.py --help
```

Exit 0. Parser help confirmed the supported arguments.

```sh
python3 -B -S tools/atlas_formal_evidence_build.py --help
```

Exit 0. Parser help confirmed the supported arguments.

```sh
python3 -B -S tools/atlas_formal_evidence_build.py --root /tmp/tmp8tv6y8tc --snapshot /workspace/scratch/db6428847655/cbz/worktrees/main-cbz/tests/fixtures/atlas_formal_projection.json --check
```

Exit 0. The temporary historical root was prepared with the existing `FormalConnectionAssembly.setUp` source fixture, then the unchanged original archive and published comparison module were copied into their required paths. Only the current CLI/validator executed; historical control text remained literal data. The temporary root was removed after this check. Output:

```json
{"source_connection": "math.side24-coefficient", "declaration": "UniversalLaw.Side24.pairing_sum_six", "receipt_sha256": "1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0", "submitted_alignment": "REJECTED", "scientific_effect": "NONE"}
```

`sha256sum -c SHA256SUMS` in the evidence directory: exit 0, all 29 entries OK.

Exact argv, stdout and stderr are retained in `final-doc-command-checks.json` alongside this report. Source-only PASS at manifest SHA-256 `49360aa6588167f2e4cd49e90ba0e8a6dfce6c47a11f1f09ce64def51e0e8590` is the controller's existing observation and was not rerun for this documentation-only change. No full or focused test suite was rerun, no Lean/kernel or hosted execution occurred, and no remote action or subagent was used. This correction does not claim that the current checkout qualifies as the historical projection or that historical alignment changed.
