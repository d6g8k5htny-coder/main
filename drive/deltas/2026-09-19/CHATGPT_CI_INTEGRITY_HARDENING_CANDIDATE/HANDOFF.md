# CI integrity hardening — candidate handoff

**Status: locally tested; not pushed, not deployed, not a mathematical review.**
Task: CG-CI-20260919-6c6f47b. Prepared by ChatGPT / OpenAI under Dylan Roy's direct authorization to improve GitHub without interfering with Claude.

## Exact base and noninterference

Repository: `d6g8k5htny-coder/main`.
Claude's source branch: `claude/drive-audit-github-migration-rrglpp`.
Pinned base: `6c6f47b7884651b2330ab1380155f667caad0178`.
Proposed integration branch: `chatgpt/ci-integrity-hardening-20260919`.

The GitHub branch-creation attempt returned HTTP 403, "Resource not accessible by integration". No GitHub code, branch, pull request, settings, permission or Claude work was changed. No different writing connection was found. The container could not resolve github.com for a clone. Development therefore used **three source-file fixtures whose Git blob hashes exactly matched the connector-returned hashes**, not a complete checkout. The local branch holds only that partial fixture history and must not be pushed as though it were the research repository.

## Five-file candidate

- `.github/workflows/ci.yml`: make the nine existing conditional checkers mandatory; add the strict manifest guard; make token contents access explicitly read-only, disable checkout credential persistence, add a 30-minute limit, use Bash's strict workflow shell and fetch full history for history-based checks.
- `.github/workflows/research.yml`: remove the lane-check failure-masking expression; fail receipt upload when no files are found; fetch full history; include tracked, staged, untracked and ignored additions in the final governed-path check. This observes final state, not every intermediate operation.
- `tools/manifest_integrity_check.py`: add a standard-library-only, read-only guard. Require nonempty verification coverage, strict JSON object/duplicate-key handling, full SHA-256 and integer size fields, exact local paths and regular non-symlink files; report exclusions separately. Reject malformed checksum lines and wrong-path basename substitutions. Repeated `--require-manifest` arguments can pin expected manifests.
- `tests/test_manifest_integrity_hardening.py` and `tests/test_workflow_integrity_hardening.py`: CLI, workflow-command and temporary-Git-fixture controls.

The existing `tools/verify_manifests.py` is **unchanged**; this is an additional stricter check, not a rewrite of old records. No research code, source export, frozen artifact, claim graph, governing instruction or accepted scientific status is altered.

## Verification actually performed

**146 scoped tests passed, zero failures/errors/skips**, in eight complete nonoverlapping batches. Workflow tests execute retrieved workflow commands against stand-in pass/fail/missing checkers; they do not run the repository's real checker suite. Temporary Git repositories exercise the final-state guard.

**10/10 deliberate regressions were detected** by the new tests. Separate before/after probes reproduced nine fail-open cases plus a valid control against the exact old verifier/workflow source. Two real mounted handoff payloads were hash-matched to the inspected upstream manifest entries and passed the stricter checker in a generated local fixture. The candidate patch passed `git apply --check` against the exact three-file source fixture; applied files matched the tested candidate bytes, and the old verifier stayed unchanged.

Actual runtime: **Python 3.13.5**, pytest 9.0.2. Python 3.11 syntax parsing passed, but Python 3.11 runtime execution, the full repository test suite, the complete manifest corpus, remote CI and independent review were **not performed**. Two initial whole-suite attempts hit execution-tool time limits; only the retained completed batch reports support the 146 count.

This is not a deployment, access-control certification, completed Drive–GitHub bridge, or proof of mathematical correctness. Explicit exclusions are declarations, not newly authorized omissions. The strict check is not a proof that all research files are covered. No fixed complete required-manifest set has been installed. Untested historical manifest variants may need explicit compatibility decisions; do not silently rewrite frozen evidence to make CI green.

## Safe integration by a write-capable worker

Use a real, clean checkout, not this package's partial fixture tree. Do not reset, force-push, merge or edit Claude's branch. The following commands make a separate branch and apply only the candidate patch; they do not push or merge:

```bash
git fetch origin claude/drive-audit-github-migration-rrglpp
git switch --create chatgpt/ci-integrity-hardening-20260919 6c6f47b7884651b2330ab1380155f667caad0178
git apply --check /path/to/ci_integrity_hardening.patch
git apply /path/to/ci_integrity_hardening.patch
python -m pytest -q tests/test_manifest_integrity_hardening.py tests/test_workflow_integrity_hardening.py
python tools/manifest_integrity_check.py
python -m pytest -q
```

Require a clean worktree before these steps and inspect the five-file diff afterwards. Run **every check in the repository's actual CI workflow** in Python 3.11, not just the new tests. In particular, full history may expose old freeze violations hidden by shallow checkout. Stop and report a compatibility failure; do not suppress it or change a frozen manifest to pass. An existing target branch name or changed base-file content requires reconciliation, not force replacement.

After the real suite passes, commit only these five files on the separate branch. A draft PR targeting Claude's working branch can present the small delta for review; creating it is not authority to merge it. If Claude advances the same files, preserve both approaches and reconcile on the separate branch. Keep the recorded scientific states untouched.

## Evidence and sources

`VERIFICATION.json` pins candidate hashes, test scope and limitations. `SOURCE_IDENTITIES.json` pins the three upstream blobs. The evidence bundle includes full completed test reports, negative-control reports, scripts, exact source fixtures, candidate source files and the patch, with no Git history or credentials.

Project sources read: AGENTS.md, CLAUDE.md, both workflow files and the existing manifest verifier at the pinned commit; the DG-EXEC manifest's payload records. GitHub's secure-use guidance informed least-privilege token and credential handling: https://docs.github.com/en/actions/reference/security/secure-use . Existing unpinned action tags and dependency locks were not changed in this candidate.
