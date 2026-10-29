# Read-only workspace Git inventory

This stdlib tool records physical Git candidates, linked worktrees, common stores, refs, tags, stashes, redacted remotes and submodules. An exit0 means **complete observations, not clean Git or CBZ-0**. Raw inventories retain exact filesystem identities and must remain private; remote credentials and diagnostic values are redacted.

This contribution belongs to [the canonical consolidation](https://github.com/d6g8k5htny-coder/main/issues/684). Canonical baseline documents and serial remote integration remain with that writer. No mathematical source, proof status, scientific disposition or dependency is changed. Local census observations do not substitute for federation remote inventory or the G0–G11 final verification vector.

## Admission corrections

The original source at `60cd4516cd1ad54d778b33b22df289dfe86ee0d8` passed its original15 tests and four mutations. An independent same-family engineering review then reproduced three real gaps; its merge-ready disposition was withdrawn before integration.

- Valid filter names containing `=` could execute configured programs. Overrides now transport complete config keys and values separately through `GIT_CONFIG_COUNT`, and verify effective suppression before status. Config discovery or transport failure refuses status. Regression fixtures first prove native Git executes both parent and recursive-submodule filter programs, then prove collection executes neither.
- Ordinary asset directories containing `HEAD` and `objects` hid nested repositories. Metadata pruning now requires exact successful `rev-parse --resolve-git-dir` validation of the candidate. Ancestor repository discovery cannot establish that identity.
- Whole-report credential redaction collapsed distinct filesystem paths. Redaction now applies to remote URL and diagnostic values; root, worktree, store, symlink, status and error-path identities remain exact, including unusual bytes.

The original four mutation purposes are retained, and three mutants restore these defects. A separate controlled experiment against the original companion also demonstrated that setup assertions could be miscredited as mutation kills. Its original green result is retained as historical evidence, not final certification. The final companion qualifies exact pristine class/function/code identities before loader execution and exact loaded/started/body-entry/body-exit populations. Only an actual escaping designated-body AssertionError can count; lifecycle, callback-only, deferred-body and subtest reports refuse. Actual parent optimization propagates to children. No scanner or kernel-gate claim follows from this engineering tool.

## Reproduce and register

Use Python3.11+ and Git2.31+ on a POSIX host. Actual collector verification uses Python3.11.17/Python3.12.14 and Git2.52.0. No third-party imports or network access are needed. Output parents must exist; files/directories must be new.

```sh
python3 -S -B tools/workspace_git_inventory.py --root /workspace --output /private/existing-parent/new-inventory.json --command-timeout 10 --total-timeout 900
python3 -S -B -m unittest discover -s tests -p 'test_workspace_git_inventory*.py' -v
python3 -S -B -O -m unittest discover -s tests -p 'test_workspace_git_inventory*.py' -v
python3 -S -B tools/workspace_git_inventory_mutations.py --output /private/existing-parent/new-mutation-directory
python3 -S -B -O tools/workspace_git_inventory_mutations.py --output /private/existing-parent/new-optimized-mutation-directory
```

Collector output is exclusive-create0600. The companion creates an exclusive0700 directory and0600 logs/results, derives source/test paths from its own location, and uses its running interpreter. Collector defaults are10seconds per Git command,1800seconds overall and32MiB combined output per command. Failures terminate/reap only owned children.

Registry handoff for canonical GOVERNANCE/TOOLING/STATUS: register `tools/workspace_git_inventory.py` as the read-only Phase1/G10 assessment tool and `tools/workspace_git_inventory_mutations.py` as its self-test companion. They are self-contained, maintenance owner none, serving criteria1/9/11 and G10/G11 assessment. Consumers must inspect recorded dirtiness, stashes, refs and coverage errors and combine remote branch/PR evidence; collector exit0 is not a cleanliness gate.


## Final corrected qualification

[Final source hashes](correction/final/SOURCE.json), [author result counts](correction/final/AUTHOR-SUMMARY.json) and [bounded implementation/replay handoff](correction/final/COMPANION-HANDOFF.md) bind the completed generation: **31 = 14 core tests + 17 companion controls**, passing normally and optimized on Python3.11.17 and3.12.14, zero skips/errors/failures. The normal and actual-optimized companions each detect **7/7 collector source mutants**. **Five separate companion-guard regressions** are caught by their designated blackbox assertions; the parent inspected all five raw traces, not just their count-based summary. Those five historical guard runs use normal Python; no optimized guard-mutation execution is asserted.

The parent independently replays the original seven review fixtures and all14 core tests in all four runtime modes. [Final frozen-source targeted runs](correction/final/parent-targeted/RESULT.json) additionally pass four loader-substitution controls, callback/subtest/setup refusal and a complete7/7 companion run with actual optimized children. The parent did not complete a new full31 aggregate run: one trial crossed a changing intermediate source generation and another was interrupted by an environment reconnect; both are retained privately without success credit. Author aggregate runs remain separately attributed. Hume's nonauthor core review is preserved separately; the final companion receives author/parent self-review and targeted execution, not a completed independent-review verdict.

Baseline completeness is an exact fourteen-method manifest, not a count alone. A loader cannot substitute duplicate methods, expose14 but execute one repeatedly, rebind a same-name class or alias every method to one body. The frozen synchronous protocol rejects coroutine/generator functions and deferred return values, every subtest report and assertions from setup/teardown/cleanup. It retains the exact escaping AssertionError object and verifies the designated method traceback. Arbitrary malicious Python source or external testing plugins are outside this finite engineering contract.

The five intentionally corrupted guard source files are [hash-indexed negative fixtures](correction/final/NEGATIVE-INPUTS.json), never admitted tools or imported by production checks.

For the optional historical five-guard replay, run `python3 -S -B docs/reports/cbz-workspace-hygiene-20261029/correction/final/guard-controls.py.txt CHECKOUT NEW_PRIVATE_OUTPUT`. Both arguments are explicit; the output parent must exist. Read each retained traceback and verify the named meta-test assertion, zero errors and the expected mutation. Its simple summary is not an independent lifecycle/certification authority; the current permanent31-test suite and seven-mutant companion are the reusable tool checks. The frozen recipe's original three-guard docstring predates its actual five-case list; original bytes are preserved.

The surrounding [legacy Atlas disposition](correction/legacy-atlas-DISPOSITION.md) is a read-only source inventory: #399 is partially superseded, and #501's acquisition delta remains unlanded. It grants neither integration nor proof coverage. Root retains disposition/integration ownership.

## Historical observations and recoverability

The following census counts came from the original collector and are historical, unqualified interval observations after the reproduced completeness gaps. They must not be used as final G10 evidence.

Initial private census:74candidates,24stores,73worktree observations,17dirty worktrees and0stashes. Its sole error was a zero-byte invalid Git marker in derived package cache; it correctly returned incomplete/exit1. Later complete census found65worktrees after the scratch fixtures were retired. After53 retired Atlas roots were archived and removed, the next complete census recorded10stores/12worktrees/0errors/0stashes; only this author checkout was dirty. These are interval observations, not a current or global G10-pass claim.

Recovery was verified before local cleanup:112,040 scratch entries and138,102 retired-root entries matched independent filesystem/path censuses and native tar byte comparison. Fifty stale IPC sockets were explicitly recorded as non-file endpoints. A native-verified all-ref/reflog Git bundle restored all42 distinct retired HEAD commits in a fresh temporary mirror. Entire shared Git metadata, including original indexes/staging states, was separately byte-verified;23 old local engineering branches were then conditionally pruned at their archived SHAs. No mathematical source branch was pruned by this unit. Private archive locators, raw inventory paths and custody mappings remain private.

An existing Python3.11 installation was discovered to have lived in scratch. Its exact archived runtime was restored to a maintained location, owner-local link repaired, member bytes/modes verified, and11 normal/optimized core tests rerun. Initial missing-runtime and comparison failures are retained privately; they are not relabelled successful checks. DefaultPython3.12 remained available.

## Limits and authority

Observations are non-atomic. Internal directory links are followed once; exact validated Git metadata is excluded. External directory links, malformed markers, coverage errors, command timeouts and output ceilings are explicit incompleteness. Git ignore rules apply to untracked status; physical discovery has no cache or node_modules exclusion. Collection disables configured hooks, clean/process filters, fsmonitor, automatic maintenance, optional locks and lazy network fetch. It performs no remote fetch, preservation/deletion, unreachable-object audit or proof certification. Git versions that ignore the required configuration transport fail closed before status.

Private archive locators, raw inventory paths and custody mappings remain private. The path-preserving report is not safe to publish merely because credential URL fields were redacted.

Actual authors: OpenAI/GPT-6/Codex Kepler for collector and companion corrections; parent for permanent companion controls, integration and verification. Hume independently reproduced the original defects but is from the same provider family. Organizational-independence credit0; scientific effect NONE. Existing source failures and historical evidence remain preserved.


Original `SOURCE.json`, logs and `mutations.json` describe the original admission. [Original evidence manifest](original-EVIDENCE.json) and [original README bytes](original-README.md) preserve that source snapshot; replay its README entry against `original-README.md`, or read all original paths at commit `60cd4516cd1ad54d778b33b22df289dfe86ee0d8`. The current evidence manifest binds the current report and correction receipts separately. Original [three-regression RED](correction/original-three-regressions-RED.log), [setup-only false admission](correction/original-lifecycle-RESULT.json), [parent core results](correction/CORE-RESULT.json), [original fixture replay](correction/REPRODUCTION-RESULT.json) and [nonauthor core correction review](correction/review-REVIEW.md) remain explicit.
