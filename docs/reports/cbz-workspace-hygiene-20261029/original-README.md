# Read-only workspace Git inventory

The federation census initially missed nested execution fixtures and an empty package-cache `.git` marker. This stdlib tool records every physical Git candidate, linked worktree, common store, ref/tag/stash, configured redacted remote and submodule. It preserves unusual path bytes and fails closed on uncovered paths, malformed markers, Git errors, deadlines or output limits. An exit0 means **complete observations, not clean Git or CBZ-0**.

This is a disjoint contribution to [the canonical consolidation](https://github.com/d6g8k5htny-coder/main/issues/684). It does not certify mathematical claims or replace G0–G4, remote repository inventory, regression, pin or status-truth gates. No dependency, mathematical source, scientific disposition or existing test is changed.

## Reproduce and register

Use Python3.11+ and Git on a POSIX host; actual parent verification used Python3.12.14 and Git2.52.0. No third-party imports or network access are needed. Output parents must already exist; output files/directories must be new.

```sh
python3 -S -B tools/workspace_git_inventory.py --root /workspace --output /private/existing-parent/new-inventory.json --command-timeout 10 --total-timeout 900
python3 -S -B -m unittest tests.test_workspace_git_inventory tests.test_workspace_git_inventory_mutations -v
python3 -S -B -O -m unittest tests.test_workspace_git_inventory tests.test_workspace_git_inventory_mutations -v
python3 -S -B tools/workspace_git_inventory_mutations.py --output /private/existing-parent/new-mutation-directory
```

Collector output is exclusive-create0600. The companion creates an exclusive0700 directory and0600 logs/results. It derives source/test paths from its own location and uses its running interpreter. Defaults:10seconds per Git command,1800seconds overall and32MiB combined output per command; failures kill/reap only owned children. The companion requires all11 core tests to pass with zero skips/errors before four known mutations. A surviving mutant or test error cannot count as detection. It retains source/mutant identities and metrics, then removes its owned temporary mutant copies.

Registry handoff for canonical GOVERNANCE/TOOLING/STATUS: register `tools/workspace_git_inventory.py` as a read-only Phase1/G10 assessment tool and `tools/workspace_git_inventory_mutations.py` as its self-test companion. Both are self-contained, maintenance owner none, serving criteria1/9/11 and G10/G11 assessment. The collector performs no mathematical certification and does not turn dirty states into a failing completeness check; gate consumers must inspect the recorded state and combine remote branch/PR evidence.

## Verification and observed cleanup

The11 collector tests cover nested/node_modules repositories, linked and bare stores, refs/tags/stash/submodules, shallow/unborn/detached state, LF/tab/non-UTF8/rename paths, metadata nonmutation, clean-filter execution prevention, credential redaction, broken markers, symlink boundaries, bounded hangs and output. Four companion tests exercise required/private/exclusive output, failed-baseline refusal, a surviving no-op mutant and test-error refusal. The independently run companion baseline passes11 and detects4/4 weakened collectors by assertion failures. It was also run from `/tmp` to verify directory independence. Core tests passed under both Python3.11.17 and3.12.14. Versioned final normal/optimized test logs and source hashes accompany this report. The companion's internal child core suite uses its explicit-S/-B flags; the separate optimized core suite checks optimized collector behavior.

Initial private census:74candidates,24stores,73worktree observations,17dirty worktrees and0stashes. Its sole error was a zero-byte invalid Git marker in derived package cache; it correctly returned incomplete/exit1. Later complete census found65worktrees after the scratch fixtures were retired. After53 retired Atlas roots were archived and removed, the next complete census recorded10stores/12worktrees/0errors/0stashes; only this author checkout was dirty. These are interval observations, not a current or global G10-pass claim.

Recovery was verified before local cleanup:112,040 scratch entries and138,102 retired-root entries matched independent filesystem/path censuses and native tar byte comparison. Fifty stale IPC sockets were explicitly recorded as non-file endpoints. A native-verified all-ref/reflog Git bundle restored all42 distinct retired HEAD commits in a fresh temporary mirror. Entire shared Git metadata, including original indexes/staging states, was separately byte-verified;23 old local engineering branches were then conditionally pruned at their archived SHAs. No mathematical source branch was pruned by this unit. Private archive locators, raw inventory paths and custody mappings remain private.

An existing Python3.11 installation was discovered to have lived in scratch. Its exact archived runtime was restored to a maintained location, owner-local link repaired, member bytes/modes verified, and11 normal/optimized core tests rerun. Initial missing-runtime and comparison failures are retained privately; they are not relabelled successful checks. DefaultPython3.12 remained available.

## Limits and authority

Observations are non-atomic. Internal directory symlinks are followed once; Git metadata aliases are excluded. External directory links and coverage errors are explicit incompleteness. Git ignore rules still apply to untracked status; discovery has no node_modules or cache exclusion. Configured hooks, clean/process filters, fsmonitor, automatic maintenance, optional locks and lazy network fetch are disabled for collection. No unreachable-object/reflog audit, remote fetch, preservation/deletion, proof closure, all-platform guarantee or global baseline pass follows from collector exit0.

Actual authors: OpenAI/Codex Kepler for the collector/migration/companion, parent for permanent companion tests and integration. Self-review and actual automated controls; organizational-independence credit0 and scientific effect NONE. Canonical baseline documents and serial remote integration remain with #684's existing writer.
