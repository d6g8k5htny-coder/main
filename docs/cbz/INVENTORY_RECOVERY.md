# Retained tracked-source inventory recovery

This recovery restores the existing inventory tool and tests from closed,
unmerged [Main PR695](https://github.com/d6g8k5htny-coder/main/pull/695), head
`82d9e480a51b946b3e74525ccf32139129d37d28`. It integrates a bounded source
integrity prerequisite into Main; it does not admit PR695's proof catalogue,
classifications, isolation assertions or canonical status documents.

| Recovered path | Bytes | Exact Git blob |
|---|---:|---|
| `tools/cbz_inventory.py` | 41,329 | `58f5cafbe151f7304ee9ea35b6fc46859d35630f` |
| `tests/test_cbz_inventory.py` | 48,756 | `ac8e5edc832ae9ded7ecad134264d3bd1f1871c5` |

At restoration commit `1e58f4d9fef8ea9016fc8e46e67a5f7dd33883dd`, both files
were unchanged from that source. The current filter-race repair below changes
the tool and tests; the table records original provenance, not current blob
identities. The recovery parent was Main
`5ebe5771d44160f56d4f2fd1e11726858d17dc90`. Missing-command baseline and
subsequent results below refer to the isolated recovery worktree, not a later
commit or integration tree.

## Run the existing controls

```sh
python3 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
```

The retained suite has 75 test methods and nine intended source mutants:
two tracked-inventory checks, two RF312 effective-source-map checks and five
replay-binding checks. Three outer mutation-controller methods skip themselves
inside their mutation subprocesses to prevent recursion; they must run at
the ordinary suite's top level. The RF312 and replay controllers require a
designated failure and reject an `ERROR:` result. The older two-mutant
inventory controller requires a nonzero exit and the designated `FAIL:` header;
it does not establish that no other error occurred. These are the preserved
classifier boundaries, not a claim of universal mutation-harness soundness.

## Generate and verify explicit source state

The tool accepts one to twelve named sibling Git checkouts. Set an external
output path that does not already exist; generation refuses to overwrite it.
Use the same explicit repository names when verifying:

```sh
python3 -B -S tools/cbz_inventory.py generate \
  --root /path/to/checkouts \
  --repositories main Math- governance- Universal-Law-Workspace meta-framework query- google-drive trial sandbox \
  --output /outside/checkouts/federation-inventory.json
python3 -B -S tools/cbz_inventory.py verify \
  --root /path/to/checkouts \
  --repositories main Math- governance- Universal-Law-Workspace meta-framework query- google-drive trial sandbox \
  --manifest /outside/checkouts/federation-inventory.json
```

This recovery's smoke run supplied no `--classifications`. Empty obligations
are explicit and cannot establish that the inventory of mathematical claims is
complete. Gitlinks are reported separately; the tool does not recursively
hydrate them or establish agreement with remote tips.

The implementation records HEAD, index, tracked working bytes and modes,
dirty status, symlink bytes and gitlink materialization. It rechecks file
fingerprints and HEAD/index/status after capture. Git invocations disable
ambient Git overrides, global/system configuration, replacement objects,
lazy fetch, protocols, hooks and fsmonitor. The original filter preflight had
a race before status; the repair below isolates status from source metadata
and retains preflight/postflight refusal. Each Git subprocess has a 60-second
timeout. Sources and any historical objects must already exist locally.

## Historical unchanged-recovery evidence — 9 November 2026

Python runtime: CPython 3.12.14. Raw local logs, command/exit records and the
inventory are retained outside source at
`/workspace/scratch/cbz-evidence/inventory-recovery/`. The original missing CLI
exited 2. After restoring only the tests, the existing deterministic-roundtrip
test failed with the intended missing-implementation assertion, exit 1.

Both complete retained-suite commands above exited 0: 75/75 tests, no top-level
skips, errors or failures, in normal and optimized mode. Each run exercised all
nine declared source mutants through the three retained controller methods.
The normal run took 105.711 seconds; optimized took 108.974 seconds. Raw stderr
logs are 11,885 bytes each, with SHA-256:

- normal: `7b3c3f3e6935b83ce071ed425c14bbe65d3d77157b039d3c2253fe0fbf3a3b54`;
- optimized: `35f7e951b52c0bb66bfba579c272c279c654cc506838efa2d365bc34659ebf57`.

The suite captures mutant subprocess output internally; separate raw child
transcripts are not emitted by the retained tests. The nine-mutant observation
therefore has exactly the classifier limits described above. Full candidate
regression, committed-tree G11 assessment and engineering review are separate
integration checks; no result for them is inferred from this focused suite.

Generate and verify each exited 0 over the nine explicit `/workspace`
checkouts below, with 27,826 tracked entries and eight Workspace gitlinks.
Both returned `cbz_green: false` and `semantic_claim_completeness: false`;
the inventory contains zero obligations. Each checkout's captured dirty flag
was false. This is an observation of those local source states, not a clean-Git
criterion, inventory of other hosts, or a future-state guarantee.

| Checkout | Captured HEAD | Tracked entries |
|---|---|---:|
| Math- | `f532c3e2d91a0c7679a37c733741660917a9b21b` | 15,499 |
| Universal-Law-Workspace | `a3e6552104d89917105a668a296121680a7f03bc` | 63 |
| google-drive | `ad08584b3ea9d533af10113e85169f471fa8ec93` | 37 |
| governance- | `1b62ca2f64b3012c2fdbbf7f55ced236c3042891` | 47 |
| main | `5ebe5771d44160f56d4f2fd1e11726858d17dc90` | 5,477 |
| meta-framework | `0f5894f81a5462c10ee5ec8f8109f466e712e47e` | 473 |
| query- | `fb4c457b5861dc49696fd0a8e11c7c405056c05f` | 55 |
| sandbox | `dce04a4174835107126ab4f36c199a2dc320f964` | 4 |
| trial | `3738c859d519d2dbf1e388fe48fe0ae60c9487a3` | 6,171 |

Inventory SHA-256:
`c03fcaa8613ff3c853c1cfe0c50f82e05e7a0bb20207a3fe462a722eaab493d3`.
The inventoried `main` checkout is the recovery parent; it is not a receipt for
the recovery candidate's later source tree.

## Certification boundary

The fixed certification scope is
`tracked-source-and-declared-binding-integrity-only`. The verifier checks
supplied source, manifest and receipt bytes and declared targets. It does not
authenticate that a claimed kernel execution happened, validate the intended
mathematics, establish semantic claim completeness, or award a CBZ criterion.
Dirty repositories can be captured and verified; recorded dirty state is not
rejected. SHA-1 Git repositories only are supported.

Optional QUARANTINED entries require nonempty boundary, blocking-reason and
isolation strings, owner `none` and maintenance `zero`. These supplied strings
do not prove logical isolation, and this older schema does not enforce the
complete eleven-field current governance contract. Historical bindings cannot
discharge active CLOSED sources. No historical classification is restored or
admitted by this recovery. Fresh classification reconciliation, actual formal
execution and the common final G0–G11 pass remain separate obligations.

## Current status-filter isolation repair — 9 November 2026

Engineering review reproduced a real side effect at the unchanged restoration:
a new clean-filter command inserted after preflight but before `git status`
executed a harmless marker-writing fixture. The inventory then refused the
configuration during a later check; that refusal happened after execution and
did not make the original preflight safe. The failing behavioral regression
preserves the indexed file size, since a size change can let Git classify the
path without invoking conversion and conceal the side effect.

Status now runs with private temporary Git metadata: fixed nonexecuting config,
captured detached HEAD, a bounded regular-file copy of the actual worktree's
index, and the actual common object directory supplied explicitly. It does not
copy the source config, commondir, refs, info files or shared-index dependencies.
Explicit absolute `--git-dir`/`--work-tree` paths prevent source metadata
discovery. No filter clean/process command is configured there, so a late
repository filter cannot execute even before its persistent configuration is
refused by the postflight probe. Other metadata-only Git queries retain the
existing hardened environment and do not run status in the source repository.

Global/system attributes and ignore files are suppressed separately from
config; private config fixes `core.attributesFile` and `core.excludesFile` to
the null device, and status sets `GIT_ATTR_NOSYSTEM=1`. Original
`.git/info/attributes` and `.git/info/exclude` are not imported. Consequently
the reported status follows the fixed metadata policy and working-tree
`.gitattributes`/`.gitignore`, not every local repository preference. Existing
source-info or ambient ignore rules may expose additional untracked paths.
This policy still records dirty state rather than rejecting it.

The index copy refuses symlinks, nonregular files, growth or identity changes,
and sizes above 256 MiB. Split indexes and other dependencies absent from the
private metadata fail closed; they are not silently expanded through original
metadata. Linked worktrees use their own index and common object directory.
Temporary metadata must be outside the common checkout parent. On this POSIX
implementation, `/var/tmp` is used if the ordinary temporary directory is
inside that parent, allowing sibling checkouts under `/tmp`. Unavailable or
unsafe temporary locations fail closed.

Six added controls cover the late-filter race, linked-worktree rename/untracked
status, split-index refusal, external temporary-directory fallback, ambient and
source-info ignore/attribute policy, and one source mutation bypassing status
isolation. That mutation first runs the unchanged copied tool against the same
single regression, then requires the mutated tool to produce exactly one
designated filter-execution assertion failure, exit 1, with no errors or skips.
The nine retained mutations and their original classifier limits are unchanged.

Repair evidence is retained separately under
`/workspace/scratch/cbz-evidence/inventory-recovery/filter-race-repair/`;
`final/` holds the stable-source qualification and new inventory artifact.
Earlier 79-test development runs are superseded and do not qualify the later
81-test source. The prior candidate-wide regression attempts stopped after the
review finding and receive no pass credit. No original failure is overwritten
or retrospectively promoted.

The final focused suite on CPython 3.12.14 and Git 2.52.0 passed 81/81 tests
in each mode, with zero top-level skips, errors or failures and all ten declared
mutants exercised. Normal runtime was 114.730 seconds; optimized was 119.235
seconds. Tool/test SHA-256 values matched before and after both runs:

| Current repaired source | Git blob | SHA-256 |
|---|---|---|
| `tools/cbz_inventory.py` | `2d100b241bc9d8bc6eb5ee6e9b16df229f41c447` | `350717ac4aa66481129a2e7ed8c2fdfd29fe9e2593b48443af826595feb3d983` |
| `tests/test_cbz_inventory.py` | `9eb45d45961169e86c3469ba81f53f5f216bae84` | `8d50ffdd4b4dac4d52279766c589407176ca99cd7226b64a30c1282fb3e43ecd` |

Each complete stderr log is 12,955 bytes; normal SHA-256 is
`81f0dadc2830aa0818724e9c10c5056a2027338fc631fb9a6a045719eda4ffca`,
optimized is
`9058643300f8eb19cd559dfe0418283c7d1a203db6601e36bb781be9bd24eb47`.
Fresh generate and verify with the repaired tool each exited 0 over the same
nine checkouts and 27,826 entries above, without classifications and with both
certification flags false. The new `final/federation-inventory.json` was
generated independently and has the same `c03fcaa8…` SHA-256 because the
inventoried source states are unchanged. This smoke still binds parent Main
`5ebe5771…`, not the repair candidate's later integration tree. Candidate-wide
regression and committed-tree G11 remain separate checks.
