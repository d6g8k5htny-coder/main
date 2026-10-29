# CBZ tooling

Checkpoint registry, **not G11 final admission**. Exact final source/run evidence remains required.

| Tool | Usage/purpose | Actual verification/boundary | Owner/criterion |
|---|---|---|---|
| Math formal/gate.py + gate_helper.py | `LEAN_NUM_THREADS=2 python3.11 -B -S formal/gate.py --execute` | 244 tests/mode at recorded cut; complete core gate passed at8151 with compiled/type/axiom/adversarial controls. No global final qualification | none;6,7 |
| Main tools/formal_gate_check.py | `python3.11 -B -S tools/formal_gate_check.py --run-lean` | Fresh build/recheck/axioms/controls passed; attempt-binding repair separate, old receipts never retroactively upgraded | none;6,7 |
| Main tools/cbz_inventory.py, mission addition | `generate --root ROOT --output EXTERNAL.json`; `verify --root ROOT --manifest EXTERNAL.json`; optional `--classifications FILE` | 75/75 tests and nine intended source-mutant kills per mode. RF312 exact effective-map/blob binding; explicit replay source/target/axiom/raw-artifact binding. Actual 94/207 component roundtrips passed. No semantic/kernel authenticity certificate | none—self-contained; 1,3,8,11 |
| Main tools/public_shop_check.py / museum_check.py | Validate pinned fixture bytes and execute Node fixture suites | Observed4/4 and45/45; broad prepared Node763/763 zero skips | none;5,11 |
| Math downstream run_validation.py | Existing documented validation campaign |71 methods and24 semantic mutations per mode observed, scoped software gate only | none;7 |
| Git bundle/SHA-lease cleanup | `git bundle verify PATH`; guarded atomic ref deletion | Ancestor checks, excluded PR/protected/workflow refs, actual absence readback; no mathematical certification | none;1,2 |
| Bounded proof-closure replay | [Exact tool commands](cbz/evidence/kernel/additional/replay-tool/README.md), [additional package replay](cbz/evidence/kernel/additional/HOW_TO_REPLAY.md), [determinant replay](cbz/evidence/kernel/determinant/HOW_TO_REPLAY.md) | 31 tests/three Python mutants per mode; actual Lean positive/six adverse/six auditor mutants in both modes. Official empty trust-zero kernel; supplied text does not authenticate execution | none;3,6,9,11 |
| PR/citation evidence | [Exact ordinary/optimized test and mutation commands](cbz/evidence/stale-pr-dispositions/README.md) | 35+9 tests and49 intended source/harness kills per mode. Actual network fetch and26-head historical regeneration. Advisory recorded facts, no current approval or mathematical certification | none;2,3,10,11,12 |

New census self-tests:

```bash
python3.11 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
```

Inventory output lives outside scanned roots. Immutable ancestor bindings retain exact Git blobs; current proof bytes must remain identical. Explicit historical quarantine bindings cannot discharge CLOSED proofs. Private evidence is not copied into public manifests. A named missing tool is not a finished tool.

The composed source prechecks and receipt-custody controls, with commands and self-verification scope, are documented in [canonical auditors](docs/cbz/CANONICAL_AUDITORS.md) and [compiled inventory](formal/COMPILED_INVENTORY.md). Their previous component results are not a final federation gate.

### Mutation-grader qualification limit

Source inspection found that both the lexical and A3 mutation graders accept any nonzero exit and an expected test failure even when foreign failures are present; neither requires the specific bypass reason. Their historical 66 and 23 mutant outcomes do not by themselves establish auditor soundness. Test-first repairs and fresh reason-bound controls are pending, so G0 and federation-wide G1 remain unqualified.

### Historical main regression skip observation

The archived [one-shot workflow](cbz/evidence/main-regression-skip-observation/workflow.yml.txt) downloaded the two specified hosted artifacts, checked their GitHub-reported ZIP digests and the original log digests, and listed the skipped test lines. The actual positive observation, exact source/run and limitations are in [result.json](cbz/evidence/main-regression-skip-observation/result.json). It is a historical diagnostic, not a Layer A auditor or a full G5 certificate; it has no dedicated adversarial test suite and must not be promoted as a certifier without one. Maintenance owner: none. To reproduce while the original GitHub artifacts remain available, run the exact archived workflow with `actions: read` or apply its listed hash and unzip commands to those exact artifact IDs.

### Atlas browser regression runner

`tools/run_atlas_browser_regressions.py` executes the two fixed Atlas UI test IDs with zero skips; [usage, ordinary/optimized self-tests and historical observations](docs/cbz/ATLAS_BROWSER_REGRESSIONS.md). Historical browser success at `c0205ca9` does not qualify the composed candidate. Owner: none, self-contained; criteria 5 and 11. This tool certifies no mathematics.

### Historical source-cut diagnostic workflows

The [sandbox artifact observer](cbz/evidence/main-regression-skip-observation/sandbox-observer-workflow.yml.txt) downloaded one recorded GitHub Actions ZIP, checked its SHA-256 and printed ten passing method lines; [result and limits](cbz/evidence/main-regression-skip-observation/sandbox-controls.json). Run its archived workflow with `actions: read` while the original artifact exists. It has no dedicated adversarial tests and is not an auditor or G5 certificate. Owner: none.

Two one-shot source-cut tests are queued: [pytest-dependent controls](https://github.com/d6g8k5htny-coder/main/blob/ba4077c2b3101cd969319b342e30c11b725a5f87/.github/workflows/cbz-pytest-controls.yml) and [pinned graph-composition controls](https://github.com/d6g8k5htny-coder/main/blob/8b03af28ccfe3276b8a8957b108ba541f17953d4/.github/workflows/cbz-graph-composition.yml). Their exact workflows give dependencies and commands; no completed result, independent self-test or gate admission is asserted. They are engineering diagnostics, not mathematical certifiers. Owner: none.

### Scoped pair-law Lean workflow

The archived [workflow](https://github.com/d6g8k5htny-coder/Math-/blob/5a470bfd739aba22e32b91587d5dfc031a7de3fc/.github/workflows/cbz-lifetime-pair-law-countermodel.yml) builds `Specifications.Lifetime`, compiles `CounterexampleE4.lean`, prints axioms and runs `leanchecker` in the pinned Math package. [Observed run and exact commands](cbz/evidence/lifetime-countermodel/observation.json). It is a source-scoped proof execution, not a general auditor; the GitHub artifact has not independently been downloaded and no final federation gate follows. Maintenance owner: none; criteria 3 and 6. Re-run on the exact branch cut or after integration with newly recorded source and results.

The separate [zero-law proof workflow](https://github.com/d6g8k5htny-coder/Math-/blob/cb1fe3d189d94ee207750fc005eadf0596fb9cef/.github/workflows/cbz-lifetime-pair-law-countermodel.yml) extends the scoped pair-law run by compiling and checking `CounterexampleZero.lean`. Its exact source and commands are at the linked commit; its hosted result is pending, and it is not an admitted general auditor. Owner: none; criteria 3 and 6.

### Pending historical classification reconstruction

The isolated [builder and exact usage](https://github.com/d6g8k5htny-coder/main/blob/43c011a79eced964b923d3fc18c3641560e79a64/cbz/reconstruction-367/README.md) reconstructs a new 367-record source/evidence binding file, then invokes the existing inventory generator and verifier on the documented main/Math source cuts. It has 15 proposed controls, five source mutations and one loader mutation per mode; run `python3.11 -B -S cbz/reconstruction-367/mutations.py` from that isolated candidate, and the README gives the builder and inventory commands. The hosted self-tests, generated file and both inventory checks are queued, not observed. The tool checks bindings, does not authenticate a kernel run or prove semantic completeness, and is not yet admitted for gating. Owner: none; criteria 3, 8 and 11. Two later pair-law declarations need separate inclusion.

The proposed [current-base pair-law workflow](https://github.com/d6g8k5htny-coder/Math-/blob/9e89980bfa4f281e00550f481202033775f35485/.github/workflows/cbz-lifetime-pair-law-countermodel.yml) in [Math #543](https://github.com/d6g8k5htny-coder/Math-/pull/543) retains the statement packet inventory, copies the exact proof temporarily for the scoped build, compares both axiom outputs and runs leanchecker. The historical workflow remains the evidence for the already recorded branch-cut declarations; the integration workflow and current-head tests are queued, not qualified. Owner: none; criteria 3 and 6.

### Replay-helper mutation-grader qualification limit

Source review found that the historical bounded replay helper's Python mutant classifier accepts a named failure plus an `AssertionError` appearing anywhere in the same failing transcript, without excluding other failures/errors or binding the assertion to the selected guard. Its test loader can also read existing bytecode. These are source-level findings, not a newly executed adverse test. The old mutation counts do not establish auditor soundness; targeted controls and any fix are pending. The historical Lean executions and literal source/axiom records remain preserved at their stated cuts. [Exact source review](cbz/evidence/replay-helper-grader-source-review.json).

The same source review found that the historical replay mutation runner's baseline condition does not check a nonzero test count, and that the separate Lean self-test normally imports the helper. A zero-test baseline and stale-bytecode path therefore need actual adverse controls; these are not claimed as executed failures yet.

### Replay delivery repair candidate

Main PR #705 head `30184a6c7337099f910809116523112394e6504b` restores both four-file sets using the original Git blobs and adds a separately checksummed source-only Python mutation-grader candidate with adverse tests. Its checksum, historical, and candidate workflows are queued. Historical files remain preserved; the candidate does not repair the independent Lean self-test issue and is not yet verified. The PR's governance and status surfaces require reconciliation with this staging branch before integration.
