# Finite companion population and callback correction

Only `tools/workspace_git_inventory_mutations.py` and
`tests/test_workspace_git_inventory_mutations.py` changed in
`/workspace/cbz-workspace-git-hygiene-20261029`. Collector and core-test hashes
remain `697f87e9184728b90caaf50e2fe9a71fbb950719c6d325987f114271c343c314`
and `b3d94e36111154fd255193c118d87128e39aca2aea1dcd8f5cc2f72837743844`.
No core edit, staging, commit, canonical documentation, GitHub/publication,
census or foreign-source change was performed. Parent owns those surfaces.
`SOURCE-FINAL.json` records exact final hashes; `PATCH.diff` is this unit's
two-file delta against the personally verified frozen `639be769...` runner.

## Reproduced gaps and bounded rule

`population-before/results.json` and `CONTROL-BEFORE.json` personally reproduce
exit0/complete=true/all_detected=true with fourteen separate instances of only
the unborn/detached test. The original assertions and named mutant selection
remain unchanged in the copied fixture. `population-RED.log` additionally shows
a suite loading all fourteen distinct methods but actually executing only one
fourteen times. `class-RED.log` reproduces admission of an unexpected subclass.
`reproduce_population.py` retains the frozen and corrected replay controls.
The corrected replay exits1 and runs zero mutants (`CONTROL-AFTER.json`).

Baseline admission now requires the exact explicit fourteen-method manifest,
once each. The companion freezes the original InventoryTests class object,
canonical ID prefix, method function objects and code objects BEFORE calling the
loader. Method names/printed IDs alone are insufficient: `pristine-class-RED.log`
reproduces a loader substituting another class with the same displayed name;
`method-alias-RED.log` reproduces all fourteen names aliased to one actual body.
Both benign controls are rejected by the final source.

Loaded identities, framework-started identities, actual body-entry identities
and body-exit identities must each match the expected multiset exactly, with
pristine class/function/code checks at load and actual execution. Duplicates,
omissions, unexpected types, rebinding and aliasing are refused. Normal return
or an actual escaping synchronous exception records body exit; rejected deferred
returns do not. Baselines must additionally pass all fourteen tests without
fixture issues, unsupported bodies, skips, errors or optimization mismatch.
Named mutants must load/start/enter/exit precisely their expected case once.

The driver retains the EXACT escaping AssertionError object from the designated
body. A reported failure must refer to that same object, in body phase and with
the method in its traceback. `callback-RED.log` reproduces a swallowed assertion
fed manually to result.addFailure; the corrected source rejects it. Existing
outer lifecycle phase attribution remains, including setup/teardown/cleanup
refusal even when a method was called by a fixture.

Current core bodies have no subtests (personally checked with rg). All subtest
reports are explicitly unsupported, including successful reports and foreign
parents. `subtest-RED.log` reproduces native subtest false credit before this
guard; black-box tests cover native failure, foreign-parent failure (one failure,
zero errors), and fourteen successful fixture subtests refusing the baseline.
No generic subtest/async/plugin engine is introduced. Deferred-body refusal,
actual -O propagation, zero-test/error/survivor/lifecycle controls remain.

This is a bounded engineering verifier for the frozen synchronous stdlib
InventoryTests protocol. It does not certify arbitrary malicious Python sources
or testing plugins, mathematical truth, clean Git, global gates or CBZ-0.

## Verification, counts and reproducibility

The unchanged source-mutation contract is **14 baseline methods / 7 meaningful
source mutants**. This sidecar does not import peer 26/22 counts. The aggregate
test population is **31 = 14 unchanged core + 17 companion meta tests**.
Final source-bound results are in `SUMMARY.json`; current aggregate logs use the
`qualified-python312-*` / `qualified-python311-*` names. Older 29/30-test logs,
trial reports and intermediate source snapshots are historical, not final
qualification. `SOURCE-INTERMEDIATE.json`, `INTERMEDIATE-PATCH.diff`, and the
explicit intermediate runner/meta files preserve those exact inputs.

Final observed qualification: **31/31 PASS in all four aggregate modes**, normal
and -O on Python3.11.17 and3.12.14, no failures/errors/skips. The two standalone
companion modes qualify **7/7** source mutants with actual child optimization0/1.
All **5/5** separate guard mutations are caught by their black-box assertion
tests, with zero unittest errors. `SUMMARY.json` records these final observations.

`mutations-qualified-normal/results.json` and
`mutations-qualified-optimized/results.json` retain the seven source-mutant
reports, exact loaded/started/body identities, real escaping assertion attribution
and actual child optimization. All seven mutation edits/targets are unchanged.

`check_guard_mutations.py` separately tests five companion-guard regressions:
remove population gates, remove escaping-object provenance, remove subtest
refusal, recapture a loader-rebound class, and bypass pristine method checks.
Each must cause its designated black-box meta test to fail by assertion, not
error/timeout. Raw mutated copies/hashes/logs/results are under
`guard-mutations-qualified/`; these five auditor controls are separate from the
seven collector source mutants.

From a checkout with existing Python/Git:

```sh
python3.11 -S -B -m unittest discover -s tests -p 'test_workspace_git_inventory*.py' -v
python3.11 -S -B -O -m unittest discover -s tests -p 'test_workspace_git_inventory*.py' -v
python3 -S -B -m unittest discover -s tests -p 'test_workspace_git_inventory*.py' -v
python3 -S -B -O -m unittest discover -s tests -p 'test_workspace_git_inventory*.py' -v
python3 -S -B tools/workspace_git_inventory_mutations.py --output /private/existing-parent/NEW-directory
python3 -S -B -O tools/workspace_git_inventory_mutations.py --output /private/existing-parent/ANOTHER-NEW-directory
```

No dependencies or runtime links were installed/changed. Existing limits remain:
60s baseline, 20s per source mutant, 90s per black-box companion CLI call; private
exclusive output directory0700/files0600. Guard-control tests have a finite180s
outer subprocess timeout. Only owned temporary copies/children are used.

Parent registry/report updates should use the final hashes, exact31 aggregate
tests,14 expected baseline identities,7 source mutants and5 separate guard
mutations. Source-bound GREEN/RED history is preserved; no approval or merge
claim is made. Author self-review plus automated evidence, actual OpenAI/GPT-6
Codex engineering only, org0/scienceNONE. This releases only this bounded unit.
