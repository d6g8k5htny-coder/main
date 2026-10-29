# Tooling — proposed contribution

The machine registry is [tool-registry.json](tool-registry.json). Original worker registries/reports are preserved under sources/tooling/ and sources/lean-foundation/. Source hashes, exact commands and scope limits accompany active tools. The baseline runner and frozen coverage companion are registered. Fresh focused test_cbz_*.py runs passed 114 tests normally and 114 with Python optimization for the observed run window (see sources/tooling/current-tool-selftests*.log). The Lean wrapper changed afterwards; the separately sealed R3 receipt now verifies the strengthened code with 19 normal + 19 optimized tests and 7 actual Lean controls. Earlier 114-test run-window evidence is retained separately. This is not all-project regression or complete tool hygiene.

| Tool | Command/self-verification | Demonstrated boundary |
|---|---|---|
| cbz_audit.py | `python -B tools/cbz_audit.py selftest`; `python -B tools/cbz_audit.py scan --root .` | finite lexical/source inventory; 43 normal and optimized tests, 12/12 finite mutants in original worker receipt; not semantic Layer A completion |
| cbz_lean_check.py | `python -B -S tools/cbz_lean_check.py`; unittest `test_cbz_lean_check.py` | 31 literal Main statements, 45 original constants, finite witness domains, named-module fresh kernel replay and 7 actual controls; analytic alignment excluded |
| formal_gate_check.py | see original lean-foundation registry; unittest `test_formal_gate.py` | existing build/kernel/target-axiom and source controls; no source parent theorem inference |
| cbz_refs.py | unittest `test_cbz_refs.py`; normal and optimized modes | 22 tests, 6 safety mutations, 2,215-tip disposable scale fixture; the fixture count is not the observed 2,115 remote-branch count. Root alone uses mutation commands against real remotes |
| cbz_storage.py | unittest `test_cbz_storage.py`; normal and optimized modes | exact immutable duplicate bytes/paths/modes and verified archive preservation; no mathematical effect |
| cbz_baseline.py | `inspect`, `run`, `cold-start` commands in STATUS.proposed.md | namespaced configuration/receipt/source/doc consistency only; missing gates stay INCOMPLETE |

Use the pinned Lean 4.34.1 release and default inherited proxy/CA. The worker verified restored runtime/header/static-library bytes against official archive members, preserved earlier failed attempts, and observed all 9 Math dependency revisions. Math cache/build residue remains explicit; a source-readiness/static validator is not a current project kernel execution.

Archive/source copies are data, not tools to execute blindly. The preservation/generation scripts and original registries retain their one-off scopes. No scanner, hash check, fixture, merged PR or green unit test substitutes for exact source-statement alignment, nonvacuity, full regression or scientific review.

<!-- cbz-gates: {"G0":"INCOMPLETE","G1":"INCOMPLETE","G2":"INCOMPLETE","G3":"INCOMPLETE","G4":"INCOMPLETE","G5":"INCOMPLETE","G6":"INCOMPLETE","G7":"INCOMPLETE","G8":"INCOMPLETE","G9":"INCOMPLETE","G10":"INCOMPLETE","G11":"INCOMPLETE"} -->

Coverage companion: original nine containers inventoried (1,042 members and 6 notebook cells); 31 archived Lean modules are QUARANTINED with 9 token hits preserved, 25 control-log hashes checked. Its 15 normal/optimized tests and 9/9 source mutations establish only the finite declared scope. Older boundary mismatches remain in their exact snapshot; the new R3 seal independently binds its own current baseline/log chain. Commands (cwd repository root):

```sh
python -B cbz/evidence/website-foundation-20261029/sources/tooling/auditor/coverage/coverage_selftest.py
python -B cbz/evidence/website-foundation-20261029/sources/tooling/auditor/coverage/coverage_companion.py --root . --g1 cbz/evidence/website-foundation-20261029/sources/tooling/auditor/coverage/G1-main.json
python -B cbz/evidence/website-foundation-20261029/sources/tooling/auditor/coverage/coverage_boundary.py .
```

These commands print observations to stdout; preserve new output outside the source root. No notebook/archive inventory certifies dynamic Python behavior or source-parent alignment.

Portable control replay is included in `verification/`. Each command requires a **new** external output directory; choose another name when repeating a run. The saved packaged runs observed 26 ref tests and 18 baseline tests passing, 7/7 meaningful mutants rejected for each tool, and four CLI refusals each. The ref run also preserved 2,215 distinct fixture tips while deleting 2,214 fixture branches. This is engineering safety evidence, not a production ref cleanup or mathematical certificate.

```sh
python -B cbz/evidence/website-foundation-20261029/verification/ref_controls.py --root . --output /tmp/cbz-ref-controls
python -B cbz/evidence/website-foundation-20261029/verification/baseline_controls.py --root . --output /tmp/cbz-baseline-controls
```

The contribution identity checker is now portable from this checkout layout and uses explicit `require` guards, including under `python -O`. Run `python -B cbz/evidence/website-foundation-20261029/construction/verify_proposal.py` and the same command with `-O`. It verifies source fields/hashes and four finite negative metadata controls; it does not execute a kernel or establish complete scientific coverage.

The additional `construction/test_verify_proposal_guard.py` runs a copied unmutated positive harness and one weakened source-guard mutant; the mutant must fail exactly one assertion with no harness error or skip. Both Python modes were observed passing this finite source-mutation control. It is not a complete fault model.
