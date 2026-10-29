# Existing tool updates for canonical registration

This is an admission input to the existing GOVERNANCE/TOOLING/STATUS writer, not a second registry.

| Tool | Purpose and change | Invocation | Self-verification and boundary |
|---|---|---|---|
| `tools/formal_gate_check.py` | Existing source/build/kernel receipt producer; now records actual `GITHUB_RUN_ATTEMPT` | `python3 -B -S tools/formal_gate_check.py --source-only`; full hosted reusable formal workflow | Producer tests include non-default attempt and absent local hosted context. Proof sources unchanged. Source mode cannot establish kernel closure. |
| `tools/required_formal_check.py` | Existing live hosted receipt binder; requires exact producer/consumer attempt | Existing `.github/workflows` required-formal producer/aggregate entrypoints; CLI `--help` documents arguments | Missing, malformed and same-run stale attempts rejected. Isolated removal of only this comparison causes 11 adverse failures in three methods. |
| `tools/atlas_formal_evidence_build.py` | Reproduce the immutable historical Atlas evidence only | `python3 -B -S tools/atlas_formal_evidence_build.py --check` | Exact historical archive, manifest, receipt and log identities remain pinned. Historical exception is not accessible through the live binder. |

Reproduction: run `python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v`, with analogous patterns for `test_formal_gate.py`, `test_atlas_formal_evidence_build.py`, `test_atlas_formal_source_bridge.py`, `test_formal_workflow_shell.py` and `test_required_formal_binding_control.py`; repeat with `-O`. Full suite: `python3 -B -S -m unittest discover -s tests -v`.

Observed focused runs:130 tests per mode, one Lean skip each. Full suite1551 run,24 explicit skips,exit0. See raw source-bound evidence. Maintenance uses existing repository tool ownership; no new ongoing service. Serves criteria4,6,7,9,10; neither semantic Layer A nor final federation G0-G11 is certified by these tests.
