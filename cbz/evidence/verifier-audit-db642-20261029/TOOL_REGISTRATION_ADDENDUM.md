# Existing tool updates for canonical registration

This is an admission input to the existing GOVERNANCE/TOOLING/STATUS writer, not a second registry.

| Tool | Purpose and change | Invocation | Self-verification and boundary |
|---|---|---|---|
| `tools/formal_gate_check.py` | Existing source/build/kernel receipt producer; now records actual `GITHUB_RUN_ATTEMPT` | `python3 -B -S tools/formal_gate_check.py` (source mode is the default); full hosted reusable formal workflow | Producer tests include non-default attempt and absent local hosted context. Proof sources unchanged. Source mode cannot establish kernel closure. |
| `tools/required_formal_check.py` | Existing live hosted receipt binder; requires exact producer/consumer attempt | Existing `.github/workflows` required-formal producer/aggregate entrypoints; CLI `--help` documents arguments | Missing, malformed and same-run stale attempts rejected. Isolated removal of only this comparison causes 11 adverse failures in three methods. |
| `tools/atlas_formal_evidence_build.py` | Reproduce the immutable historical Atlas evidence only | Qualified historical CLI below; runnable current-checkout fixture verification: `python3 -B -S -m unittest discover -s tests -p test_atlas_formal_evidence_build.py -v` | Exact historical archive, manifest, receipt and log identities remain pinned. Historical exception is not accessible through the live binder. |

The historical CLI requires an externally prepared pinned source projection and
its exact snapshot, as described in [the existing historical reproduction guide](../../../docs/site/README.md#one-source-linked-formal-companion).
From the current repository root, with `HISTORICAL_SOURCE_PROJECTION` and
`EXACT_HISTORICAL_JSON` set to those absolute paths, the actual arguments are:

```sh
python3 -B -S tools/atlas_formal_evidence_build.py --root "$HISTORICAL_SOURCE_PROJECTION" --snapshot "$EXACT_HISTORICAL_JSON" --check
```

The projection must contain the original `3b39c42` formal manifest (SHA-256
`8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090`),
all files bound by that manifest with their exact historical bytes, the matching
`docs/site/dependency-source/GRAPH.json`, and the original `review.json`,
`review.md` and `verified-execution.zip` under
`docs/site/evidence/side24-main-3b39/`. With `--check`, it must also contain the
unchanged `docs/site/atlas-formal-evidence-data.mjs` to compare. The snapshot must
match that graph and proof inventory; an arbitrary snapshot cannot establish
historical identity. The existing fixture test assembles the historical source
projection from `tests/fixtures/atlas_formal_build_3b39/` and unchanged repository
files, uses `tests/fixtures/atlas_formal_projection.json` as its qualified snapshot,
and supplies the original archive and comparison evidence directly from the
repository. It runs the current validator and never executes historical control text. The current
checkout root is intentionally refused, including when a snapshot is supplied;
it is not the required historical projection. No new Lean execution is performed.

Reproduction: run `python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v`, with analogous patterns for `test_formal_gate.py`, `test_atlas_formal_evidence_build.py`, `test_atlas_formal_source_bridge.py`, `test_formal_workflow_shell.py` and `test_required_formal_binding_control.py`; repeat with `-O`. Full suite: `python3 -B -S -m unittest discover -s tests -v`.

Observed focused runs:130 tests per mode, one Lean skip each. Full suite1551 run,24 explicit skips,exit0. See raw source-bound evidence. Maintenance uses existing repository tool ownership; no new ongoing service. Serves criteria4,6,7,9,10; neither semantic Layer A nor final federation G0-G11 is certified by these tests.
