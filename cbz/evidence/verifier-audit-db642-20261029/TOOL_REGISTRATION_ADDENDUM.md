# Existing tool updates for canonical registration

This is an admission input to the existing GOVERNANCE/TOOLING/STATUS writer, not a second registry.

| Tool | Purpose and change | Invocation | Self-verification and boundary |
|---|---|---|---|
| `tools/formal_gate_check.py` | Existing source/build/kernel receipt producer; records actual `GITHUB_RUN_ATTEMPT`, enforces exact negative diagnostics and retains prior attempt evidence | `python3 -B -S tools/formal_gate_check.py` (source mode is the default); full hosted reusable formal workflow | Producer tests include non-default attempt and absent local hosted context; seven arithmetic controls require exact exit/stdout/empty stderr, and axiom controls exercise the actual auditor. Mandatory diagnostic tests include adverse infrastructure outputs and prior-receipt retention. Proof sources unchanged. Source mode cannot establish kernel closure. |
| `tools/required_formal_check.py` | Existing live hosted receipt binder; requires exact producer/consumer attempt | Existing `.github/workflows` required-formal producer/aggregate entrypoints; CLI `--help` documents arguments | Missing, malformed and same-run stale attempts rejected. Isolated removal of only this comparison causes 11 adverse failures in three methods. |
| `tools/atlas_formal_evidence_build.py` | Reproduce the immutable historical Atlas evidence only | Qualified historical CLI below; runnable current-checkout fixture verification: `python3 -B -S -m unittest discover -s tests -p test_atlas_formal_evidence_build.py -v` | Exact historical archive, manifest, receipt and log identities remain pinned. Fixed historical source reader admits only three already retained manifest identities and validates bound bytes as inert data; current live source rules are not imposed on historical contracts. Historical receipt exception is not accessible through the live binder. |

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
repository. It runs the current fixed historical identity reader and exact receipt validator, and never executes historical control text. The current
checkout root is intentionally refused, including when a snapshot is supplied;
it is not the required historical projection. No new Lean execution is performed.

Reproduction: run `python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v`, with analogous patterns for `test_formal_gate.py`, `test_atlas_formal_evidence_build.py`, `test_atlas_formal_source_bridge.py`, `test_formal_workflow_shell.py` and `test_required_formal_binding_control.py`; repeat with `-O`. Full suite: `python3 -B -S -m unittest discover -s tests -v`.

Observed focused runs:130 tests per mode, one Lean skip each. Full suite1551 run,24 explicit skips,exit0. See raw source-bound evidence. Maintenance uses existing repository tool ownership; no new ongoing service. Serves criteria4,6,7,9,10; neither semantic Layer A nor final federation G0-G11 is certified by these tests.

Frozen composition scope: native receipt input `5aa6fbac2c66fac0e6e2332b7e95cacc3c37350e` plus negative-diagnostic input `4da34569f2e6e70267d2ee9d3170916b9b598279`. `tools/atlas_formal_source_bridge.py` uses the same fixed historical identity reader; `python3 -B -S -m unittest discover -s tests -p test_atlas_formal_source_bridge.py -v` reproduces both retained profiles. Any subsequent PR689 compiled-inventory work is excluded. These checks establish bounded Python/source behavior; fresh hosted Lean qualification remains required.

## Successor registration input

Frozen PR689 successor `50575e9823b883f7d766e0d9a45008c5e49c2fc6` adds existing-gate helper `tools/formal_compiled_inventory.py`, documented in `formal/COMPILED_INVENTORY.md` and exercised with `python3 -B -S -m unittest tests.test_formal_compiled_inventory -v` and the same argv with `-O`. It losslessly enumerates every package kernel constant, checks exact module/name/kind/provenance inventory and transitive axiom closure, and separately counts/binds IR ownership entries. The current source inventory is 45 kernel constants (31 target theorems and14 definitions) plus11 IR names. Only actual pinned-Lean execution can establish a compiled observation; absent runtime is an explicit test skip.

`tools/formal_gate_check.py --run-lean` runs compiled enumeration after build/leanchecker and target audit, records the aggregate inventory and transcript digest in the receipt, and retains exact attempt identity. The bound hosted workflow runs all gate/diagnostic/compiled controls in both modes before its dedicated final kernel run and live binding, and checks the official elan4.2.4 archive SHA before installation. Workflow, shell regression, compiled helper/tests/docs and every inventory record remain mandatory bound controls. All seven Python auditor mutations and permissive axiom-auditor control remain tests; they are not claims of real kernel execution in this session.
