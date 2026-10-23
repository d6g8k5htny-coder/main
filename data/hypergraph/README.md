# Math artifact hypergraph — organization only, NOT A RESULT

credit 0 · OBL OPEN · no flags. This indexes existing records; it makes no correctness claim.

Built with Hypergraph-DB 0.4.0 by `tools/hypergraph/math_hypergraph.py`. Each coverage-ledger row is
**one hyperedge** whose `roles` map labels members as unit / source / review / impl / node / lane / record.

## Inputs (all in-repo)
- `math_coverage_ledger_20261023.md` — RETRO-MATH-BIND-02 ledger copy (header corrected: sourced 211, not 215).
- `../../docs/site/dependency-source/GRAPH.json` — 49 graph nodes (#527).
- `graph_binding_report_20261023.json` — #527 binding report snapshot (21 bound nodes).

## Outputs
`math_hypergraph.json` (vertices + hyperedges), `query_results.json` (gaps, reach example, shared impls),
`bindings_hypergraph.png` (CI artifact only: the hyperedges that touch a graph node).

## Rebuild / query
Locally (from the repo root, Python 3.11/3.12):
```
pip install -r tools/hypergraph/requirements.txt
python -m tools.hypergraph.math_hypergraph build            # regenerate these JSON files (main only)
python -m tools.hypergraph.math_hypergraph build --check    # CI gate: exit 1 if stale
python -m tools.hypergraph.math_hypergraph reach main:node:math.d1-component.congruence-erratum 2
python -m tools.hypergraph.math_hypergraph gaps | validate | shared-impl 5
python -m tools.hypergraph.math_hypergraph export /tmp/hg-bundle --repos all --png
pytest tools/hypergraph
```
On GitHub: `.github/workflows/math-hypergraph.yml` runs the same steps on PRs/pushes touching these inputs or via
workflow_dispatch, with read-only checkouts of public Math- and query-, and uploads the all-repo bundle + PNG as the
`math-hypergraph` artifact. Vertex keys are repo-prefixed (`main:`, `Math-:`, `query-:`). See `tools/hypergraph/README.md`.
The PNG is a build artifact (optional, `--png`), not committed.

## Gap counts (2026-10-23)
344 units / 1235 vertices / 344 hyperedges. Sourced 211 (133 missing). Implemented 296 (48 missing),
of which 30 are placeholder-only ("in PR head" / "per census", no artifact). Bound units 14.
28 of 49 graph nodes unbound per #527; 13 nodes touched by no unit. Role problems 0.
Dropbox not crawled.
