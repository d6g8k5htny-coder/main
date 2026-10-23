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
`bindings_hypergraph.png` (the 22 hyperedges that touch a graph node).

## Rebuild / query
```
pip install -r tools/hypergraph/requirements.txt
python tools/hypergraph/math_hypergraph.py build --png
python tools/hypergraph/math_hypergraph.py reach node:math.d1-component.congruence-erratum 2
python tools/hypergraph/math_hypergraph.py gaps
python tools/hypergraph/math_hypergraph.py shared-impl 5
python tools/hypergraph/math_hypergraph.py validate
pytest tools/hypergraph
```

## Gap counts (2026-10-23)
344 units / 1235 vertices / 344 hyperedges. Sourced 211 (133 missing). Implemented 296 (48 missing),
of which 30 are placeholder-only ("in PR head" / "per census", no artifact). Bound units 14.
28 of 49 graph nodes unbound per #527; 13 nodes touched by no unit. Role problems 0.
Dropbox not crawled.
