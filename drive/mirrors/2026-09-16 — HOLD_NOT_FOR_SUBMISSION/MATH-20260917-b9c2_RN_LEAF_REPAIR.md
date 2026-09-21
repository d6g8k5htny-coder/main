# RN-UNIF adaptive-cover bookkeeping repair

Task: DQ-MATH-20260917-b9c2. Author-side implementation repair; **D3-LEMMA-RN-UNIF remains OPEN**.

The existing adaptive certifier records the largest attempted parent bound before deciding whether to subdivide. A rejected parent therefore remains in the final maximum even after its children all certify. Since the final check compares that maximum with the budget, any successful refinement still causes failure.

The patch updates `stats.max_sup` and `stats.max_cell` only for accepted leaves. Rejected parents are replaced by their descendants in the cover; failed terminal leaves still abort. The numerical bound formula, acceptance threshold, domain, subdivision rules, refinement floor, and frozen source carrier are unchanged.

## Evidence

`verify_leafmax.py` extracts the actual adaptive-control definitions from the original and derived engine files using Python AST. It uses mock derivative bounds and runs the original polar-cover loop. The underlying mock function is the constant 0.5, with an intentionally loose but valid gradient bound on the first parent, then exact zero derivative bounds.

- Original code evaluates all 2,307 accepted leaves after one parent subdivision, then fails its final aggregate check.
- Derived code accepts that same cover: 2,307 leaves, one refinement, 2,308 evaluations, maximum accepted bound 0.5.
- A terminal leaf with value 0.681, exceeding the 0.68 budget, still fails closed and does not enter accepted-leaf statistics.
- Normal and `python -O` outputs are byte-identical; both exit 0. No kernel, mathematical derivative envelope, or actual RN-UNIF cover was executed by these tests.

This is a control-flow regression, not a certification of the field or its theorem.

## Files and reproduction

- `accepted_leaf_max.patch`: minimal unified patch.
- `d3_rn_unif_leafmax.py`: derived full engine, preserving the original source file.
- `build_leafmax_patch.py`: source-hash-bound reconstruction of the patch and derived engine.
- `verify_leafmax.py`: actual-code regression with mock bounds.
- `BUILD_RECEIPT.json`, `REPAIR_VERIFICATION.json`, `test_normal.json`, `test_optimized.json`: scope, hashes, and results.

Run `python build_leafmax_patch.py`, then `python verify_leafmax.py` and `python -O verify_leafmax.py` from this directory. The build script resolves the source in the session's `intake/rn_source` tree and fails if its exact SHA-256 differs.

Source SHA-256: `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930`.

Derived SHA-256: `a04eb84d987514d855178b7c117f46d3080caf6ac7497fb471c1df49fc279d35`.

Shared exposure: the agent read the root reconnaissance memo, SHA-256 `29a117b8702e590e0a34005497bda19409c167c31d7d8ec291cb1b4452badb9a`, and its verified custody record before the repair. No blind-review or independent-provider credit is claimed. No Drive changes or scientific status promotions were made.
