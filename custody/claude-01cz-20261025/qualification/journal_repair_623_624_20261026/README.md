# Journal repair combined qualification prep (#623 + #624)

This is an operator preparation record from the stage claimed at main#605 comment 6405178037. It is not adoption, not a merge candidate, and not a review. Scientific effect: NONE.

## Composition

- Base: carrier `71049ac9eb4743a3d8c94477b8a13cc8e2739537`.
- #623 delta: head `3dcff2b86f3b2166f42fa67f53a4f5c86f2883a9` (LF-byte framing in `Journal._load`).
- #624 delta: head `5b5555089502aa77963a220bfbe0d4f156c4a62e` (exact-record byte bound in `Journal.append`).
- Combined tree: `git merge-tree --write-tree 3dcff2b8 5b555508` gives `d476fc21a00dbc10d726ee986b3eaa51d4a1aedf`. It merged cleanly and differs from the base in 4 files (+372/−8).

## Hydration (verified originals only, identical for base and combined)

- 17 files from custody `93e0984313a7f4283805cc2ec5d1186f53b07b6c` (`v19_dependency_supplement/sources/Universal_Law_v19_Research_Frontier/`), copied to the same relative paths. Every SHA matches the published manifest, and no copy overwrote an existing file.
- `engine/model.py`: the original archive member, extracted from main#605 comment 6404135507. SHA256 `0ff53aa9…`, 10,033 B.
- `engine/dataset.json`: the canonical dataset at `11f7b82e`, SHA256 `4cfae309…`. This is not the P15 successor.

See `HYDRATION_MANIFEST.tsv`.

## Outcomes

`OUTCOMES.txt` has the full table, and the per-file `*.summary.txt` files list the failing test names.

- Every test file's failing set is **identical** between hydrated base and hydrated combined (sorted FAIL/ERROR lists compared).
- The combined tree adds `JournalLfFramingTests` 6/6 (#623) and `test_journal_size_bound` 8/8 (#624), both OK.
- `ProofConductorTests` now executes against the real `model.py` and dataset: 32 of 38 pass on both trees.

## Unrun boundary (not a pass)

- **6 `ProofConductorTests`** (`'blocked' != 'scoped_kernel_checked'`) need the native Lean backend, which means the pinned Linux Lean 4.34.1 resources. These are absent here, and there is no executor workaround.
- `test_frontier_api` and `test_workspace` need module `library`; `test_flow` needs module `release`; `test_frontier_runtime` needs module `frontier_gpu`. None of these is in the carrier or the supplement.
- `test_formal_queue` needs `formal_examples/profile.json`, which is also absent. The supplement has `formal_frontier/profile.json` instead.
- One `test_flow` assertion (`'failed' != 'diagnostic_completed'`) fails identically on both trees.

These gaps exist on both trees, so the two repairs introduce no difference. A full PASS requires the missing inputs and the Lean resources.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
