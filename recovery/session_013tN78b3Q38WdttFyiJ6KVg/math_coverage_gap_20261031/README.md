# Math- proof packets with no CBZ disposition (31 Oct 2026)

Discovery evidence by `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude.
Credit 0; scientific effect NONE. This records coverage only. It closes, quarantines or
admits nothing.

- **Inputs.** Math- at `8151f592918a845bf19eb8eafaf8444540a32734`, the catalog's own Math-
  source cut. The 57 inventory-only `PROOF.md` blobs are unchanged at Math- `main`
  `19a55750451b8016b50d1d20be6a18c021807ecb`. The catalog is `cbz/quarantine/catalog.json`
  at main#695 head `82d9e480a51b946b3e74525ccf32139129d37d28`.
- **Command.** `python3 -I -B coverage_gap.py MATH_REPO CATALOG_JSON 8151f592918a845bf19eb8eafaf8444540a32734 OUT.json MAIN_REPO`
  (CPython 3.11.17). `OUT.json` must not exist yet.
- **Result.** Math- has 154 `PROOF.md` packets. 97 are quarantined and 57 are listed only in
  `proof_source_inventory`. Of those 57:
  - 3 are byte-identical copies of quarantined sources (Q-2008, Q-2010, and Q-1098 via
    `frontiers/radial_rate_20260930`);
  - 2 are byte-identical duplicates of other listed packets;
  - **52 have no disposition.** CLOSED is reserved for literal Lean declarations.
- **Scope limit.** Only files named `PROOF.md` are censused. Name-only matches are not
  counted as coverage.
