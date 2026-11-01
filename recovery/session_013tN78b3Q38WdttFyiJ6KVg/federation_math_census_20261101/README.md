# Federation census of mathematical artifacts against the CBZ quarantine catalog (1 Nov 2026)

By `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0; scientific effect
NONE. This extends `../math_coverage_gap_20261031/`, which counted `PROOF.md` files only, to every
mathematical artifact in all 12 repositories of the account. It records coverage only. It closes,
quarantines or admits nothing.

- **Catalog.** `cbz/quarantine/catalog.json` on `cbz/foundation-catalog-reconciled-20261029` at
  `32ab7d90`: 307 quarantines, blob `702f0737`. The CLOSED declarations are #695's
  `cbz/evidence/kernel/all-closed-declarations.json` (367 entries).
- **Refs.**
  - `config_cut.json` uses the catalog's own cuts: Math- `8151f592` and main `81acc060`.
  - `config_tip.json` uses current tips: Math- `19a55750` and main `fe8c0fbfbaa729256f168f3bcd5423902680747b`.
  - Both use the other ten repositories at their HEADs, as listed in the configs.
- **Candidates.**
  - `.lean` files;
  - proof notes: Markdown whose name contains PROOF, THEOREM, DERIVATION, ERRATUM or LEMMA, plus
    all Markdown under Math- `imports/` and `specifications/*/sources/`;
  - certificate programs and certificate data.
- **Classes.** CLOSED_SOURCE, QUARANTINED (by location, at packet level), PROPOSED_QP_C54,
  COPY_OF_COVERED (by Git blob, across repositories), INVENTORY_ONLY and NONE. `reviews/*` rows are
  marked `review_evidence`; they are not proof obligations.
- **Command.** `python3 -I -B federation_census.py CONFIG CATALOG CLOSED QP_RECORDS OUT.json`.
  It is deterministic: a rerun with the same refs gives identical output.
- **`gap_summary_tip.json`.** The NONE and INVENTORY_ONLY rows grouped by packet. Each group lists
  any quarantine record that names the packet or file in its text.
  - A record that names the packet but does not use it as a location may still dispose of it at
    claim level ("bytes not pinned"); Q-1043 for `frontiers/intermediate_window_20260928` is one.
  - A record that only cites it, for example as a review link, does not.
  - Read each named group before treating it as a gap.
- **Limit.** Files whose names match none of the patterns are not censused.
