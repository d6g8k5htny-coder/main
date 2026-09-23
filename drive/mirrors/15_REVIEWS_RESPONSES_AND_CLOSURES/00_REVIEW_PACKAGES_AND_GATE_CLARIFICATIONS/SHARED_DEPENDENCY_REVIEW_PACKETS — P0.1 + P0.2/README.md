# Mirror of `…/00_REVIEW_PACKAGES_AND_GATE_CLARIFICATIONS/SHARED_DEPENDENCY_REVIEW_PACKETS — P0.1 + P0.2` (partial)

Drive folder id `128nZmfBbFTMQNwonG8N4sLNgfqNxQB4m` in the reviews lane (16 inventory items). This
directory holds one reading copy: `GP-DER-197-v1.0`, the tenth exact P0.2 object, which the Drive files
here rather than in the canon lane. The other nine exact objects and their routing documents are
mirrored under `drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/` (see its README). The lane README one
level up describes the rest of `15_REVIEWS_RESPONSES_AND_CLOSURES`. `_MANIFEST.jsonl` is checked by
`tools/verify_manifests.py`. **Mirroring is not review, endorsement or promotion.** Ported 2026-09-19.

## What is here

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact | frozen body: bytes / SHA-256 / rule / matches register |
|---|---|---|---:|---|---|---|
| `1hGQVFDZKHMcdG94_OIgVEwipTR6pvKCk_-llHyIq3uQ` | GP-DER-197-v1.0 — Measurable Selected Unstable Branch and Borel Exact Adjacency Event | `GP-DER-197-v1.0 — Measurable Selected Unstable Branch and Borel Exact Adjacency Event.export.txt` | 25,495 | `1974183747617041ab84d679fa632f8c22d9cf3dd09ed97ae27ad9f7b9573f71` | false (reading copy) | 13,797 / `035d5a18018606bd8736dc5774a7adbed29de453b30a1780976e1e4f69285d4d` / OP-PROT-017 / **true** |

The body digest (13,797 B, `035d5a18…`) reproduces the row `registers/json/p02_exact_hash_review_manifest.json`
declares for this Drive id, and the digest the export's own prepended banners and the P0.2 routing
documents cite (identity of the body, not review).

## The banners, verbatim

* Header: `Class: DER / MEASURABLE DYNAMICAL MARK / BOREL EVENT / P0.1-P0.2 INTERFACE REPAIR` · `Authority: none` · `Canonical mathematical impact: NONE` · `Independence credit: ZERO — same OpenAI organizational/model family` · `Status: COMPLETE SAME-LINE PROOF USING STANDARD PARAMETER-DEPENDENT UNSTABLE-MANIFOLD AND ODE-CONTINUITY THEOREMS / ORGANIZATIONALLY DISTINCT REVIEW OPEN`.
* Prepended "CURRENT LM013 SAME-LINE SUCCESSOR VERIFICATION — 2026-07-30": "Status remains NONTERMINAL: organizationally distinct exact-hash conversion and the remaining parent/integration predicates are still open. Zero independent-review credit; no parent-body change; no promotion." `[[GLOBALIZATION:v1.1-SAME-LINE-PASS]] [[EXTERNAL-CONVERSION:OPEN]]`.

## How these bytes got here

Every object was fetched through the Drive connector (`download_file_content`, base64) and the
base64 was decoded from the session transcript straight to disk, so no model retyped any byte.
For a raw file the SHA-256 and byte count were recomputed from disk and had to equal the
`drive/inventory.jsonl` row (2026-09-17 snapshot) or the bytes were not stored; every stored raw
file here passed. A native Google Doc has no payload digest anywhere in the corpus: its text export
is stored as `<title>.export.txt` with `exact: false` (a reading copy), and where the export carries
full-line `BEGIN_*BODY` / `END_*BODY` markers the marker-delimited body digest is recorded and
compared with the digest the registers declare for that Drive id. `_MANIFEST.jsonl` holds one row
per object and `tools/verify_manifests.py` re-checks every digest in CI. Archives are not extracted
and nothing was executed; `<name>.zip.members.txt` is a derived member listing, not a Drive object.

## What this directory does not establish

Nothing here reviews, converts, promotes or closes P02-LM-013 or any interface. A reading copy is not
the object; a body digest equal to the register's is identity, not a verdict. The status is the
register's. No independence credit is awarded and no gate moves.
