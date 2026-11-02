# QP-C54 catalog candidate (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE.

This is a ready-to-apply candidate `cbz/quarantine/catalog.json`. It carries the verified QP-C54 proposals
**beside** the adopted rows, not inside them. It is for the composer or integrator of
`cbz/foundation-catalog-reconciled-20261029` (staging) and #695. Nothing here is pushed to those branches, and
nothing is adopted.

## What the patch does

`0001-qp-c54-catalog-candidate.patch` is a `git am` patch against staging `32ab7d9013264d2874aa4b778a90eede0cda0cdf`.
It changes one file, `cbz/quarantine/catalog.json`, by appending four keys after `reconciled_utc`:

| Key | Rows | Source |
|---|---|---|
| `proposed_quarantines` | 96 (52 tranche 1, 44 tranche 2) | `qp_c54_quarantine_proposals_20261101/records_verified.json` and the NEW_RECORD rows of `qp_c54_t2_20261102/records_verified_t2.json` |
| `proposed_pinning_amendments` | 2 (QP-C54-T2-031 → Q-1043, QP-C54-T2-034 → Q-1025) | PINNING_AMENDMENT rows of `records_verified_t2.json`, each with `target_record` added |
| `proposed_non_obligations` | 7 | `not_obligations` of `records_verified_t2.json` |
| `proposal_provenance` | — | source paths and sha256, staging sha256, counts, adoption boundary |

`quarantines`, `counts`, `status` and `adoption_scope` are untouched. So every consumer of the adopted rows
sees exactly what it saw before. The staging bytes up to the closing brace are unchanged: the diff is one
changed line (a comma after `reconciled_utc`) and 15,962 added lines. **Adopting** a proposed row means moving
it into `quarantines` with `proposal_status` ADOPTED. Adopting a pinning amendment means adding its
`source_binding` to the target record. Both are composer decisions.

Result after `git am` on `32ab7d90`: `cbz/quarantine/catalog.json` blob `4700eec52b72b122aff2170d5c0a39e575e50f36`,
sha256 `0f956acec6d30519bb80bd758c3b6828b814a584150c2a08c12f999c64dc01db`, 4,935,909 bytes; tree
`8db9915a600edcf013c49c61ea210b431d861c61`.

## Reproduce

From a checkout of `main` with this branch's `recovery/` directory, and a Math- clone holding commits `8151f592`
and `19a55750`:

```sh
R=recovery/session_013tN78b3Q38WdttFyiJ6KVg
git show 32ab7d9013264d2874aa4b778a90eede0cda0cdf:cbz/quarantine/catalog.json > /tmp/staging.json
python3 -I -B $R/qp_c54_catalog_candidate_20261102/build_candidate.py /tmp/staging.json \
  $R/qp_c54_quarantine_proposals_20261101/records_verified.json $R/qp_c54_t2_20261102/records_verified_t2.json /tmp/cand.json
sha256sum /tmp/cand.json        # 0f956ace...
python3 -I -B $R/qp_c54_catalog_candidate_20261102/validate_candidate.py /tmp/staging.json /tmp/cand.json \
  $R/qp_c54_quarantine_proposals_20261101/records_verified.json $R/qp_c54_t2_20261102/records_verified_t2.json MATH_CLONE
python3 -I -B $R/qp_c54_catalog_candidate_20261102/mutate.py /tmp/staging.json /tmp/cand.json \
  $R/qp_c54_quarantine_proposals_20261101/records_verified.json $R/qp_c54_t2_20261102/records_verified_t2.json MATH_CLONE
```

Rebuilding from the retained files gives the same sha256. `git am` of the patch on a fresh `32ab7d90` worktree
gives a byte-identical file.

## Checks (`validate_output.txt`: 10/10 PASS)

1. Additive only: the staging bytes are a prefix, every staging key and value is unchanged, and the
   serialization matches staging's (indent 2, `ensure_ascii` false, trailing newline).
2. The proposals equal the verified tranche files; their digests and the counts agree.
3. Every proposal carries, non-empty, the 13 keys that all 307 adopted rows carry.
4. Every proposal has `status` QUARANTINED, `proposal_status` PROPOSED, `owner` none and `maintenance` zero, and a
   `floating_point` field classified USED, NONE or NO_SCRIPTS.
5. IDs are unique across all 405 rows, and every proposal ID is in the `QP-C54-` range.
6. 1,240 file bindings: blob, bytes and sha256 read from Math- at each binding's commit.
7. 354 quoted statements are verbatim at their lines.
8. Every proposal location exists in Math- at its commit.
9. Each pinning amendment targets an adopted record that names the object and is in its `mentioned_by` list.
10. No new proposal sits at or under an adopted Math- location, so there is no double disposition.

**Negative controls** (`mutate_output.txt`): 16 of 16 mutants fail on the intended check. They cover a changed
or removed staging value, an edited or dropped proposal, a wrong count, an emptied required field, an owner, an
unclassified floating-point field, a duplicate ID, a changed binding sha256 or byte count, an altered or shifted
statement, a missing location, a wrong pinning target, and a proposal placed under an adopted location.

## Drift found in staging (status truth, A7)

- **`cbz/quarantine/README.md` line 3** says "The 184 listed obligations are terminal QUARANTINED". The
  catalog's `counts` has `adopted_quarantines: 307` and `original_quarantine_proposals: 184`. `BOUNDARIES.md`
  line 3 says "The 307 rows".
- **`cbz/quarantine/VALIDATION.json`** has `proposal_rows: 184`, so its checks describe the original 184 rows only.
- Both files last changed at `3ccd631c` (29 Oct 18:45Z). The reconciliation that added 123 rows came later, and
  `BOUNDARIES.md` was updated at `db14616b`.
- `check_adopted_bindings.py` (`adopted_bindings_output.txt`) runs a narrower version of those checks over all
  307 adopted rows:
  - required fields are non-empty;
  - IDs are unique;
  - every row has owner none and maintenance zero;
  - **190 unique complete `source_binding(s)` entries match Math- or main bytes, with 0 problems.**
  - It does **not** check 115 rows that bind only through other fields, or 95 binding entries with other key shapes.
- I have not edited either file. Correcting the README and re-issuing VALIDATION.json belong to whoever composes staging.

## Limits

- All drafters and verifiers behind these proposals are Anthropic Claude. Different-family checks:
  - tranche 1: GROK-H04 (main#738/6565322273);
  - tranche 2: offered to GROK-H02 (main#738/6566781875).
  - Neither result is recorded here.
- This is engineering accounting. No row is a verdict on any mathematics, and nothing is promoted to CLOSED.
- The tranche-3 classification of the 102 Math- `reviews/` directories, and the A3 audit of QP-C54-T2-008, -016
  and -029, were still running when this was built. Their output is not in this candidate.
