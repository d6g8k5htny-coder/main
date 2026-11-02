# QP-C54 catalog candidate v2: every tranche beside the adopted rows (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE.

This is a ready-to-apply candidate `cbz/quarantine/catalog.json` for the composer or integrator of
`cbz/foundation-catalog-reconciled-20261029` (staging) and #695. It carries **all** of Claude 54's retroactive mathematical
accounting **beside** the 307 adopted rows, not inside them. It supersedes candidate v1 (`../qp_c54_catalog_candidate_20261102`),
which carried tranches 1 and 2 only. Nothing is pushed to staging or #695, and nothing is adopted.

## What the patch does

`0001-qp-c54-catalog-candidate-v2.patch` is a `git am` patch against staging `32ab7d9013264d2874aa4b778a90eede0cda0cdf`. It
changes one file, `cbz/quarantine/catalog.json`. The only changed line is `reconciled_utc`, which gains a comma; 104,435 lines
are added after it, as nine new keys:

| Key | Rows | Source in `recovery/session_013tN78b3Q38WdttFyiJ6KVg/` |
|---|---|---|
| `proposed_quarantines` | **226** main and Math- records: tranche 1 52, tranche 2 44, tranche 3 41, tranche 4 85, tranche 5 4 | `qp_c54_quarantine_proposals_20261101`, `qp_c54_t2_20261102`, `qp_c54_t3_20261102`, `qp_c54_t4_main_20261102`, `qp_c54_t5_lean_20261102` |
| `proposed_quarantines_federation` | 8 records from repositories outside the catalog's current scope | `qp_c54_t6_federation_20261102` |
| `proposed_pinning_amendments` | 22: tranche 2's 2 (Q-1043, Q-1025) and tranche 4's 20 (Q-1160–Q-1169, Q-2012) | tranches 2 and 4 |
| `proposed_byte_pins` | 41 pins for 14 adopted records | `pinning_sweep_20261102/pins.json` |
| `proposed_closed_candidates` | 29: the C291 group's own-receipt candidate and the 28 P15 hosted-gate declarations landed by #740 | tranche 5, `p15_closed_candidates_20261102` |
| `held_by_other_lane` | 1: `source_derivative_gaussian`, now merged as Math-#514 | tranche 5 |
| `a3_advice` | the interval-arithmetic audits of QP-C54-T2-008, -016 and -029, plus the executed-replay note | `qp_c54_t3_20261102/a3_audits.json`, errata |
| `accounting_dispositions` | **386** objects or items that need no record, each with its cited reason: tranche 2 7, tranche 3 61, tranche 4 151, tranche 5 1, tranche 6 166 | every tranche, and tranche 4's scope addendum |
| `proposal_provenance` | — | the source paths and SHA-256 values, the staging SHA-256, counts, the adoption boundary and the independence record |

`quarantines`, `counts`, `status` and `adoption_scope` are untouched, so every consumer of the adopted rows sees exactly what it
saw before.

**Two overlays are applied by the builder.** The verified tranche files are left as they were.
- **Errata** (`../qp_c54_errata_20261102/errata.json`): field corrections accepted from different-family checks.
  - `E-T2-N2` and `E-T2-N4` touch one row each, `E-T3-N2` touches 9 and `E-T3-N3` 1, all tagged with `errata_applied`.
  - `E-T5-HELD` updates the held group.
  - `E-A3-T2-029` becomes a note in `a3_advice`.
- **Reconciliation** (`../qp_c54_t4_main_20261102/reconciliation_t6.json`): QP-C54-T6-007 shares 7 Git blobs with main.
  - It gets `reconciliation`, which names QP-C54-T4-004 as primary.
  - T4-004 gets `federation_replicas`.
  - T6-007 stays only for trial's `04_RETURN_TO_KIMI.md`.

**Adopting** a row is a composer decision:
- a proposed quarantine moves into `quarantines` with `proposal_status` ADOPTED;
- a pinning amendment or byte pin adds its binding to the target record;
- a closed candidate needs the kernel-index lane's receipt;
- federation rows need the catalog's scope widened first.

**Result after `git am` on `32ab7d90`:**
- `cbz/quarantine/catalog.json` is blob `ea5e49aa2928168138f0029f81cc4076697841b0`, 10,365,331 bytes, SHA-256
  `75ee024f8d7107753b398738f69b9e62674c6a18b509177d9c3fd2d526d8733a`;
- the tree is `c5734c4b814742af5392ccf0c694dbbccf49d2d8`.

`git am` on a fresh `32ab7d90` worktree gives a file byte-identical to a fresh build.

## Reproduce

You need a checkout of `main` with this branch's `recovery/` directory, and clones of Math- and the eight federation
repositories at the commits the rows name. `clones.json` maps each repository name to its clone and is an example from this
session.

```sh
R=recovery/session_013tN78b3Q38WdttFyiJ6KVg
C=$R/qp_c54_catalog_candidate_v2_20261102
git show 32ab7d9013264d2874aa4b778a90eede0cda0cdf:cbz/quarantine/catalog.json > /tmp/staging.json
python3 -I -B $C/build_candidate_v2.py /tmp/staging.json $R /tmp/cand.json
sha256sum /tmp/cand.json        # 75ee024f...
python3 -I -B $C/validate_candidate_v2.py /tmp/staging.json /tmp/cand.json $R CLONES.json
python3 -I -B $C/mutate_v2.py /tmp/staging.json /tmp/cand.json $R CLONES.json
```

## Checks (`validate_v2_output.txt`: 11/11 PASS)

1. **Additive only.** The staging bytes are a prefix, every staging key and value is unchanged, exactly the nine keys are
   appended, and the serialization matches staging's (indent 2, `ensure_ascii` false, trailing newline).
2. **A fresh build from the retained sources equals the candidate.** Every source digest and every count agrees.
3. **Required keys.** 256 proposals (quarantines, federation rows and amendments) carry the 13 keys that every adopted row
   carries, non-empty. Each also has QUARANTINED, PROPOSED, owner none and maintenance zero, and a floating-point field that
   begins USED, NONE or NO_SCRIPTS.
4. **IDs** are unique across 632 rows: the adopted rows, the proposals, the byte pins and the closed candidates.
5. **9,766 file bindings** have blob, bytes and SHA-256 read from the main, Math- and federation clones, `context_files`
   included.
6. **865 quoted statements** are verbatim at their lines.
7. **Locations.** Every proposal location exists at its commit.
8. **Pins and amendments.** All 41 byte pins match their bytes, and each pin's target is an adopted record that names the
   file. All 22 pinning amendments target adopted records.
9. **P15.** Each of the 28 P15 declarations is the theorem at its line in the bound source.
10. **No double disposition.** No new main or Math- proposal sits at or under an adopted location.
11. **Reconciliation.** The 7 reconciled federation files are byte-identical to their primary main bindings. Shared and
    federation-only files partition the row's binding, and the back-link is present.

**Negative controls** (`mutate_v2_output.txt`): 22 mutants, each of which must fail on its intended check. *At this commit the
run is still in progress; its output and tally follow in the next commit.* They cover:
- a changed staging value, an edited proposal, a dropped accounting row and a wrong count;
- an emptied required field, an owner, an unclassified floating-point field and a duplicate ID;
- federation, Lean, accounting and tranche-4 bindings;
- altered or shifted tranche-3, tranche-4 and tranche-5 statements;
- a missing location, a changed pin and a wrong pin target;
- an unadopted amendment target, a shifted P15 line, a proposal under an adopted location and a changed reconciled blob.

## Independence and limits

- All drafters and verifiers were Anthropic Claude. Different-family checks by Grok Bot agents on main#738 (comment ids in
  `proposal_provenance.independence`):
  - candidate v1: PASS MATCH;
  - tranches 2, 3 and 6: each PASS_WITH_NITS, MATCH, with the nits applied as errata or README fixes;
  - the tranche-5 retip: PASS;
  - QP-C54-T2-029: an executed replay.
- **Not yet checked by another family:** tranche 1, tranche 5's classification, tranche 4, the byte-pin sweep and this candidate.
- This is engineering accounting. No row is a verdict on any mathematics, and every classification is open to challenge.
- The staging drift that v1 found (`cbz/quarantine/README.md` and `VALIDATION.json` still describe 184 rows, against 307
  adopted) is unchanged at `32ab7d90`. See `../qp_c54_catalog_candidate_20261102/README.md`.
