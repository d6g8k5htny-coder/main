# QP-C54 catalog candidate v2: every tranche beside the adopted rows (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE.

This is a ready-to-apply candidate `cbz/quarantine/catalog.json` for the composer or integrator of
`cbz/foundation-catalog-reconciled-20261029` (staging) and #695. It carries **all** of Claude 54's retroactive mathematical
accounting **beside** the 307 adopted rows, not inside them. It supersedes candidate v1 (`../qp_c54_catalog_candidate_20261102`),
which carried tranches 1 and 2 only. Nothing is pushed to staging or #695, and nothing is adopted.

## What the patch does

`0001-qp-c54-catalog-candidate-v2.patch` is a `git am` patch against staging `32ab7d9013264d2874aa4b778a90eede0cda0cdf`. It
changes one file, `cbz/quarantine/catalog.json`. The only changed line is `reconciled_utc`, which gains a comma; 123,805 lines
are added after it (git counts 123,806 insertions, including the changed line), as nine new keys:

| Key | Rows | Source in `recovery/session_013tN78b3Q38WdttFyiJ6KVg/` |
|---|---|---|
| `proposed_quarantines` | **239** main and Math- new records (tranche 1 48, tranche 2 44, tranche 3 41, tranche 4 85, tranche 5 3, tranche 8 18) | the tranche directories `qp_c54_quarantine_proposals_20261101` (T1), `qp_c54_t2_20261102`, `qp_c54_t3_20261102`, `qp_c54_t4_main_20261102`, `qp_c54_t5_lean_20261102`, `qp_c54_t8_math_residual_20261102` |
| `proposed_quarantines_federation` | 8 records from repositories outside the catalog's mathematical scope | `qp_c54_t6_federation_20261102` |
| `proposed_pinning_amendments` | **29** rows that bind their bytes into adopted records which already hold their claims (tranche 1: 4 to Q-3008; tranche 2: Q-1043, Q-1025; tranche 4: 18 to Q-1160–Q-1168, one each to Q-1169 and Q-10085; tranche 5: Q-1146; tranche 8: Q-1041, Q-1032) | tranches 1, 2, 4, 5 and 8, after the errata |
| `proposed_byte_pins` | 41 pins for 14 adopted records. 38 carry `subsumed_by`: an amendment binds the same record, path and blob, so adopting it makes the pin redundant | `pinning_sweep_20261102/pins.json` |
| `proposed_closed_candidates` | 29: the C291 group's own-receipt candidate and the 28 P15 hosted-gate declarations landed by #740 | tranche 5, `p15_closed_candidates_20261102` |
| `held_by_other_lane` | 1: `source_derivative_gaussian`, now merged as Math-#514 | tranche 5 |
| `a3_advice` | the final A3 verdicts for QP-C54-T2-008, -016 and -029, the boundary and the executed-replay note; the audits are referenced by path and SHA-256 | `qp_c54_t3_20261102/a3_audits.json`, errata |
| `accounting_dispositions` | **515** objects, items or archive objects that need no record, each with its cited reason (tranche 2 7, tranche 3 61, tranche 4 153, tranche 5 1, tranche 6 166, tranche 7 59, tranche 8 68) | every tranche, the T4 scope addendum and T7's mechanical copies (including 200 trial members that are copies of T6 bindings) |
| `proposal_provenance` | — | 14 required sources with SHA-256, the staging SHA-256, counts, the adoption boundary and the independence record |

`quarantines`, `counts`, `status` and `adoption_scope` are untouched, so every consumer of the adopted rows sees exactly what it
saw before.

**Two overlays are applied by the builder.** The verified tranche files are never edited.
- **Errata** (`../qp_c54_errata_20261102/errata.json`, 30 items).
  - The six `E-*` items come from different-family checks.
  - The 24 `E-R-*` items come from the two same-family review rounds below:
    - QP-C54-015..018, T5-003 and T4-005 become pinning amendments to the adopted records that hold their claims (Q-3008, Q-1146
      and Q-10085). Binding the bytes puts every statement in them under the target's isolation.
    - T4-025 becomes a new record.
    - T6-005's reason, blocking reason and resumption cost are corrected: its identity is CLOSED as K-SPEC-075 with K-SPEC-045,
      through an interface.
    - Thirteen rows get `claim_overlaps`: T4-010/012/015/018/020/021/023, T2-008/029, T5-001 and T6-005/007/008.
  - Every touched row lists the items in `errata_applied`.
- **`claim_overlaps` relations** (as `proposal_provenance.adoption_boundary` defines them):
  - HOLDS_TRANSCRIBED_PART, HOLDS_SAME_IDENTITY, HOLDS_SAME_CLAIM_SUMMARY and PRIMARY_FOR_TRANSCRIBED_PART name another record
    that holds part of the row's claim, so the row is narrowed on adoption;
  - TRANSCRIBED_BY marks the row itself as primary;
  - CLOSED_DECLARATION names a CLOSED kernel declaration that proves part of the claim, possibly through an interface;
  - INSIDE_ADOPTED_ENGINEERING_REGION is a location note only (Q-9001 in trial);
  - SAME_CLAIM_AS_PROPOSAL (tranche 8) asks the composer to merge two proposals;
  - ADOPTION_NOTE is a note to the composer.
- **Reconciliation** (`../qp_c54_t4_main_20261102/reconciliation_t6.json`): QP-C54-T6-007 shares 7 Git blobs with main.
  - It gets `reconciliation`, which names QP-C54-T4-004 as primary.
  - T4-004 gets `federation_replicas`.
  - T6-007 stays only for trial's `04_RETURN_TO_KIMI.md`.

**Adopting** a row is a composer decision:
- a proposed quarantine moves into `quarantines` with `proposal_status` ADOPTED;
- a pinning amendment or byte pin adds its binding to the target record;
- a row with `claim_overlaps` is narrowed or merged as its relations and notes say;
- a closed candidate needs the kernel-index lane's receipt;
- federation rows need the catalog's mathematical scope widened first.

**Result after `git am` on `32ab7d90`:**
- `cbz/quarantine/catalog.json` is blob `6af494bbaf90b7fc0ee8305183485f75cbda82df`, 11,797,565 bytes, SHA-256 `f689bc945297337f9ef49e496cbd145f5f76a39c5cf463bd399dead3aa0174a9`;
- the tree is `3f675992368e565f3d5c17002a525f70dd64461c`.

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
sha256sum /tmp/cand.json        # f689bc94...
python3 -I -B $C/validate_candidate_v2.py /tmp/staging.json /tmp/cand.json $R CLONES.json
python3 -I -B $C/mutate_v2.py /tmp/staging.json /tmp/cand.json $R CLONES.json
```

## Checks (`validate_v2_output.txt`: 13/13 PASS)

1. **Additive only.** The staging bytes are a prefix, every staging key and value is unchanged, exactly the nine keys are
   appended, and the serialization matches staging's (indent 2, `ensure_ascii` false, trailing newline).
2. **A fresh build from all 14 required sources equals the candidate.** The source list equals the builder's, none is missing,
   and every digest and count agrees.
3. **Required keys.** 276 proposals (quarantines, federation rows and amendments) carry the 13 keys that every adopted row
   carries, non-empty, with QUARANTINED, PROPOSED, owner none and maintenance zero, and a floating-point field that begins USED,
   NONE or NO_SCRIPTS.
4. **IDs** are unique across 652 rows: the adopted rows, the proposals, the byte pins and the 28 P15 closed candidates. The C291
   own-receipt candidate and the held group are keyed by location and carry no id.
5. **10,037 file bindings** have blob, bytes and SHA-256 read from the main, Math- and federation clones. This includes
   `context_files`, `identical_copies` and the blob ids of the four tranche-4 mechanical resolutions.
6. **926 quoted statements** are verbatim at their lines; archive-member statements are read from the member.
7. **Locations.** Every proposal location exists at its commit, archive members included.
8. **Pins and amendments.** All 41 byte pins match their bytes. Each pin's target is an adopted record that names the file,
   either by its full path in a mention that is not negated (as in "NOT in"), or by the bare name in its title together with its
   directory. Every `subsumed_by` names an amendment that binds the same record, path and blob. All 29 pinning amendments target
   adopted records, in either target form.
9. **P15.** Each of the 28 P15 declarations, fully qualified, is the theorem at its line, and the line lies inside the bound file.
10. **No proposal sits at or under any of 246 adopted locations.** Every location form staging uses is read: strings, GitHub URLs,
    dicts and the two 'main PR' prose rows. A federation row may sit inside an adopted engineering region only when it declares
    it; main and Math- rows may not. Claim-level overlaps at different locations are not detected here; they are recorded as
    amendments or `claim_overlaps`.
11. **Reconciliation.** The 7 reconciled federation files are byte-identical to their primary main bindings. Shared and
    federation-only files partition the row's binding, disjointly, federation-only blobs match, and the back-link is present.
12. **1,705 archive-member bindings** match the bytes inside their bound archives. A plain `.gz` member must be named after the
    archive.
13. **30 errata items** each set their value on exactly their target rows: a section target's tranche and object are checked, every
    `errata_applied` tag names an item, and a3 notes sit under audited ids.

**Negative controls** (`mutate_v2_output.txt`): *the 39-mutant run on this build is in progress at this commit; its output and
tally follow in the next commit.* (On the interim build without tranche 8, 31 of 31 earlier mutants were killed.) Every mutant also fails check 2,
because any change makes the candidate differ from a fresh build; check 2 is itself the intended check of mutants 2-4. The
intended-check failure is the one that matters.

## Same-family adversarial reviews

Two rounds of review ran over this candidate. Each round had four lenses, and every finding went to an independent refuter
that defaulted to "not real". Only reproduced findings were applied.

**Round 1** (workflow `wf_1e2e73e5-c9f`, over the earlier build `75ee024f…`): builder fidelity, validator gaps, the tranche-4
assembly, and catalog-level double dispositions.
- **Claim-level double dispositions** (two major, one minor): QP-C54-015..018 against Q-3008, T5-003 against Q-1146, and T4-005
  against Q-10085. All are now pinning amendments.
- **A false pinning amendment** (major): T4-025's only match was in Q-2012's `nonclaims` quotation. It is now a new record.
- **Partial overlaps:** T4-012/018/020/021 and T6-005, against T2-008/029 and adopted records. They are recorded as `claim_overlaps`.
- **An `ls-tree` quoting defect** (no `-z`) dropped two main files. They are now in T4's scope addendum; the federation repositories
  have no affected paths.
- **Tooling:**
  - all sources are required;
  - errata targets resolve strictly;
  - check 2 verifies the source list;
  - check 5 reads `identical_copies` and the tranche-4 mechanical resolutions;
  - check 8 has a stricter pin rule;
  - check 9 compares fully qualified names;
  - check 10 reads every adopted location form;
  - check 11 requires a disjoint partition;
  - new checks 12 (archive members) and 13 (errata).
- **Wording fixes:** `a3_advice` holds verdicts; two nits were not taken; the "not yet checked" list; the C291 id; the added-line
  count; the T4 Drive-id and referent counts; Q-9001.
- **Refuted as a defect but taken up:** the pattern census of Math- left directories uncensused. This is a stated limit, not a
  false claim, but a directory census with `ls-tree -z` found 37 such objects (198 files), which became tranche 8.
- **Refuted, no change:**
  - the PIN-038 / T4-037 overlap, which T4-037's reason documents;
  - checks 6, 7 and 8 claim no more than their README says;
  - T5-004 binds only Lean files and excludes the CLOSED_SOURCE modules that Q-3006 describes.

**Round 2** (workflow `wf_26ea8588-f37`, over an interim build): the `E-R-*` errata, the new tooling, tranche 7 and the corrected
READMEs.
- **A wrong erratum** (round 1's T6-005 text said the D₂ identity was not CLOSED). It is CLOSED as K-SPEC-075 with K-SPEC-045
  (K-MAIN-024 on main), for every ConeMomentInterface inhabitant; no inhabitant is constructed and alignment is pending. Corrected,
  and the CLOSED entries are added to T6-005 and T4-012.
- **Missed overlaps:**
  - T5-001 against Q-3007 (major; kept as a new record, because Q-3007 is a README summary);
  - T4-023 against Q-1014 and Q-10013;
  - T4-010 and T4-015 against Q-1011, Q-1048 and Q-1076.
- **Corrected citations** in the T4-018 and T4-020 notes, and a completed T4-021 adoption note.
- **Tooling:**
  - T6 rows are routed like every tranche;
  - check 10's engineering-region exemption applies to federation rows only;
  - check 8 tests both target forms, rejects negated mentions and verifies `subsumed_by`;
  - check 9 bounds line numbers;
  - check 12 checks `.gz` member names;
  - check 13 checks section-target identity, unknown tags and a3 note keys;
  - the adoption boundary defines each overlap relation;
  - 38 pins are marked `subsumed_by`.
- **Tranche 7:** 200 trial members are byte copies of T6 bindings, and seven of them had a different class. They now resolve to the
  T6 rows. The fidelity pass is retained, the privacy list is complete, and the copy wording is corrected.
- **README fixes:** the claim_overlaps semantics, the severity count, the mutant sentence, the scope-addendum reading methods, and
  the T4 README's reference to the tranche-8 census.
- **Refuted, no change:**
  - amendments "losing" claims that the target holds only at summary level (QP-C54-016..018, T4-005, T5-003): the bytes are bound
    under the target's isolation, so nothing leaves quarantine;
  - check 2 tests the retained sources it names;
  - check 11 and federation_replicas;
  - check 10's prose parser;
  - a second git bundle in the trial archive;
  - inventory evidence for clones without archives;
  - one more Drive id in a T4 reason field.

## Independence and limits

- All drafters and verifiers were Anthropic Claude. Different-family checks by Grok Bot agents on main#738 (comment ids in
  `proposal_provenance.independence`):
  - candidate v1: PASS MATCH;
  - tranches 2, 3 and 6: each PASS_WITH_NITS, MATCH, with the nits applied as errata or README fixes, except T3 nit 4 (kept) and
    the `**` markup in T6's quoted reasons (not taken);
  - the tranche-5 retip: PASS;
  - QP-C54-T2-029: an executed replay.
- **Not yet checked by another family:** tranche 1, tranche 5's classification, tranches 4, 7 and 8, the byte-pin sweep, the 28
  P15 closed candidates, the A3 verdicts for T2-008 and -016, the reconciliation, the scope addendum, the `E-R-*` errata and this
  candidate.
- Each tranche's `problems_fixed` lists the final verifier stage's problems. The earlier fidelity passes of tranches 4, 7 and 8 are
  kept in each tranche directory.
- This is engineering accounting. No row is a verdict on any mathematics, and every classification is open to challenge.
- The staging drift that v1 found (`cbz/quarantine/README.md` and `VALIDATION.json` still describe 184 rows, against 307
  adopted) is unchanged at `32ab7d90`. See `../qp_c54_catalog_candidate_20261102/README.md`.
