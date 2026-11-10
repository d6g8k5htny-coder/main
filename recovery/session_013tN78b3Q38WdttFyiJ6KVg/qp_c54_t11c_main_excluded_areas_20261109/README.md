# QP-C54 tranche 11c: recovery/, drive/ and incoming/ on main refs (IN PROGRESS, interim retention of 10 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684 comment
6752703580. Checkpoint: 6754842631. Proposals only: nothing was pushed to any PR or branch, and no tag was created.

**This retention is interim.** It holds the finished stages, so that a container restart cannot lose them. The deep pass, the
assembled records and the final README follow in a later commit to this directory.

## Finished stages

1. **Census** (`excl_census.py` → `excl_census.json`).
   - Input: the census entries under `recovery/`, `drive/` and `incoming/` of T11b's main history census. Cut `d2b0216e`; the
     689 PR heads and 292 branch tips of 2 November.
   - Each blob is compared with every blob reachable from main at `1902aac1`, with every ref of the 11 other federation clones,
     with the census at non-excluded paths, and with the 40-hex ids in QP-C54 files and the catalog.
   - **8,725 blobs.** 8,497 occur nowhere else.
2. **Normalized dedup and supersession** (`norm_dedup.py` → `norm_dedup.json`). This drops:
   - 3 normalized copies (BOM, CRLF, trailing whitespace);
   - 24 blobs superseded in their lineage;
   - 44 normalized duplicates within the residual.

   **8,426 blobs (114 MB) remain.**
3. **Triage** (`t11c_batches.py` → `triage_inputs/`; workflow `t11c_triage_part.js`, 87 agents; `extract_triage.py` → `triage.json`;
   `t11c_objects.py` → `triage_ledger.json` and `objects_t11c.json`).
   - Every blob is labelled exactly once, which is checked mechanically: 1,784 MATH, 525 MATH_REVIEW, 4,054 WORKING and 2,063
     ENGINEERING.
   - The MATH and MATH_REVIEW files form 483 deep-pass objects. `t10_batches.py` packs them into `facts/`.
4. **Drive check, metadata only** (`workflow_drive_sample.js`; `drive_sample.json` → `drive_sample_result.json`).
   - A stratified sample of 116 of the 2,349 Drive ids in the mirror manifests. Snippets were excluded, and no owner or sharing
     data was recorded.
   - **116 of 116 resolve, and none is trashed.** All 57 byte-exact rows match their manifest size.
5. **Triage audit** (`mislabel_screen.py` → `mislabel_flags.json`; `audit_sample.py` → `audit/`; `workflow_triage_audit.js` →
   `audit_result.json`).
   - 385 WORKING/ENGINEERING labels were re-judged independently.
   - **Random 5% stratified sample (314 rows): 6 hid mathematics (1.9%).** Among the 71 rows flagged by the keyword screen alone,
     2 did.
   - So a second screen covers all 5,732 unaudited WORKING/ENGINEERING rows (`screen2.js`, inputs in `screen2_inputs/`).
     `t11c_supplement.py` turns every confirmed miss into a supplementary deep-pass object.

## Still running at this commit

- the deep pass (`workflow_t11c.js`, 59 batches);
- the second screen (48 batches).

Then come the supplement, normalization (`normalize_verdicts_t10.py`), line-range repair (`fix_line_ranges.py`), the fail-closed
assembler (`assemble_t11c.py`), reconciliation (`reconcile_t11c.py`) and the comment-census cross-reference
(`census_crossref.py`).
