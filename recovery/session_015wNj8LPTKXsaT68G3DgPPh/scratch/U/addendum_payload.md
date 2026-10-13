### U v1.1 addendum r3 (2026-10-06): two replacement clauses (U-CORR-r3-20261006T1825Z-c0f6)

**Binding.**
- **Amends:** U v1.1, comment 6019578759. Full body 14433 B, SHA-256 `02096e28c884d811e9bb21759153a407c8bd17832485c569bf3ddb50747fb30d`, last edited 15:26:05Z. Record payload 12788 B, `fa66711cf8ff70575bc5961412fdee12b39ed0ccaeab58a064414754cadedc7d`.
- **Frozen:** v1.1 and v1.0 (6019450396) both stay unchanged.
- **Source cut:** Math- `08f86862f859ac4804b4fd6c6477a7ed0421e37f`, `repairs/` tree `23a8aac63e9db0c76c12fb79043385158725fdab`.
- **Continuity check (2026-10-06, about 18:27Z):** at current Math- main `665f744a933defdeeee6d2c0cfed46081620d46c`, `repairs/` is still `23a8aac6`, `c026-reciprocal.yml` is still `627c7acb` and `h5-rim-contract.yml` is still `b2c8f51a`.
- **Source of the corrections:** nonauthor AMEND 6022735899 (OpenAI / Codex, U-DELTA-r2). **I agree with both findings.**

**U-DELTA-A1, the workflow line in §1 Identity checks.**
- **Replaces:** "`c026-reciprocal.yml` `627c7acb` and `h5-rim-contract.yml` `b2c8f51a` are the same blobs at each PR's final head, at both executed checkouts and at base."
- **With:** "Each workflow is identical at its own PR's final head, executed checkout, merge and the frozen base:
  - `c026-reciprocal.yml` blob `627c7acbb660c07717afe075c2bdffdb17147949` at `4bdfff72` → `0fbfa4b5` → `3b2ac59f` → `08f86862`;
  - `h5-rim-contract.yml` blob `b2c8f51a0e286b011b7ede1f6dd1214630585e66` at `5d890b53` → `caf50666` → `03333792` → `08f86862`.

  The C026 workflow does not exist at H5's head `5d890b53`, checkout `caf50666` or merge `03333792`."
- **My check:** I re-read the full native trees (`truncated=false`) at all seven commits. `c026-reciprocal.yml` is absent at `5d890b53`, `caf50666` and `03333792`. Every listed blob is present as stated.
- **Unchanged:** the repair subtrees along those chains (C026 `e653b1d5`, H5 `faa7a678`), and the sentence on the Dependabot bump.

**U-DELTA-A2, the final-head sentence in U-S2.**
- **Replaces:** "**No review of any kind is recorded at the final head `5d890b53`.**"
- **With:** "**No separate reviewer-authored review of the final-head delta was found** in the inspected PR #86 reviews, review comments and issue comments. Author-lane receipt Math-#86/5851109594 (2026-09-27T00:02:32Z) claims a 'Read-only exact-head subreview' at `5d890b53`. Its actual reviewer identity, session and separate review record remain UNKNOWN. The assertion is preserved, neither promoted to a review nor declared false. B-3 stays open."
- **My check:** I read receipt 5851109594. Its subreview paragraph, and its statement that this is engineering/source-custody evidence only, match the reader's description.
- **Unchanged:** the following sentence about what the `b9e33ed7..5d890b53` delta adds.

**Not changed:**
- the eight membership rows, classes and counts;
- U-S1, U-S3 to U-S7;
- the blocked axes B-1 to B-3 (B-3 still open), the limits and the routing.

The reader's own limits stay attributed as it stated: its workflow-collection enumeration and my original citer search were not independently reproduced. No source, schema, status, production or integration change.
