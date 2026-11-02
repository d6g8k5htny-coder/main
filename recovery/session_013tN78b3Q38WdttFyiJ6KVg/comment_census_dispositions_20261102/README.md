# Comment-census artifacts against CBZ dispositions (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. This is a
mechanical cross-reference. It makes no disposition, and it is not a verdict on any mathematics.

## Question

Claude S's census (main#745 at `683fdfa2`, `cbz/evidence/comment-math-census-20261031/CENSUS.json`, sha256
`9046e7a4…`) lists 942 mathematical artifacts that were delivered in comments. Its `records` field records which
files *mention* each artifact. Which of the 942 actually have a CBZ disposition?

## Inputs

| Input | Identity |
|---|---|
| Census | `CENSUS.json` at main `683fdfa225d3bec536d5756cbb1f2701f05c695b`, sha256 `9046e7a47bd45e226d2535d5f62810267e0e27375948d0c15ba2c6effea325e8` |
| Catalog | The QP-C54 candidate: staging `32ab7d90` plus the proposals, rebuilt by `../qp_c54_catalog_candidate_20261102/`, sha256 `0f956acec6d30519bb80bd758c3b6828b814a584150c2a08c12f999c64dc01db` |
| CLOSED list | `cbz/evidence/kernel/all-closed-declarations.json` at staging `32ab7d90`, sha256 `c25b7e46…`. It holds the same 367 declarations as #695's copy and gives identical rows. |
| Tranche-3 directories | `t3_review_dirs.json`: the 102 Math- `reviews/` directories that no record names |

## Method (`census_dispositions.py`)

Each committed carrier of an artifact is classified as follows; the first rule that matches wins:

1. **CLOSED_SOURCE:** a `.lean` file that hosts a CLOSED declaration.
2. **ADOPTED_QUARANTINE:** an adopted quarantine location is the carrier's path, an ancestor directory, or, as a
   weaker SIBLING match, another file in the same directory.
3. **PROPOSED_QP_C54:** the same test against the QP-C54 proposals.
4. **NOT_OBLIGATION_QP_C54.**
5. **COPY_OF_DISPOSED:** the carrier's blob equals a blob at a disposed location.
6. **NONE.**

Path rules apply only to carriers at the census cuts (Math- `19a55750`, main `4712fc13`). Branch carriers are
matched by blob only. An artifact takes the best result among its carriers.

Separately, each artifact whose comment ID or key appears in the catalog text is listed as a lead in
`mentioned_in_catalog_text`.

The output is deterministic: three runs under different `PYTHONHASHSEED` values give the same bytes.

```sh
python3 -I -B census_dispositions.py CENSUS.json CANDIDATE_CATALOG CLOSED_JSON MATH_CLONE MAIN_CLONE t3_review_dirs.json OUT.json
```

## Result (`comment_census_dispositions.json`, `run_output.txt`)

| Custody (census) | Artifacts | CLOSED_SOURCE | ADOPTED_QUARANTINE | PROPOSED_QP_C54 | NONE |
|---|---|---|---|---|---|
| Committed on main or Math- | 358 | 21 | 146 (32 exact, 104 ancestor, **10 sibling only**) | 151 (all ancestor) | 40 |
| Committed on an unmerged branch only | 180 | 0 | 0 | 0 | 180 |
| Drive or Dropbox only (cited) | 15 | 0 | 0 | 0 | 15 |
| Inline only | 389 | 0 | 0 | 0 | 389 |
| **Total** | **942** | **21** | **146** | **151** | **624** |

- **151 comment-delivered artifacts** have their committed home inside a QP-C54 proposal: tranche 1 or tranche 2.
  Adopting those proposals would give each of them a disposition.
- **CLOSED_SOURCE is file-level.** The carrier hosts at least one CLOSED declaration. This does not show that the
  artifact's statement *is* that declaration.

**The 50 committed artifacts with NONE or a sibling-only match** (`lane` field):

| Lane | NONE | Sibling only | What they are |
|---|---|---|---|
| T3_REVIEW_DIR | 16 | 0 | Math- `reviews/` directories that tranche 3 is classifying now |
| MAIN_SIDE | 19 | 2 | main: `docs/manuscript/side24_v3_supplement_draft_20261013/` (tex, CONCORDANCE, checks), `experiments/zeta_fold_exploration/`, `experiments/h3_band_v2/`, `audits/exact_sample_shrinking_bin/`, `reviews/grok_uploads_20260926/`, `experiments/universality/amplitude_mixture_falsifier.py` |
| LEAN | 4 | 6 | `companions/torus_filtration_20261011/*.lean`, `cap_first_exit_lean_20261002`, `asymmetric_elder_bridge_20261017`, `regular_superlevel_geometry_20261020`, `specifications/lean_statements_20261010/Specifications/RN.lean`: for the kernel lanes |
| MATH_REVIEW_PARTLY_NAMED | 1 | 1 | `reviews/p15_overlap_tail_claude_20260928`, `reviews/collision_mechanism_20260925` |
| OTHER | 0 | 1 | `frontiers/determinant_moment_20261025/SOURCE_GAUSSIAN_MOMENTS.md` (Q-3001 names `SCOPE.md` beside it) |

**Leads (16 artifacts with NONE whose comment ID appears in catalog text).** Almost all are quoted inside QP-C54
records, for example QP-C54-052 for `reviews/iba2_012_finite_radius`. Those comments are the delivery of a packet
that now has a proposal. Whether the comment's content equals the packet's is a judgement left to the adopter.

## Limits and open questions

- This uses the census's custody judgements as given. Like the census, it cannot prove absence.
- For **584 artifacts that are not committed** on main or Math- at the cuts, no location-based disposition exists.
  - The catalog does bind off-cut material. Q-8336 and Q-8340 bind PR heads with archive tags, and many rows use
    blob URLs at older commits.
  - So the 180 branch-only artifacts could take the same form. Two of their refs are now also held at `c8df4e14`.
  - For the 389 inline-only and 15 Drive-only artifacts, a record would need a comment or Drive location. The
    census records each comment's REST path and body sha256.
  - Whether the catalog admits such a location is the composer's decision.
- The main-side lane is outside my tranches 1 and 2, which covered Math- only.
