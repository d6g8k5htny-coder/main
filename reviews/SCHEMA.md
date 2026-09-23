# The machine-checkable form of an R17 §4 technical review

`reviews/review_record.schema.json` is the normative shape. `tools/reviews_check.py`
enforces it plus the cross-record and cross-register rules a schema cannot express.
`tests/test_reviews.py` carries a negative control for every rule below. This file
explains what each requirement is for and where it comes from.

A record in this form is **a verdict on an object**. It is not a gate movement, not
a premise discharge, and not an independence credit. See `README.md` in this
directory before writing one.

---

## 1. What the protocol actually requires

OP-PROT-019-v1.1 (R17) §4, verbatim:

> Record four separate dimensions: exact object correctness verdict; scope/dependency
> verdict; reviewer authorship/exposure; organizational independence.

> A valid review records: source ID/hash/bytes and extraction, author and reviewer
> session/provider, exposure, precise hypotheses, reconstructed argument, actual
> execution and negative controls (if applicable), findings by criterion, unresolved
> dependencies, verdict and reproducible output. A source search or hash match alone
> is not mathematical review. No confidence voting.

> Technical statuses: READY, IN_REVIEW, PASS_TECHNICAL, AMEND, FAIL, CANNOT_VERIFY,
> or NEEDS_RECONCILIATION. Independent status is a separate field. A task may finish
> its technical review while an external-independence predicate remains open. Never
> relabel an independence-required theorem terminal solely because its technical
> review passed.

> A fresh nonauthor session of any provider may review an exact object, including the
> author's provider. It must disclose source exposure and perform substantive
> reconstruction, counterexample search or meaningful execution. Same provider is
> **zero organizational independence**, not a prohibition on useful technical review.
> Different provider alone does not establish independence if it coauthored the target
> or reused the same reasoning.

> Existing independent verdicts retain their exact scope; do not award a second credit
> to the same lineage.

R17 §5 adds the aging rules: request time is recorded separately from last substantive
review activity; a refresh or metadata touch does not reset age; and "Never turn age
into automatic approval, invalidation, or extra independence."

## 2. Where `docs/CONTRIBUTION_PLAN.md` §5 differs from the protocol

The plan summarises the required contents as: *source ID / hash / bytes and extraction
rule, exposure disclosure, precise hypotheses, reconstructed argument, executed negative
controls, findings by criterion, unresolved dependencies, verdict, reproducible output.*

The protocol's own wording governs. Five differences, each resolved in the protocol's
favour, and each visible in the schema:

| # | R17 §4 | CONTRIBUTION_PLAN §5 | Resolution in this schema |
|---|---|---|---|
| 1 | "author and reviewer session/provider" is a listed record content, and "reviewer authorship/exposure" is one of the four dimensions | omitted entirely | `reviewer_family`, `reviewer_session`, `author_family` and `author_family_determination` are all **required** |
| 2 | four separate dimensions, of which "exact object correctness verdict" and "scope/dependency verdict" are two | one item, "verdict" | `technical_verdict` (exact-object correctness, from the R17 status set) and `scope_dependency_verdict` (prose) are **separate required fields** |
| 3 | "actual execution **and** negative controls (if applicable)" | "executed negative controls" | `negative_controls_executed` is required; it may be empty **only** with a non-empty `negative_controls_not_applicable_reason`, which is what "(if applicable)" licenses and no more. `execution_performed` carries the separate "actual execution" item |
| 4 | "Independent status is a separate field"; same provider is zero; a different provider alone establishes nothing; no second credit to one lineage | not mentioned | `independence_credit` + `independence_reason` are required and separate from any verdict field; `independence_evidence` is required whenever the credit is nonzero |
| 5 | "No confidence voting." | not mentioned | the checker rejects confidence scores, percentages of belief and high/medium/low confidence labels in any free-text field |

Two further notes on fidelity:

* The plan writes "extraction **rule**" where R17 writes "extraction". That is the rest
  of the corpus's own name for the same thing (R17 §2 requires revision-aware extraction
  for native Docs and warns that `modifiedTime` is not a body digest), so the schema uses
  `extraction_rule` and means R17's "extraction".
* `gate_status_after` and `review_id` are **ours, not the protocol's**. R17 §4 states the
  prohibition in prose — a technical pass never relabels an independence-required theorem
  terminal. `gate_status_after` exists so that prohibition is a field a checker can fail
  on, and its only admissible value is `UNCHANGED`. Do not read it as a protocol-defined
  register column.

## 3. Fields

| Field | Type | Requirement |
|---|---|---|
| `schema_version` | string | exactly `"1.0.0"` |
| `review_id` | string | `REV-…`, unique in `reviews/records/`, and equal to the filename stem |
| `route_key` | string | a **Review key** present verbatim in `registers/json/review_queue.json` |
| `review_utc` | string | UTC instant of the substantive review activity (R17 §5's "last substantive review activity", never the file's modification date) |
| `object_id` | string | Drive file ID, carrier ID + relative member path, or repository path — an identity, never a title or folder |
| `object_title` | string | the title at the source; a title match is not an identity match (R17 §6) |
| `object_bytes` | integer | byte count of the exact reviewed body; `> 0` when `obtained` |
| `object_sha256` | string | lowercase 64-hex digest of those exact bytes; empty only when `obtained` is false |
| `extraction_rule` | string | export MIME type, revision id for native Docs, ZIP member path, decode step |
| `obtained` | boolean | true only if the reviewer actually read those bytes |
| `obtained_how` | string | the concrete route to the bytes — or, when false, exactly what was unobtainable and what was tried |
| `reviewer_family` | string | lowercase provider family of the reviewing session |
| `reviewer_session` | string | reviewer session identity |
| `author_family` | string | lowercase provider family of the object's author lineage, or `unknown` |
| `author_family_determination` | string | how that was established: register column, in-body attribution, Work Events row, or that it could not be |
| `exposure_disclosure` | string | ≥ 300 characters and ≥ 40 distinct tokens; what the reviewer had already seen of this object and its neighbours, and how that could bias the verdict |
| `hypotheses` | list | ≥ 1 precise hypothesis the object's statement is conditional on |
| `reconstruction` | string | ≥ 400 characters; the argument rebuilt in the reviewer's own words from the read bytes |
| `negative_controls_executed` | list | each `{control, would_have_caught, fired}`; empty only with a stated not-applicable reason |
| `execution_performed` | string | optional: the counterexample search or meaningful execution, and its result |
| `findings` | list | each `{criterion, severity ∈ INFO/MINOR/MAJOR/BLOCKING, statement}` |
| `unresolved_dependencies` | list | named premises, lemmas, drivers and heads still open after this review |
| `technical_verdict` | enum | `READY`, `IN_REVIEW`, `PASS_TECHNICAL`, `AMEND`, `FAIL`, `CANNOT_VERIFY`, `NEEDS_RECONCILIATION` — nothing else |
| `scope_dependency_verdict` | string | the separate scope/dependency dimension |
| `independence_credit` | integer | `0` or `1`; `0` for every record written by this Anthropic-family session |
| `independence_reason` | string | why that credit and no more, naming the licensing predicate |
| `independence_evidence` | string | required non-empty whenever the credit is nonzero: the documented OP-PROT-012 §5 (a)–(j) items |
| `gate_status_after` | string | exactly `UNCHANGED` |
| `does_not_establish` | string | ≥ 200 characters and ≥ 25 distinct tokens, specific to this object |
| `reproducible_output` | list | ≥ 1 command, script path or receipt reference |
| `sha_mismatch_explanation` | string | required non-empty when the digest disagrees with the register row's `Body SHA-256` |
| `notes` | string | optional free text, scanned like every other field |

The schema is **closed**: an unknown field is a violation, so no record can smuggle in a
`waiver`, a `confidence`, or an `approved_by`.

## 4. Rules the checker enforces beyond field shapes

1. **Independence.** `independence_credit` must be `0` when `reviewer_family` is
   `anthropic`, when reviewer and author share a family, or when the author lineage is
   `unknown`; a zero credit still requires a stated reason; a nonzero credit requires
   documented evidence. R17 §4: same provider is zero organizational independence.
2. **Gate.** `gate_status_after` must be `UNCHANGED` on every record, always.
3. **Vocabulary.** `technical_verdict` must be one of the register's own R17 statuses.
   The checker also verifies that `registers/json/review_queue.json` itself still uses
   only that vocabulary, so a new status in the export cannot be quietly inherited.
4. **No verdict without reading.** `obtained == false` forces `CANNOT_VERIFY`;
   `obtained == true` forces a 64-hex digest and a positive byte count. A CANNOT_VERIFY
   record naming exactly what was unobtainable is a legitimate, valuable result.
5. **No promotion language**, anywhere in any free-text field, case-insensitively:
   `discharged`, `closed the`, `promotes`, `premise is now`, `gate satisfied`,
   `independence satisfied`. The check is literal and it has no negation escape: write
   "the obligation remains OPEN" or "this does not discharge anything", not a sentence
   that reuses promotion wording under a negation. The offending field path is named.
6. **No confidence voting**, per R17 §4.
7. **Register membership.** Every `route_key` must exist in the exported Review Queue,
   and a digest disagreeing with that row's `Body SHA-256` must be explained.
8. **Anti-boilerplate.** `exposure_disclosure` and `does_not_establish` must clear a
   length floor, a distinct-token floor (so padding by repetition fails), and must not be
   verbatim identical to another record's. A disclosure copied between records is not a
   disclosure.
9. **Identity.** `review_id` is unique and matches the filename stem.

Run it:

```bash
python3 tools/reviews_check.py                 # reviews/records/ against the real register
python3 tools/reviews_check.py --records DIR --schema PATH --queue PATH
python3 -m pytest -q tests/test_reviews.py     # the rules plus their negative controls
```

Exit status is non-zero on any violation.

## 5. What this form does not do

* It does not verify any mathematics. A record can conform perfectly and be wrong.
* It does not award organizational independence, and it cannot: the credit is a number
  the record states about itself, and the checker's only power is to stop that number
  from exceeding what the predicate licenses.
* It does not write to any register. `registers/` is read-only to this directory; the
  `Independence status` and `Technical status` columns of a Review Queue row are
  untouched by anything here.
* It does not close, reopen, escalate or de-escalate a route. R17 §5's aging ladder
  decides pickup order and nothing else; age never becomes approval.
* It does not detect a reviewer who reconstructs the argument by reading a summary
  rather than the bytes. The digest proves which bytes exist, not which bytes were read.
  `exposure_disclosure` is where that honesty has to live, and no checker can supply it.
