# `reviews/` — nonauthor technical review records

**Every record in this directory carries ZERO organizational independence credit.**

The sessions writing them are Anthropic-family. OP-PROT-019-v1.1 (R17) §4 states the
consequence directly: *"Same provider is **zero organizational independence**, not a
prohibition on useful technical review."* So both halves hold at once, and neither
softens the other:

* R17 §4 **permits** these reviews. A fresh nonauthor session of any provider may review
  an exact object, including the author's own provider, provided it discloses its source
  exposure and does substantive reconstruction, counterexample search or meaningful
  execution. That is what these records are.
* The independence-requiring gates **REMAIN OPEN** regardless of what any record here
  concludes. A `PASS_TECHNICAL` verdict in this directory does not satisfy, weaken,
  partially satisfy, or make progress against any predicate that requires an
  organizationally distinct reviewer. R17 §4: *"A task may finish its technical review
  while an external-independence predicate remains open. Never relabel an
  independence-required theorem terminal solely because its technical review passed."*

Every record therefore carries `independence_credit: 0`, a stated reason, and
`gate_status_after: "UNCHANGED"`. `tools/reviews_check.py` rejects any other value for
either, and `tests/test_reviews.py` proves it rejects them.

## What a record here is

A **verdict on an object**: these exact bytes, this digest, this byte count, this
extraction rule, reconstructed by someone who actually read them, with the hypotheses
named, the negative controls run and reported including the ones that fired, the findings
listed by criterion, and the remaining dependencies named as still unresolved.

If the object could not be obtained, the record says `CANNOT_VERIFY` and names exactly
what was unobtainable and what was tried. That is a legitimate and useful result, and the
checker enforces it: `obtained: false` cannot be paired with any verdict except
`CANNOT_VERIFY`. **A review of something nobody read is never written here.**

## What a record here is not

* **Not a gate movement.** Nothing in this directory promotes, closes, discharges or
  reclassifies any claim, premise or obligation. Nothing here edits a register.
* **Not independence.** See above. The credit is zero and the reason is written out.
* **Not peer review** in the sense any external venue means. R17 §4 and OP-PROT-012 §5
  both distinguish a technical pass from an independent-eyes predicate; only the first
  is available to this session.
* **Not a second credit to an existing lineage.** R17 §4: *"Existing independent verdicts
  retain their exact scope; do not award a second credit to the same lineage."*
* **Not mathematics, verified.** The checker verifies the form of a review. A record can
  conform perfectly and still be wrong about the object.
* **Not a solution to any original prize problem.** Zero remain solved.

## Aging approves nothing

Nineteen of the twenty-four Review Queue routes are 51–54 days old and sit at ESCALATE.
R17 §5 is explicit about what that means: escalation names a missing capability and
prioritises a qualified nonauthor reviewer. *"Never turn age into automatic approval,
invalidation, or extra independence."* An overdue route that finally receives a record
here is an overdue route with a technical verdict attached — its independence status is
exactly what it was the day before.

## Layout

```
reviews/
  README.md                   this file
  SCHEMA.md                   the required contents, field by field, and where
                              R17 §4 and the CONTRIBUTION_PLAN §5 summary differ
  review_record.schema.json   the normative schema the checker enforces
  records/                    one JSON record per review, named <review_id>.json
```

`records/` is empty until a review is genuinely performed. An empty directory is the
honest state; a directory of records for objects nobody opened would not be.

## Checks

```bash
python3 tools/reviews_check.py              # every record, against the real register
python3 -m pytest -q tests/test_reviews.py  # the rules and their negative controls
```

Non-zero exit on any violation. Read `SCHEMA.md` §4 for the full rule list, and §5 for
what these checks cannot do.
