# `attestations/` — architectural admission records

**An attestation is not a review, not a grade, not a promotion and not an
independence credit.** It records one thing: that a named object cleared a
named, machine-checkable *architectural* predicate at a stated commit.

## Why this exists

The program's throughput problem is not that work is unfinished. It is that
finished work has nowhere to be *seen* waiting. A result can be complete,
ported, hashed and structurally sound, and still be indistinguishable — to any
automated reader — from a result that was never started. The Review Queue
answers "who must look at this next"; nothing answered "which objects are
already structurally admissible and are now purely waiting on people".

That question is answerable without moving a single status, and this directory
answers it.

## What admission means, exactly

`ARCH-ADMISSION-V1` passes when every structural gate the repository runs over
the object's lineage exits zero: the claim-graph firewalls accept it, logical
quarantine is enforced, every manifest matches its SHA-256 and byte count, the
register-consumer map is free of drift, no lane reads stronger than the claim
graph, frozen digests agree, and the object's own replay driver reproduces its
stated quantities.

**Passing means the object is structurally admissible here. It does not mean
the mathematics is correct.** A wrong argument that is correctly filed,
correctly hashed and correctly scoped is `ADMITTED`, and that is deliberate:
the predicate is about structure, and structure is all it can see. Anyone
reading an `ADMITTED` record as mathematical endorsement has misread it.

## The four properties that keep this from becoming a promotion

1. **`claim_grade_after` is pinned to `UNCHANGED`** by the schema, and
   `tools/attestations_check.py` refuses any other value. An attestation cannot
   raise, lower or create a grade.
2. **`gate_status_after` is pinned to `UNCHANGED`.** No gate, premise or
   obligation moves.
3. **`independence_credit` is pinned to `0`.** An attestation is not a review
   and cannot carry organizational independence. R17 §4 is untouched.
4. **`awaiting` cannot be empty on an `ADMITTED` record.** Admission never
   finishes anything, so a record claiming nothing remained would be false by
   construction. The checker refuses it.

`tests/test_attestations.py` carries a negative control for each, and for the
fail-closed rule below.

## Fail-closed

A single non-zero exit code among the recorded checks **forbids** `ADMITTED`.
A record claiming admission while carrying a failed gate is a violation, not a
judgement call. The converse also holds: a `REFUSED` record must name the check
that refused it, so a refusal cannot be asserted without evidence either.

Exit codes are transcribed from real runs. A check nobody ran has no business
in the list.

## What this directory does not do

* **It does not promote.** Nothing here is a licensing predicate, and CLAUDE.md
  rule 1 is unaffected: status labels are still transcribed from the source
  registers and only an operator moves a gate.
* **It does not review.** It supplies no reviewer, of any provider, and no
  record here counts toward any independence-requiring predicate.
* **It does not verify mathematics.** See above, twice.
* **It does not schedule.** Nothing consumes these records; this form is NOT
  DEPLOYED as an enforcement mechanism. It is a queryable record, and a
  queryable record is a record.
* **It does not expire.** Age never converts a `REFUSED` into an `ADMITTED`,
  and never converts an `ADMITTED` into anything further.

## Run it

```bash
python3 tools/attestations_check.py
python3 tools/attestations_check.py --records DIR --schema PATH
python3 -m pytest -q tests/test_attestations.py
```

Exit status is non-zero on any violation.
