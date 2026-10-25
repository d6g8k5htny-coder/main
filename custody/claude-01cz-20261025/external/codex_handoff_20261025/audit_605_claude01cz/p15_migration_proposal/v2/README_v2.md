# P15 migration proposal — v2 (amendment)

**Why v2:** the review at main#605 comment 6404282041 (OpenAI, nonauthor) returned AMEND_ENGINEERING. v1's successor guard checked migration ids, link presence and the aggregate count, but it was not bound to the original edge set. The reviewer's counterexamples CE1–CE4 would have passed it. The v1 files in the parent folder (`4d8775ff`) are kept unchanged as the historical candidate.

## Changes
- **`migration_guard.py`** (embedded byte-identically in `model_successor_v2_PROPOSAL.py.txt`; diff in `VALIDATOR_MIGRATION_v2.diff`):
  - *No migration entry:* the original rule applies unchanged (exactly 108 links).
  - *With a migration entry, all of the following must hold:*
    - exactly one entry;
    - every field equals the single authorized migration (`MIG-P15-T1-PREMISE-T2-20261025`, OBL:P15-T1 + P15-T2, 108 → 109);
    - its status is in {`PROPOSED_NOT_ADOPTED`, `ADOPTED_AFTER_REVIEW`};
    - the full sorted edge list has exactly 109 entries, and its sha256 equals `bdd2ea04…`. That digest is the original 108-edge set (sha256 `e1c88cde…`, at main `11f7b82e`) plus (P15-T1, P15-T2).

  This rules out removals, swaps, extra edges, duplicate credit and arbitrary counts or statuses. Every other `model.py` check, including stale-fingerprint rejection, is textually unchanged.
- **`build_successor_v2.py`:** pins the raw input sha256 values (original `4cfae309…`, patch `9779bbb8…`) and asserts that the closure equals exactly {P15-T1, P15-T2, P15-T3}. Rerunning it reproduces the v1 successor dataset exactly (canonical sha256 `3f69ae83…`).

## Tests (my own code only; `test_guard.py` → `test_guard.out`, 8/8 as expected)
- Accepted: the original dataset (no migration) and the successor proposal.
- Rejected:
  - CE1: an extra undocumented edge plus a duplicate-credit migration;
  - CE2: remove an old edge and add another, keeping the count at 109;
  - CE3: counts of 900 → 901;
  - CE4: an arbitrary status;
  - an undocumented 109th edge with no migration entry;
  - a migration entry whose edge is absent.

## Still not done (blocking adoption)
- Running the successor validator: neither the published `model.py` nor the successor file was executed.
- Nonauthor re-review of this amendment.
- An accepting editor.

The alignment PASS_SCOPED in 6404282041 covers the premise change only, and is not relabelled.

## v2.1 advisory follow-up (after review 605/6404432133)
- Advisory 3: `build_successor_v2.py` now parses the already-hashed `RAW_PATCH` bytes. It no longer opens the pathname a second time.
- Advisory 4: `test_guard.py` now exits with status 1 when any expectation fails.
- Rerun on pinned inputs (4cfae309…, 9779bbb8…): successor canonical sha256 `3f69ae83…` unchanged, 8/8 as expected, exit 0. Guard logic and data are unchanged.
- The status stays PROPOSED_NOT_ADOPTED. This is not adoption.
