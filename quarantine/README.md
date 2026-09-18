# Quarantine and triage (non-authoritative)

Git-side equivalent of `90_QUARANTINE_AND_TRIAGE — NONAUTHORITATIVE` and the
package-local quarantine containers linked from the R17 Quarantine Index.

Nothing in scope here is evidence. Classes, from OP-PROT-019 §6:

| Class | Evidence required | Action |
|---|---|---|
| `EXACT_DUPLICATE` | matching raw digest/bytes, or declared native-body equivalence with format limits | choose a keeper, preserve IDs, move the surplus copy, record keeper and rollback |
| `SUPERSEDED` | explicit numbered successor or source-backed retirement | history/archive, **not** a claim of mathematical falsity |
| `DEFECTIVE_SCOPE` | concrete failed statement, counterexample, or reproducible invalid certificate chain | exclude the affected claim; preserve valid unrelated content; link repair and re-review |
| `UNVERIFIED / CONFLICT` | missing identity, unresolved custody, conflicting heads | isolate from active consumption pending resolution |
| `LEGACY_INSPIRATION` | existing legacy classification | inspiration only until rederived and reviewed |

`EXISTING_CONTAINER` also appears in the register for three pointers to
pre-existing package-local quarantine directories. OP-PROT-019 §6 does not
define it; it is recorded in `registers/KNOWN_FINDINGS.json` with a proposal to
add `CONTAINER_POINTER` to the protocol table.

## Logical quarantine

`EXCLUSIONS.json` holds the 17 current exclusions, keyed the way the protocol
requires for artifacts that cannot be moved independently: **carrier ID +
relative member path + payload SHA-256**. Eleven of them are the frozen H5
archive members (`Q-R17-H5-01..11`) held pending the full corrected-kernel
replay. Their bytes remain intact; what is excluded is the certification claim.

`tools/quarantine_check.py`, run in CI, asserts that:

1. the exclusion list and the register agree, both ways, on keys and classes;
2. every archive-member exclusion resolves to a real member of a real carrier
   with a matching payload digest;
3. **no excluded payload digest appears in any manifest in this repository** —
   nothing under quarantine has been silently pulled into verified content;
4. every exclusion carries a restoration test.

## What is not here

The Drive's `99_DO_NOT_OPEN` vault is **not** mirrored. Only its metadata
appears, in `drive/inventory.jsonl`. The standing order is that models must not
open it for authority, proofs, certificates or "latest" status unless the
operator names a vault ID for forensic recovery.

Every move in git is a commit, so every move has a rollback record. Nothing is
deleted.
