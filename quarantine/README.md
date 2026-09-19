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

`EXCLUSIONS.json` holds the 22 current exclusions (register export
2026-09-18), keyed the way the protocol requires for artifacts that cannot be
moved independently: **carrier ID + relative member path + payload SHA-256**.
Eleven of them are the frozen H5 archive members (`Q-R17-H5-01..11`) held
pending the full corrected-kernel replay. Five are the RN5 scope holds of
2026-09-17T17:19Z, all class `DEFECTIVE_SCOPE`:

| Key | Object | Excluded scope (the register's words) |
|---|---|---|
| `Q-RN5-MOMENT-001` | archive member `K3_SIDE24_LB/UPPER2D/D3_percolation/d3_perc.py` (`5bc09241…`, 40,337 B) | `envelope_v`; `window_cap_env` and consumers relying on the claimed polarity-safe upper bound. Other functions are outside this finding. |
| `Q-RN5-MOMENT-002` | archive member `…/d3_amend_v2.py` (`6399a0e3…`, 14,846 B) | `v3_annulus_numerator` and assembled near/remote numerical upper-bound claims that consume `E.window_cap_env`. H3 denominator repair and unaffected far components retained. |
| `Q-RN5-MOMENT-003` | archive member `…/D3_REMOTE_AMENDMENT_v2.md` (`2527ab5a…`, 5,701 B) | `I_ann=17.6804 r^3` and `B_remote=21.9279 r^3` as upper-bound certification claims, plus conclusions depending on those values. Historical evidence is retained. |
| `Q-RN5-MOMENT-004` | Drive object `1aCa-QG9CSrNUB9SUFKISifghSf-41fRy`, the CL-RNU-003 progress report (`59b8f002…`, 7,407 B) | Section 3 claim that the near integrand with `sqrt(E dy^4)` is certified, and the derived near/remote forecasts. Far progress and declared partial coverage are retained. |
| `Q-RN5-LM004-A1` | Drive object `1CHErajW7G2y8y7Mr0bKYMy2lJbA2HIAt`, the AUTO-RV-LM004 technical reconciliation (`c0571b7a…`, 5,093 B) | A1 statement that the C1 derivative supremum is duplicated, and the proposed cleanup. Prior technical reconciliation is retained with the erratum. |

The successor for the four RN5-MOMENT holds is the RN5 near-moment repair
(`1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5`, frozen as `RN5-PROOF`); for the LM004
hold it is `LM004_REVIEW_ERRATUM_v1.1` (`1QKujzzkp_nzCWFLky9ZayTzP_fbStxry`,
frozen as `RN5-LM004-ERRATUM`). Their bytes remain intact; what is excluded is
the certification claim at the named scope. `DEFECTIVE_SCOPE` is not a
declaration that every conclusion in the object is false (OP-PROT-019 §6).

### A bound member under logical quarantine

`Q-RN5-MOMENT-001` names a payload this repository binds byte-exact:
`engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_perc.py`,
record `RNENG-03` in `engine/rn_engine/BINDING.json`. It is **not** unbound —
the protocol says bytes and old manifests remain intact — and the checker is
**not** weakened. Instead the binding record carries the exclusion's key, class
and scope under `quarantine_exclusions`, stating that the member is bound as
bytes and logically quarantined at that scope, and invariant 5 below fails
closed if that annotation is missing. (`d3_amend_v2.py`, named by
`Q-RN5-MOMENT-002`, is a different file from the `d3_amend.py` bound as
`RNENG-04`; the exclusion does not name it.) The one carrier-manifest record an
exclusion already named by Drive id, `CR-RNU-DS3-SCALAR-SUPERSEDED`
(`Q-R17-RN-OLD`), carries the same annotation so the rule holds uniformly.

`tools/quarantine_check.py`, run in CI, asserts that:

1. the exclusion list and the register agree, both ways, on keys and classes;
2. every archive-member exclusion resolves to a real member of a real carrier
   with a matching payload digest;
3. **no excluded payload digest appears in any manifest in this repository** —
   nothing under quarantine has been silently pulled into verified content;
4. every exclusion carries a restoration test;
5. every bound member (`engine/rn_engine/BINDING.json`,
   `engine/carriers/MANIFEST.json`) whose payload digest or Drive id an
   exclusion names carries that exclusion's key, class and a non-empty scope
   under `quarantine_exclusions`; an annotation naming an exclusion that does
   not name the record is refused too.

## What is not here

The Drive's `99_DO_NOT_OPEN` vault is **not** mirrored. Only its metadata
appears, in `drive/inventory.jsonl`. The standing order is that models must not
open it for authority, proofs, certificates or "latest" status unless the
operator names a vault ID for forensic recovery. The manifest that states that
order and logs what was vaulted and why sits outside the vault and is held as a
reading copy under `drive/mirrors/01_ACTIVE_RESEARCH_PACKAGES — ROOT (the vault
manifest)/` (since 2026-09-19); holding it opens nothing.

Every move in git is a commit, so every move has a rollback record. Nothing is
deleted.
