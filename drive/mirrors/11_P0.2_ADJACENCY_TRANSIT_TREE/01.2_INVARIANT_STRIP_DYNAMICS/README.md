# `01.2_INVARIANT_STRIP_DYNAMICS` — reading copy

Drive folder id `1yzR9gpy4qsKaQix4Gw2ESAttcoRs2H4G`, 2 inventory items: this folder
(tree-only, indexed in the lane-root `_MANIFEST.jsonl`) and one native Google Doc,
stored here as a text export.

**Nothing here is byte-exact and nothing here can be made byte-exact.** The object is a
native Google Doc; `drive/inventory.jsonl` declares no `sha256` for it and no register
tab declares a body digest for it, so the digest in `_MANIFEST.jsonl` was computed here
from this repository's own export and can be compared with nothing the Drive publishes.

| file | Drive id | exactness | bytes stored | inventory bytes |
|---|---|---|---:|---:|
| `00_LEAF_CARD — 01.2 Invariant Strip Dynamics (CL-NAV-036.3).export.txt` | `1P58tHG9TQ6dKLWcdi4yihD8dsxK39O3UjKIzpAN7boI` | `exact: false` — reading copy | 2,444 | 2,969 |

The two byte figures differ because the inventory cell is the Drive-reported size of a
native Doc, which is neither a payload length nor a digest.

## The status banner, verbatim

The card's header line reads "ARTIFACT: CL-NAV-036.3-v1.0 · AUTHOR: Claude (Anthropic),
CL-* instance · CREATED: 2026-07-21 · CLASS: NAV — leaf charter + live-state card ·
CANONICAL IMPACT: NONE · AUTHORITY: none", and it closes "Non-authority header guards
against stale-card citation."

Under the heading the card spells "What must not be claimed" it says: "Strip invariance
is a sufficient local criterion, not the adjacency theorem; no numerical margin from
stored runs is a certified constant. P0.2 remains OPEN."

Three further lines of the card are fences rather than results:

* "Falsified simplification — do not reuse: the fixed threshold m22_S > −2r is falsified
  as the deterministic inclusion." The card adds that "The three stored ρ=0
  counterexamples falsify only the ρ=0 simplification (full margins positive ≈0.988,
  1.829, 0.978)." and that "The live sufficient condition is the jet-dependent
  GP-DER-037 threshold."
* Of the mass estimate it says the complement "has O(r³) typed pair-Palm mass in the
  exact finite-jet model (GP-DER-043 / CM-040) — an analytic reduction, not an empirical
  slope."
* "Pending ballot: GP-CLS-PROP-015 proposes terminal closure of the closed
  strip-threshold majorant (in _CONSENSUS_BALLOTS; unresolved as of this card)."

The adaptive scales the card lists it labels itself: "CANDIDATE / NONCANONICAL".

## What this does not establish

Mirroring is not review, replay, endorsement, promotion or acceptance. Every status word
above is the card's own and none was decided here; in particular the pending ballot is
recorded as unresolved and stays unresolved, and the three decimals the card prints are
quoted as the card's own margins, not as constants. The reading copy is not the object.
None of `GP-DER-037`, `GP-DER-043` or `GP-DER-047` is held in this repository. P0.2 is
open in `docs/OPEN_PROBLEMS.md` §D and in `claims/graph.json`, and is exactly as open
after this port as before it.
