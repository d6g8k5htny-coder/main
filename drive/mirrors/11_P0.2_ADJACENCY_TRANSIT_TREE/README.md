# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/11_P0.2_ADJACENCY_TRANSIT_TREE`

Drive folder id `1_V2QM-Kqy_-CoohZTg29QjV4eL1NMxDc`, **19 inventory items** in the
2026-09-17 snapshot: this lane folder, its four sub-folders, and fourteen files.

**This lane holds reading copies only. Not one byte-exact object is possible here.**
All fourteen files are native Google Docs. `drive/inventory.jsonl` carries an empty
`sha256` cell for every one of them; so does the `Source SHA-256` column of
`drive/source_map/Files.csv`; no register tab declares a body digest for any of the
fourteen; and `drive/source_map/Archive_Members.csv` carries no row for any of them.
So every stored row below is `exact: false`, the digest in each manifest row was
computed here for the first time from the text export, and **nothing stored here can
be checked against the Drive's own identity for the object** — because the Drive has
no identity on record for it. A text/plain export of a native Doc is not the object;
it is a rendering of one.

One qualification, and it is about payloads rather than objects: rows 3,999 and 4,000
of `drive/source_map/Payloads.csv` do carry SHA-256 values reached through two of these
fourteen ids, under member paths that end `!/decoded_payload`. Those digests belong to
the Base64 payload decoded *out of* `RS2-DATA-GP214A-v1.0` and `RS2-DATA-GP214B-v1.0`,
not to the two Docs that carry it. They make the payload inside those two documents
checkable against the corpus, and they leave all fourteen Docs exactly as uncheckable
as the previous paragraph says.

The inventory's `bytes` cell for a native Doc is the Drive-reported size, not a
payload length: the fourteen rows sum to 65,768 declared bytes, and the fourteen
exports stored here sum to 77,559. Both numbers are true and neither is a digest.

## What is here

Fourteen reading copies (`exact: false`, `stored: true`) and five tree-only folder
rows. Every one of the lane's 19 inventory items appears exactly once across the
five `_MANIFEST.jsonl` files, and no item appears twice.

Each sub-folder row's `inventory items` counts that folder itself as well as its files,
and the lane-root row counts all five folders, so the column deliberately over-counts and
does not sum to 19; the `stored` and `tree-only` columns do.

| directory | Drive folder id | inventory items | stored | tree-only | bytes stored |
|---|---|---:|---:|---:|---:|
| lane root | `1_V2QM-Kqy_-CoohZTg29QjV4eL1NMxDc` | 5 (this folder + four sub-folders) | 0 | 5 | 0 |
| `01.1_SADDLE_EXIT_BOUNDARIES/` | `1cMCkfzm5WeWW3V27LZGh2ubVUypWHFZz` | 2 | 1 | 0 | 2,639 |
| `01.2_INVARIANT_STRIP_DYNAMICS/` | `1yzR9gpy4qsKaQix4Gw2ESAttcoRs2H4G` | 2 | 1 | 0 | 2,444 |
| `01.3_PALM_INTEGRATION_REMAINDERS/` | `1R6GoA7c-qzseLoFKuW8IsYHup1jNIMb7` | 3 | 2 | 0 | 15,202 |
| `01.4_VALIDATION_CERTIFICATES/` | `1XEyCsUIScKCYBFCP03QXu8uAc1VhrG04` | 11 | 10 | 0 | 57,274 |
| **total** | | **19** | **14** | **5** | **77,559** |

The folder rows are tree-only because a folder has no bytes, not because anything was
withheld. Nothing in this lane was skipped, and nothing was fetched from
`99_DO_NOT_OPEN` (`1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`); no object of this lane is in
that vault and none was opened.

Enumerating each folder through the Drive connector on 2026-09-20 returned exactly the
four sub-folders and fourteen files the 2026-09-17 inventory lists, with the same ids:
no object of this lane was created, moved or removed after the snapshot. No id of this
lane appears in `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` or in any `_MANIFEST.jsonl`
under `drive/deltas/`. Two of them do appear elsewhere in the deltas:
`drive/deltas/2026-09-18/07_MODEL_ACCESSIBILITY_extras/Reading_Links.csv` carries a
published reading-copy link for `RS2-DATA-GP214A` and `RS2-DATA-GP214B`. That file
records where a reading copy was published, not any change to the object, and declares no
digest for anything.

## The status banners, verbatim

Each of the four leaf cards carries the same header shape. The 01.1 card's reads
"ARTIFACT: CL-NAV-036.2-v1.0 · AUTHOR: Claude (Anthropic), CL-* instance · CREATED:
2026-07-21 · CLASS: NAV — leaf charter + live-state card · CANONICAL IMPACT: NONE ·
AUTHORITY: none", and the 01.2, 01.3 and 01.4 cards carry the same line with
CL-NAV-036.3, CL-NAV-036.4 and CL-NAV-036.5 in place of CL-NAV-036.2. The middots are
the source's own; the line is one line in each file. Each card closes with
"Non-authority header guards against stale-card citation."

Each card also carries a fence of its own, under a heading the card spells "What must
not be claimed":

* 01.1 — "The endpoint blocks are local, degree-four-model results with an exact-field
  transfer criterion — they are not, alone, the P0.2 theorem. P0.2 remains OPEN."
* 01.2 — "Strip invariance is a sufficient local criterion, not the adjacency theorem;
  no numerical margin from stored runs is a certified constant. P0.2 remains OPEN."
* 01.3 — "The O(r³) power count applies to the certified bad-set decomposition, not yet
  to every true nonadjacent configuration; GT5 is proved-proposed within the exact
  six-pin model. P0.2 remains OPEN; no decimal here is a theorem constant."
* 01.4 — "No certificate stored here promotes anything by existing. A full-pass replay
  is evidence for the promotion case; the promotion itself requires the independent
  reconstruction, dependency crosswalk, and explicit human authorization listed in
  CM-044/CM-047."

The ten documents under `01.4_VALIDATION_CERTIFICATES/` and the audit under
`01.3_PALM_INTEGRATION_REMAINDERS/` each carry their own header banner. Nine of the ten
under 01.4, and the 01.3 audit, record zero independence credit and none canonical
impact, in the source's own words. The tenth is the 01.4 leaf card, which records
"CANONICAL IMPACT: NONE" and says nothing about independence credit either way; it is
not among the bullets below:

* `LCR-AUD-012-v1.0` — "Authority: same-organization technical audit only",
  "Canonical mathematical impact: NONE", "Independence credit: ZERO". Its section 1
  gives the current status as "PROOF-CANDIDATE / SAME-LINE VERIFIED / ORGANIZATIONALLY
  DISTINCT REVIEW PENDING." and its section 9 verdict is "SURVIVES — PROPOSED
  P02-LM-003 DETERMINANT-WEIGHTED POLYNOMIAL BOUNDARY-LAYER INTEGRATION."
* `GP-DER-190-v1.0` — "Status: PROVED DETERMINISTIC IDENTITY / ADDITIVE COMPANION /
  EXTERNAL REVIEW OF P02-LM-006 STILL OPEN", "Independence credit: ZERO — same OpenAI
  organizational/model family".
* `RS2-AUD-214-v1.0` — "Status: SAME-FAMILY ADVERSARIAL PASS / ORGANIZATIONALLY DISTINCT
  REVIEW STILL REQUIRED / NO PROMOTION", "Organizational independence credit: ZERO —
  same OpenAI organizational family as GP-DATA-214".
* `RS2-AUD-LM009-v1.0` — "Verdict: SAME-FAMILY ADVERSARIAL PASS / ORGANIZATIONALLY
  DISTINCT REVIEW STILL OPEN / NO PROMOTION", "Independence credit: ZERO — same OpenAI
  organizational family as the proof author".
* `RS2-AUD-LM012-v1.0` — "Verdict: V1.0 AMEND REQUIRED / V1.1 REPAIR AND DRIVE ROUNDTRIP
  PASS / EXTERNAL REVIEW OPEN / NO PROMOTION".
* `RS2-DATA-GP214A-v1.0` — "Status: RECONSTRUCTABLE SAME-FAMILY EVIDENCE / EXTERNAL
  REVIEW STILL OPEN / NO PROMOTION".
* `RS2-DATA-GP214B-v1.0` — "Status: 13/13 MUTATION GUARDS PASS / 5/5 DIRECT TORUS CHECKS
  PASS / EXTERNAL REVIEW STILL OPEN / NO PROMOTION".
* `RS2-DATA-LM009A-v1.0` — "Status: EXECUTED PASS / EXTERNAL REVIEW STILL REQUIRED / NO
  PROMOTION", closing "No terminal or promotion effect follows."
* `RS2-DATA-LM009B-v1.0` — "Status: PASS / EXTERNAL REVIEW STILL REQUIRED / NO
  PROMOTION".
* `S2-DATA-002-v1.0` — "Authority: factual execution-result publication only",
  "Canonical mathematical impact: NONE", "Independence credit: ZERO".

Three of those documents fence themselves explicitly against the reading that a PASS
closes anything:

> This is not a qualifying external approval. GP-REQ-215 remains open because the audit
> was performed by the same OpenAI organizational family. No independent-eyes credit,
> theorem promotion, P0.1 closure, P0.2 closure, machine change, package seal, release,
> or deletion follows.

> This verdict earns zero organizational-independence credit and therefore does not
> close P02-LM-009. The exact organizationally distinct review request LCR-REQ-029-v1.0
> remains open.

> This audit does not close P02-LM-012, P02-LM-011, or P0.2. It does not approve any
> imported interface, alter the terminal/review-pending object count, promote a theorem,
> seal a package, modify a machine root, authorize release, or authorize deletion.

`GP-DER-190-v1.0` carries the same discipline as a list: "This artifact proves only
deterministic repeated-node identities and confluent sign/factor limits. It does not
prove:", and the items it then names include "P02-LM-003, P0.1, P0.2, or Theorem B"
and "theorem promotion or external release". Of the Gaussian facts P02-LM-006 still
needs it says "This certificate proves none of those probabilistic statements."

`LCR-AUD-012-v1.0`'s own scope section is the narrowest statement in the lane:
"P02-LM-003 would close only a conditional measure-theoretic interface. It would not
prove H1, H2, a deterministic inclusion for the exact field, local-to-global adjacency,
or P0.2."

## What the leaf cards warn about, in their own words

* An identifier collision. The 01.1 card records "Identifier collision warning
  (GP-CLS-001): a second, mathematically distinct document also carries ID
  GP-DER-044-v1.0" and instructs "Always cite by full title + Drive ID."; the 01.3
  card repeats it from the other side. Two different Drive objects share one artifact
  id, and this repository does not resolve that — it mirrors both cards' warning.
* A falsified shortcut. The 01.2 card reads "Falsified simplification — do not reuse:
  the fixed threshold m22_S > −2r is falsified as the deterministic inclusion."
* An open ballot. The same card records "Pending ballot: GP-CLS-PROP-015 proposes
  terminal closure of the closed strip-threshold majorant (in _CONSENSUS_BALLOTS;
  unresolved as of this card)." Unresolved is where it stays; nothing here resolves it.
* A decimal that is not a constant. The 01.3 card says of the normalizer that "the
  decimal 3.2309785… is a planar-limit value, not an exact fixed-torus identity".
* An open obligation. The same card: "Open audit obligation: the determinant-weight
  second-moment estimate DM2 (E[W_r²] ≤ C r⁴) and the capture majorant still need
  independent audit (CM-043/CM-047)."
* What an instrument is worth. The 01.4 card: "Binding rule from CM-047: instrument
  certification must not be cited as theorem proof.", and of certificate packages,
  "Packages missing any element should be treated as incomplete, not approximately
  complete."

A leaf named `01.4_VALIDATION_CERTIFICATES` is the source's name for its own folder.
`P0.2` is an open problem in this repository's own `docs/OPEN_PROBLEMS.md` §D and in
`claims/graph.json`, whose `RV-LM003-MAIN` record carries the queue state "EXACT-HASH
THEOREM VERDICT OPEN / MATHEMATICS-PASS / P0.2 OPEN". A document in this lane calling
itself a certificate closes nothing. The 01.4 leaf card says so before anyone else does
— "No certificate stored here promotes anything by existing." — and it is the only one of
the four that addresses certificates; 01.1, 01.2 and 01.3 do not mention them.

## Two byte-level checks that were run, and what they are not

Neither check is a certificate, a replay or a review. Each is an arithmetic statement
about the stored export and the numbers the stored export declares about itself; no
extracted payload was executed and nothing was graded.

1. **The two GP214 carriers reconstruct, and the corpus corroborates the payloads.**
   Applying the four reconstruction steps printed inside `RS2-DATA-GP214A-v1.0` to its
   stored export yields 4,380 gzip bytes with SHA-256 `83319d6e…4f6a` and, after
   gunzip, 12,834 bytes with SHA-256 `3c076998…0f91`. The same for
   `RS2-DATA-GP214B-v1.0`: 2,941 gzip bytes `82abf99a…90b2`, inflating to 10,649 bytes
   `1657bd96…eee3`. Each of the four figures equals what that carrier declares about
   itself, and the two inflated figures also equal, digit for digit, the SHA-256 and
   byte count that rows 4,000 and 3,999 of `drive/source_map/Payloads.csv` record for
   the `decoded_payload` reached through those same two Drive ids. So these two inner
   payloads *are* checkable against the corpus and they check out — which is a fact
   about bytes surviving a Docs round trip, and about nothing else. Neither payload was
   executed, and this says nothing about whether the verifier they contain is correct or
   about what its result means.
2. **The two LM009 carriers reconstruct to the identities the audit recorded, not to
   the ones they declare.** Applying the marker rule printed inside
   `RS2-DATA-LM009A-v1.0` gives 5,294 bytes `44e605c1…fb22`; inside
   `RS2-DATA-LM009B-v1.0`, 1,326 bytes `fe44ed10…a3a6`. Those differ from the
   `SOURCE_BYTES: 5959` / `RESULT_BYTES: 3241` the carriers themselves print, and they
   agree exactly with the two figures `RS2-AUD-LM009-v1.0` section 6 records as the
   marker-extracted Drive identities. That audit anticipates the difference in terms:
   "The Drive carriers are review aids. Their identities are distinct from the larger
   pre-publication local source/result files recorded as provenance inside the
   carriers." So this is a documented two-layer provenance, not a defect found here.

## A document in this lane whose payload is absent

`S2-DATA-002-v1.0 — Explicit GT5 Radius Certificate Result Carrier`
(`1fB0sz_PTfieU4usNrXufwvXTJjDgoSSZa08Qat6jFKQ`) declares a 16,883-byte JSON payload
with SHA-256 `04be1073…e4a0` and prints a seven-step extraction rule whose step 2 is
"Match BEGIN_BASE64_RESULT and END_BASE64_RESULT as unique whole lines." The stored
export ends at the `BEGIN_BASE64_RESULT` line. There is no Base64 between markers,
there is no closing marker, and the declared payload cannot be extracted from this
object.

That is not a discovery of this port; the corpus already records it, twice, in
`drive/inventory.jsonl` titles. The title of `13EoNjXvlycWcqmzZRSdRNs9znKk7JSA0W25aeA_onIo`
says of the S2-REQ-002 review that "G1–G18 mathematics ALL PASS and source carrier
byte-exact, but the RESULT CARRIER WAS NEVER PUBLISHED — VERDICT: AMEND REQUIRED
(publication only; converts to PASS on publication)", and the title of
`1HJr58NmwIEuVRXFxKzNjrYTEOXkulE1Dh28nVncmHow` records the later "PASS CONVERSION of
CL-AUD-245 under S2-REQ-011: S2-DATA-002 result carrier V0–V7 ALL PASS (5714 B gzip →
16883 B, SHA 04be1073…ae4a0; 16/16 checks, 7/7 controls)". The object that carries the
published payload is a different Drive id in a different lane
(`1lEDwlVhZOhfQiDx9qrYaB0_dezldUTTYkgKB0vDq3Sw`, titled
`S2-DATA-002-RESULT-v1.0 — Explicit GT5 Radius Certificate Result Carrier`, under
`15_REVIEWS_RESPONSES_AND_CLOSURES`), and it is not held here. The empty stub in this
lane is mirrored as what it is. Which of the two ids the P0.2 lane should hold is a
source-register question for an operator; nothing was substituted, re-based or
repaired here.

## Where the register disagrees with a mirrored verdict, and which is which

`RS2-AUD-214-v1.0` is a same-family audit of `GP-DATA-214` / `GP-DER-214` and its own
verdict is a pass. A later organizationally distinct review of the same objects reached
a different disposition, and the register is where that lives, not these bytes: the
`registers/json/file_catalog.json` row for `1SsG0vN-XYWbJfkJ8sc1JSo7LGar5f2rIRBlYNFk6x68`
is titled in part "AMEND REQUIRED — RECORD-CLASS ONLY. NO MATHEMATICAL DEFECT FOUND.",
and among its four amendments it records one as "convergent with RS2-AUD-214 §7" — the
same observation the mirrored audit makes of itself when it says "This is a
record-quality limitation, not a demonstrated theorem defect." and that its thirteen
guards "should not be described as thirteen executions of mutated GP source." The two
records are not in conflict about the mathematics; they differ in class, and the
independence-requiring gate the mirrored audit names remains open in its own words:
"OPEN — GP-REQ-215 STILL REQUIRES AN ORGANIZATIONALLY DISTINCT VERDICT." The same
document forbids the inheritance a reader might reach for: "NO APPROVAL TRANSFERS FROM
GP-DATA-208 TO GP-DATA-214." Of its own numerical margin it says "This is evidence of
margin, not authorization to tighten the controlling certificate without a new frozen
successor." None of that is weighed here.

## How these fourteen objects sit in the register

`registers/json/file_catalog.json` carries a row for all fourteen. Every one of those
rows has `P02` as its domain and `METADATA_ONLY` as its routing status, with the
coverage note "Metadata inventoried; proof review only where explicitly recorded in the
closure report […]" — the cell continues "; baseline active-tree inventory + bounded R17
updates; refresh target metadata before mutation". Mirroring the bytes does not change
that cell and this repository does not edit it.

`registers/json/p02_exact_hash_review_manifest.json` is the register tab for this
problem. Its second row reads "TERMINAL DEPENDENCIES: P02-LM-001 · P02-LM-002 ·
P02-LM-005 · P02-LM-008" and its third reads "EXACT-HASH APPROVED: P02-LM-007.
REMAINING REVIEW-PENDING GRAPH INTERFACES: P02-LM-003 · LM004 v1.1 · LM006 · LM009 ·
LM010 v1.1 · LM011 · LM012 · LM013. LM013 requires three exact review objects." Four of
the mirrored documents target interfaces on that review-pending list — LM003, LM006,
LM009 and LM012 — and none of them is the terminal verdict that list is waiting for.

The body identities the mirrored audits quote for their targets agree, digit for digit,
with the ones `docs/OPEN_PROBLEMS.md` §D transcribes from `review_queue.json`: 6,874
bytes and `5173d26d…` for LM003's `LCR-DER-014-v1.0`, 3,919 and `ccc07d95…` for LM009's
`LCR-DER-027-v1.0`, 11,543 and `68df3208…` for the LM012 successor `LCR-DER-043-v1.1`.
Those are the objects under review. This lane holds none of their bytes, and no
byte-exact copy of any of them exists anywhere here — but each is held as a reading copy
under `drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/`, at `exact: false`, so the bytes a
reader would find there are a text export and not the object the review names.

## What this does not establish

Mirroring is not review, replay, endorsement, promotion or acceptance. Every status word
above is transcribed from the mirrored bytes or from a register tab, and none was
decided, weighed, aged or moved here.

Nothing in this directory is byte-exact and nothing in it can be made byte-exact from
the corpus as it stands. A text/plain export of a native Google Doc loses the object's
own bytes, carries no declared digest, and is not a frozen body; a PDF rendering of the
same Doc would not be one either. The fourteen digests in these manifests are digests of
this repository's exports, computed here, comparable to nothing the Drive publishes — so
a future re-export that differs from them is evidence about the export path, not
evidence that a Drive object changed.

No carrier in this lane was executed. The two payload reconstructions recorded above
decode and hash bytes; they run no verifier, reproduce no computation, and confirm no
claim the carriers make about mathematics. A document titled `…CERTIFICATE…` is the
source's word for its own document, and a folder named `01.4_VALIDATION_CERTIFICATES` is
the source's name for its own folder; neither is a certificate in the sense
`research/interval/` uses, and no bound, radius, exponent, threshold or decimal appearing
in these exports is certified by anything in this repository.

P0.2 is open, and remains exactly as open as it was before this port: three of the four
leaf cards say "P0.2 remains OPEN" — 01.1, 01.2 and 01.3; the 01.4 card does not mention
P0.2 at all, and what it says instead is quoted below. The audits say their reviews earn zero
organizational-independence credit, and `CLAUDE.md` rule 6 is why that matters — "A
same-provider reviewer earns zero independence credit." Every document mirrored here was
prepared by OpenAI or Anthropic lines already inside the program, so none of them is the
organizationally distinct verdict that P02-LM-003, LM006, LM009 and LM012 are waiting
for. The five validity premises of Theorem D1 v2.2(2) and `D3-LEMMA-RN-UNIF` are
untouched by anything here. Nothing here composes the 2D upper or lower tracks with the
3D lifetime track, nothing here enters the prize track in either direction, and no
original prize problem is solved.

Nothing stored under this directory is imported, executed, scheduled, tested or read by
any module, checker, receipt or claim in this repository. `tools/verify_manifests.py`
confirms the byte count and SHA-256 of each stored export against its manifest row, and
`tools/mirror_quotes_check.py` confirms that every quotation above is verbatim in a
stored byte, a `registers/json` cell, a `drive/inventory.jsonl` title, `CLAUDE.md` or
`claims/graph.json`. Neither tool reads, grades or moves any claim, premise or
obligation.
