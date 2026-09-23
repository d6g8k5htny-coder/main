# `01.4_VALIDATION_CERTIFICATES` — reading copies

Drive folder id `1XEyCsUIScKCYBFCP03QXu8uAc1VhrG04`, 11 inventory items: this folder
(tree-only, indexed in the lane-root `_MANIFEST.jsonl`) and ten native Google Docs,
stored here as text exports.

**Nothing here is byte-exact and nothing here can be made byte-exact.** All ten objects
are native Google Docs; `drive/inventory.jsonl` declares no `sha256` for any of them,
`drive/source_map/Files.csv` leaves their `Source SHA-256` column empty, and no register
tab declares a body digest for any of them, so the ten digests in `_MANIFEST.jsonl` were
computed here from this repository's own exports and can be compared with nothing the
Drive publishes. Two inner payloads are the exception, and they are payloads rather than
objects: rows 3,999 and 4,000 of `drive/source_map/Payloads.csv` carry a SHA-256 and byte
count for the `decoded_payload` reached through `RS2-DATA-GP214B-v1.0` and
`RS2-DATA-GP214A-v1.0`. Those two digests describe what is encoded inside two of these
Docs, never the Docs. The folder's name is the source's name for
its own folder; it is not this repository's assessment of what the ten documents are.

| file | Drive id | exactness | bytes stored | inventory bytes |
|---|---|---|---:|---:|
| `00_LEAF_CARD — 01.4 Validation Certificates (CL-NAV-036.5).export.txt` | `170r3ScfQ0XhW-Hf6zjs-WGzoxUgFFmju8Utayyt0prc` | `exact: false` | 2,556 | 2,905 |
| `GP-DER-190-v1.0 — Exact Hermite Pin Certificate for P02-LM-006.export.txt` | `1ugHOaFmfuo7hoCoFeNbvjjsJ5qwXaiL17FEIWWpc_g8` | `exact: false` | 6,001 | 5,166 |
| `RS2-AUD-214-v1.0 — Adversarial Replay and Exact-Torus Cross-Check of GP-DATA-214 ∕ GP-DER-214.export.txt` | `1f0VG_xji4QumQTSSsjwuS8rpfwW2ByOUxUDVqXDwA28` | `exact: false` | 10,250 | 7,360 |
| `RS2-AUD-LM009-v1.0 — P02-LM-009 Same-Family Adversarial Reconstruction.export.txt` | `1XXHOvgtk-Kv28v7essHbXShpgXzk7aB4QHqlpwMFO7E` | `exact: false` | 7,968 | 6,188 |
| `RS2-AUD-LM012-v1.0 — P02-LM-012 Deep-Threshold Defect, Repair, and Freeze Audit.export.txt` | `1LxBFj9rDf_eV5kYINLlRZDF_b4HHKhptj-hkev-tqxI` | `exact: false` | 8,357 | 6,483 |
| `RS2-DATA-GP214A-v1.0 — GP-DATA-214 Adversarial Verifier Source Carrier.export.txt` | `19skReZC0BgalbXykwQJCVsDzNYFTBqPigE1ermcie3M` | `exact: false` | 6,878 | 6,342 |
| `RS2-DATA-GP214B-v1.0 — GP-DATA-214 Adversarial Verifier Result Carrier.export.txt` | `1xOcNb4qkUw3tQChCq3pV9PUP4NYX965wbQyrIHN4rxQ` | `exact: false` | 4,983 | 4,874 |
| `RS2-DATA-LM009A-v1.0 — P02-LM-009 Adversarial Verifier Source Carrier.export.txt` | `1vJcxZE8VAyxfrWT9Qge5kACcrL4zeos6pIcyD3xtZuA` | `exact: false` | 6,478 | 4,751 |
| `RS2-DATA-LM009B-v1.0 — P02-LM-009 Adversarial Verifier Result Carrier.export.txt` | `1udGxXR4teNyDR-Hz9vP_XgRB_k_IygjdGWYPDHS3gc8` | `exact: false` | 2,757 | 2,760 |
| `S2-DATA-002-v1.0 — Explicit GT5 Radius Certificate Result Carrier.export.txt` | `1fB0sz_PTfieU4usNrXufwvXTJjDgoSSZa08Qat6jFKQ` | `exact: false` | 1,046 | 1,675 |

The stored and inventory byte figures differ in both directions because the inventory
cell is the Drive-reported size of a native Doc, which is neither a payload length nor a
digest. The file name carrying `∕` writes the Drive title's `/` as U+2215 so it stays
one path segment; the `title` field of the manifest row keeps the real slash.

Every one of these ten ids appears in `registers/json/file_catalog.json` with `P02` as
its domain and `METADATA_ONLY` as its routing status, under a coverage note beginning
"Metadata inventoried; proof review only where explicitly recorded in the closure
report […]"; the cell continues "; baseline active-tree inventory + bounded R17 updates;
refresh target metadata before mutation".

## The status banners, verbatim

**`00_LEAF_CARD — 01.4 Validation Certificates`.** Header line: "ARTIFACT:
CL-NAV-036.5-v1.0 · AUTHOR: Claude (Anthropic), CL-* instance · CREATED: 2026-07-21 ·
CLASS: NAV — leaf charter + live-state card · CANONICAL IMPACT: NONE · AUTHORITY: none";
closing line: "Non-authority header guards against stale-card citation." Under the
heading it spells "What must not be claimed": "No certificate stored here promotes
anything by existing. A full-pass replay is evidence for the promotion case; the
promotion itself requires the independent reconstruction, dependency crosswalk, and
explicit human authorization listed in CM-044/CM-047." Of instruments it records
"Binding rule from CM-047: instrument certification must not be cited as theorem
proof."; of what promotion would still need, "Replay obligation (blocking): promotion of
the GP-DER-047 candidate requires replaying the full certificate on frozen exact
finite-r Gaussian-law draws"; and of packages, "Packages missing any element should be
treated as incomplete, not approximately complete." It also keeps a falsifier in view:
"Decisive falsifier to keep in view: one exact-law configuration satisfying REG_r, s ≤
−C_cap(U), and H_{5,r}(D) ≤ r^{−1} whose exact selected unstable branch does not
converge to M falsifies the candidate theorem (GP-DER-047)."

**`GP-DER-190-v1.0`.** "Class: DER / DETERMINISTIC CONFLUENT-COORDINATE CERTIFICATE /
REUSABLE HELPER LEMMA", "Authority: research preparation only", "Canonical mathematical
impact: NONE", "Independence credit: ZERO — same OpenAI organizational/model family",
"Status: PROVED DETERMINISTIC IDENTITY / ADDITIVE COMPANION / EXTERNAL REVIEW OF
P02-LM-006 STILL OPEN". Its section 7 is headed by "This artifact proves only
deterministic repeated-node identities and confluent sign/factor limits. It does not
prove:", and the items it names include "convergence of the conditioned Gaussian laws",
"P02-LM-003, P0.1, P0.2, or Theorem B" and "theorem promotion or external release". Of
the five Gaussian facts it lists as still necessary to complete P02-LM-006 it says
"This certificate proves none of those probabilistic statements."

**`RS2-AUD-214-v1.0`.** "Authority: additive research audit only", "Canonical
mathematical impact: NONE", "Organizational independence credit: ZERO — same OpenAI
organizational family as GP-DATA-214", "Status: SAME-FAMILY ADVERSARIAL PASS /
ORGANIZATIONALLY DISTINCT REVIEW STILL REQUIRED / NO PROMOTION". Its section 1 closes:

> This is not a qualifying external approval. GP-REQ-215 remains open because the audit
> was performed by the same OpenAI organizational family. No independent-eyes credit,
> theorem promotion, P0.1 closure, P0.2 closure, machine change, package seal, release,
> or deletion follows.

Of its own numeric margin it says "This is evidence of margin, not authorization to
tighten the controlling certificate without a new frozen successor."; of its thirteen
guards, "They should not be described as thirteen executions of mutated GP source."; of
the target's internal controls, "This is a record-quality limitation, not a demonstrated
theorem defect." Its section 8 records the controlling external-review status as "OPEN —
GP-REQ-215 STILL REQUIRES AN ORGANIZATIONALLY DISTINCT VERDICT." and a predecessor
firewall reading "CL-AUD-253 REMAINS AMEND REQUIRED FOR GP-DATA-208." and "NO APPROVAL
TRANSFERS FROM GP-DATA-208 TO GP-DATA-214." Among the uses it forbids are "counting this
as organizationally distinct review" and "promoting GP-DATA-214".

**`RS2-AUD-LM009-v1.0`.** "Authority: research preparation and review-readiness evidence
only", "Canonical mathematical impact: NONE", "Independence credit: ZERO — same OpenAI
organizational family as the proof author", "Verdict: SAME-FAMILY ADVERSARIAL PASS /
ORGANIZATIONALLY DISTINCT REVIEW STILL OPEN / NO PROMOTION". Its section 7:

> This verdict earns zero organizational-independence credit and therefore does not
> close P02-LM-009. The exact organizationally distinct review request LCR-REQ-029-v1.0
> remains open.

> No status changes follow for P02-LM-003, P02-LM-006, P02-LM-008, P02-LM-011, or P0.2.
> No theorem is promoted, no package is sealed, no machine state is changed, and no
> release or deletion is authorized.

**`RS2-AUD-LM012-v1.0`.** "Authority: research preparation and review-routing evidence
only", "Canonical mathematical impact: NONE", "Independence credit: ZERO — same OpenAI
organizational family", "Verdict: V1.0 AMEND REQUIRED / V1.1 REPAIR AND DRIVE ROUNDTRIP
PASS / EXTERNAL REVIEW OPEN / NO PROMOTION". Its decisive finding is "Therefore v1.0 is
AMEND REQUIRED for the capture-shortcut overstatement.", its distinction is "The
deep-saddle condition is a sample-event restriction, not a radius.", and of the amended
predecessor it says "The v1.0 object remains preserved provenance and may still be cited
for defect history, not affirmative approval." Its section 8 closes: "This audit does
not close P02-LM-012, P02-LM-011, or P0.2. It does not approve any imported interface,
alter the terminal/review-pending object count, promote a theorem, seal a package,
modify a machine root, authorize release, or authorize deletion."

**`RS2-DATA-GP214A-v1.0`** — "Class: DATA / SAME-FAMILY ADVERSARIAL VERIFIER SOURCE /
GZIP-BASE64 CARRIER", "Organizational independence credit: ZERO", "Canonical
mathematical impact: NONE", "Status: RECONSTRUCTABLE SAME-FAMILY EVIDENCE / EXTERNAL
REVIEW STILL OPEN / NO PROMOTION".

**`RS2-DATA-GP214B-v1.0`** — the same class and credit lines, and "Status: 13/13
MUTATION GUARDS PASS / 5/5 DIRECT TORUS CHECKS PASS / EXTERNAL REVIEW STILL OPEN / NO
PROMOTION".

**`RS2-DATA-LM009A-v1.0`** — "Authority: research preparation only", "Canonical
mathematical impact: NONE", "Independence credit: ZERO — OpenAI organizational family",
"Status: EXECUTED PASS / EXTERNAL REVIEW STILL REQUIRED / NO PROMOTION", closing "No
terminal or promotion effect follows."

**`RS2-DATA-LM009B-v1.0`** — the same authority, impact and credit lines, "Status: PASS
/ EXTERNAL REVIEW STILL REQUIRED / NO PROMOTION", and a scope list whose items include
"supplies no organizationally distinct review credit", "does not approve P02-LM-006 or
P02-LM-008" and "does not close or promote P0.2."

**`S2-DATA-002-v1.0`** — "Class: DATA / BYTE-RECOVERABLE RESULT CARRIER / BASE64",
"Authority: factual execution-result publication only", "Canonical mathematical impact:
NONE", "Independence credit: ZERO", and of its declared payload, "Execution status
represented by payload: 16/16 checks PASS; 7/7 controls PASS."

## Two byte-level checks, and one absence

These are arithmetic statements about the stored exports and the numbers those exports
print about themselves. Nothing was executed, replayed, reviewed or graded.

* **The GP214 pair reconstructs to its declared identities.** Following the four
  reconstruction steps printed in `RS2-DATA-GP214A-v1.0` — decode the Base64 between the
  markers, check the gzip digest and byte count, gunzip, check the raw digest and byte
  count — yields 4,380 gzip bytes `83319d6e…4f6a` inflating to 12,834 bytes
  `3c076998…0f91`; for `RS2-DATA-GP214B-v1.0`, 2,941 gzip bytes `82abf99a…90b2`
  inflating to 10,649 bytes `1657bd96…eee3`. All four figures equal what each carrier
  declares, and the two inflated figures also equal the byte count and SHA-256 that rows
  4,000 and 3,999 of `drive/source_map/Payloads.csv` record for the `decoded_payload`
  reached through those two Drive ids — so the corpus, not only the carrier, corroborates
  the payload. The extracted Python was not run and the extracted JSON was not evaluated;
  a payload that decodes to its declared digest is still only bytes.
* **The LM009 pair reconstructs to the audit's marker-extracted identities, not to the
  local ones it also prints.** The marker rule in `RS2-DATA-LM009A-v1.0` yields 5,294
  bytes `44e605c1…fb22`; the one in `RS2-DATA-LM009B-v1.0` yields 1,326 bytes
  `fe44ed10…a3a6`. Those are not the "SOURCE_BYTES: 5959" and "RESULT_BYTES: 3241" the
  carriers declare as their local pre-publication identities, and they are exactly the
  two figures `RS2-AUD-LM009-v1.0` section 6 records, which that section explains: "The
  Drive carriers are review aids. Their identities are distinct from the larger
  pre-publication local source/result files recorded as provenance inside the carriers."
  So the difference is documented in the corpus, not found here.
* **`S2-DATA-002-v1.0` has no payload.** It declares "Bytes: 16883" and "SHA-256:
  04be107343f4a7a45e9df03d01e1f8d7fb54ebb324d1396ec490016a2d8ae4a0", and its extraction
  rule says "Match BEGIN_BASE64_RESULT and END_BASE64_RESULT as unique whole lines." The
  stored export ends at the `BEGIN_BASE64_RESULT` line: no Base64, no closing marker, no
  extractable payload. The corpus already records this, in `drive/inventory.jsonl`
  titles: `13EoNjXvlycWcqmzZRSdRNs9znKk7JSA0W25aeA_onIo` is titled in part "the RESULT
  CARRIER WAS NEVER PUBLISHED — VERDICT: AMEND REQUIRED (publication only; converts to
  PASS on publication)", and `1HJr58NmwIEuVRXFxKzNjrYTEOXkulE1Dh28nVncmHow` records the
  later "PASS CONVERSION of CL-AUD-245 under S2-REQ-011: S2-DATA-002 result carrier V0–V7
  ALL PASS (5714 B gzip → 16883 B, SHA 04be1073…ae4a0; 16/16 checks, 7/7 controls)". The
  published payload lives under a different Drive id in a different lane
  (`1lEDwlVhZOhfQiDx9qrYaB0_dezldUTTYkgKB0vDq3Sw`) and is not held here. Nothing was
  substituted or repaired; the stub is mirrored as the stub it is.

## What this does not establish

Mirroring is not review, replay, endorsement, promotion or acceptance. Every status word
above is transcribed from these bytes or from a register tab and none was decided,
weighed or aged here. A document calling itself a certificate does not make it one, and
a folder named `01.4_VALIDATION_CERTIFICATES` names itself; no bound, radius, exponent,
threshold, margin or decimal in these ten exports is certified by anything in this
repository, and none of them goes through `research/interval/`.

Every verdict here is a same-provider verdict. `CLAUDE.md` rule 6: "A same-provider
reviewer earns zero independence credit." Each document says the same of itself, and
each names an independence-requiring gate that stays open — GP-REQ-215 for
`RS2-AUD-214-v1.0`, LCR-REQ-029-v1.0 for `RS2-AUD-LM009-v1.0`, an external exact-hash
review for `RS2-AUD-LM012-v1.0`, and external review of P02-LM-006 for `GP-DER-190-v1.0`.
None of those gates moves here.

No reviewed object is held in this lane, and no byte-exact copy of any of them is held
anywhere in this repository. Three are held elsewhere as reading copies, at
`exact: false`, under `drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/`:
`LCR-DER-019-v1.0`, `LCR-DER-027-v1.0` and the v1.1 successor to `LCR-DER-043-v1.0`. A
reading copy is a text export, not the object the review names. `GP-DATA-214`,
`GP-DER-214`, `GP-DER-047` and `GP-DATA-048` are not held at all. What is held here is
ten documents *about* them. The identities those documents declare for their targets agree with what
`docs/OPEN_PROBLEMS.md` §D transcribes from `registers/json/review_queue.json` — 3,919
bytes and `ccc07d95…` for LM009, 11,543 and `68df3208…` for the LM012 successor — and
agreement of transcribed digits is not verification of a body this repository does not
hold.

P0.2 is open in `docs/OPEN_PROBLEMS.md` §D and in `claims/graph.json`, and is exactly as
open after this port as before it. Nothing here composes the 2D upper or lower tracks
with the 3D lifetime track, nothing here enters the prize track in either direction, no
original prize problem is solved, and the five validity premises of Theorem D1 v2.2(2)
and `D3-LEMMA-RN-UNIF` are untouched. Nothing stored in this directory is imported,
executed, scheduled, tested or read by any module, checker, receipt or claim in this
repository.
