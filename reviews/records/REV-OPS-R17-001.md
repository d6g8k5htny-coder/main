# RV-OPS-R17 — nonauthor technical review of OP-PROT-019-v1.1 (R17)

**Record:** `reviews/records/REV-OPS-R17-001.json` (the normative artifact; this file is its
readable companion and carries the scripts).
**Route:** `RV-OPS-R17`, Review Queue technical status **READY**, Independence status
**EXTERNAL_REVIEW_OPEN**, Reviewer **UNASSIGNED**.
**Object:** `OP-PROT-019-v1.1_R17.md`, Drive ID `1hBph5Fpxd5dVrolNkvzU8nb7xpxUiUdc`,
**14,073 bytes**, **sha256 `04987ba47b58be623a3b2a5a8a21236f43d4393cc04009e87f4a62b59a787390`**.
**Author family:** openai (`OpenAI / OPS4-20260917-b9c2`). **Reviewer family:** anthropic.
**Technical verdict:** `AMEND`. **independence_credit: 0.** **gate_status_after: `UNCHANGED`.**

> **Read this first.** This is a verdict on an object. It moves no gate, it edits no
> register, and it creates no organizational independence. The external-review obligation on
> this route is exactly as open as it was before this file existed. This route is the one
> where the reviewed object *is* the governance machinery, so it needs saying plainly: the
> favourable half of this review is not authority to invoke R17 for anything, and an
> unfavourable half is not authority to set it aside.

---

## 1. Obtaining the object

`tools/drive_index.py` resolved the route's source link to the file, and the connector
returned it as base64, which I decoded and hashed myself:

```
$ python3 tools/drive_index.py id 1hBph5Fpxd5dVrolNkvzU8nb7xpxUiUdc
  bytes   14073
  sha256  04987ba47b58be623a3b2a5a8a21236f43d4393cc04009e87f4a62b59a787390
  path    04_ACTIVE_GOVERNANCE_AND_DIRECTIVES/04_PROTOCOL_STANDARDS_AND_REFERENCE_IMPLEMENTATIONS/
          AI-DRIVE-R17 — NAVIGATION_QUARANTINE_REVIEW_2026-09-17/OP-PROT-019-v1.1_R17.md

my own computation after base64-decoding the connector's response:
  bytes   14073
  sha256  04987ba47b58be623a3b2a5a8a21236f43d4393cc04009e87f4a62b59a787390
```

**Exact agreement with the `Body SHA-256` and `Body bytes` of the RV-OPS-R17 row.** The
object is `text/markdown`, not a native Doc, so R17 §2's revision-aware export requirement
does not apply; no `exportMimeType` was passed and the raw bytes were hashed with no
normalisation.

### 1.1 Finding OBJ-01 — the repository's copy is not the object

The task pointed me at `governance/protocols/OP-PROT-019-v1.1_R17.md`. It is **not** the
reviewed bytes:

```
object (Drive)      14073 bytes  04987ba47b58be623a3b2a5a8a21236f43d4393cc04009e87f4a62b59a787390
repository copy     14073 bytes  efcfdd5cde32e7bd706f432689f5b022b0a2a8ec289164eed170a61d7a0071d9
```

The byte counts are **identical** and the digests are not. `diff` reduces the difference to
two whitespace edits that cancel in length — a blank line inserted after line 1, between the
H1 and the `AI-DRIVE-AUTONOMY-R17 ·` subtitle, and one trailing newline removed at line 91 —
and the resulting one-byte shift makes `cmp -l` report **13,724 differing byte positions**.

This matters more than a whitespace nit. R17 §2 makes the exact digest the identity, and §6
requires "matching raw digest/bytes" for EXACT_DUPLICATE. The repository copy matches
neither, and because the byte count coincides, *a reviewer who checks the byte count alone
confirms the wrong bytes*. Severity **MAJOR**.

---

## 2. Exposure, written before the verdict

Heavy, and it disqualifies this review for independence purposes regardless of the family
mismatch. In short: I had already read this repository's derived port of the object in full;
the author's own `docs/R17_IMPLEMENTATION_REPORT.md`, which states the conclusions I was
asked to test; `governance/GIT_ADAPTATION.md` including the row I was asked to challenge;
`OP-PROT-012`; the README, the research-map headings, and `OPEN_PROBLEMS.md` §D/§E/§F; and
`reviews/README.md` and `SCHEMA.md`, which quote R17 §4 verbatim. This session is also
downstream of the Anthropic lineage that produced that port.

What I had **not** seen, and what therefore carries the weight here: `r17_reference.py`,
`test_r17_reference.py`, `TESTS_NORMAL.txt`, `TESTS_OPTIMIZED.txt` and
`00_HANDOFF_SUMMARY_OPS4.json`. None of the five is reproduced anywhere in this repository.

The bias direction is toward agreement, and I did confirm the author's headline claim. The
counterweight applied throughout: every adverse finding below came out of executing
something, not out of reading.

The full disclosure is the `exposure_disclosure` field of the JSON record.

---

## 3. Obtaining the accompanying implementation

The object's closing sentence is its load-bearing self-claim:

> The accompanying local reference validator and adversarial tests demonstrate protocol
> predicates on specified inputs. They are not a deployed enforcement service and do not
> prove race freedom for Google Drive.

Those live inside `R17_NAVIGATION_REVIEW_AND_ROLLBACK.zip` (3,426,811 bytes), whose only
transport is base64 in a tool result — no byte range, no in-archive extraction. Rather than
record a capability blocker I used the accessibility reading copies that `Payloads.csv` lists
for each member, and **verified every recovered member against its recorded digest**:

| member | recorded bytes / sha256 | recovered |
|---|---|---|
| `r17_reference.py` | 5,637 / `74c7d83bea89eef4…` | MATCH |
| `test_r17_reference.py` | 5,804 / `35477c5b1cb5e544…` | MATCH |
| `TESTS_NORMAL.txt` | 1,370 / `1f9787fefa6d7b6f…` | MATCH |
| `00_HANDOFF_SUMMARY_OPS4.json` (loose file) | 1,114 / `905b6de5088ffab0…` | MATCH |

### 3.1 Finding OBJ-10 — the reading-copy channel is lossy

The reading volumes wrap each member in a header and `CONTENT START`/`CONTENT END` markers,
normalise line endings, and **do not preserve runs of blank lines**. No single extraction rule
worked across one carrier: the two `.py` members needed *strip trailing blanks, append two
newlines*; `TESTS_NORMAL.txt` needed *collapse blank-line runs, append one newline* (the `.py`
rule gave 1,373 bytes against a recorded 1,370).

I only knew which normalisation was right because the source map already carried the digest.
**For an object whose digest is not known in advance, this channel cannot establish exact
bytes** — which is precisely the precondition R17 §2 makes identity depend on. Severity
**MAJOR**, against the accessibility channel rather than against the object.

---

## 4. The argument, reconstructed

The object is policy in eight sections, so reconstruction means restating each normative rule
and then executing the accompanying predicates against adversarial inputs rather than reading
along with the author's tests.

§1 makes one entry point normative. §2 replaces a mandatory whole-Drive preflight with bounded
cached entry and makes identity digest-based; the handoff is an immutable
`{payload, payload_sha256}` envelope over a deterministic restricted JSON encoding with no
self-referential hash field. §3 makes coordination an append-only log ordered by *confirmed
append position*, with 120-minute leases and guarded publication in which two successors of one
head are held as a conflict for explicit reconciliation. §4 separates four dimensions and
permits a fresh nonauthor session of any provider at zero organizational independence. §5
separates request time from last substantive activity and makes 7/14/30-day aging a pickup
ladder that never approves. §6 gives five quarantine classifications plus a logical quarantine
keyed by carrier ID, relative path and hash, restorable by a new erratum or successor plus a
fresh review. §7 is the DRAFT → CANDIDATE_VERIFIED → READY_FOR_REVIEW → REVIEWED/AMEND ladder.
§8 rejects three of v1.0's four proposed guarantees.

`r17_reference.py` implements nine predicates. `handoff_status` is a guard chain — envelope
shape, recomputed digest, required fields, per-source identity, future check, then equality
against live policy/target/task/sources/head/claim, and **only then** age. Putting age last is
right and I confirmed it: a stale handoff whose head also moved reports `HEAD_CHANGED`, not
staleness. `winner` canonicalises events, treats a byte-identical repeat as idempotent and a
conflicting reuse as an error, bounds leases, validates heartbeats, and returns the eligible
claim with the smallest append row. `publication_conflicts` groups PUBLISH events by
`(target, old_head)`. `review_class` gates on disclosure and substantiveness before routing
authors and coauthors to `INTERNAL_VERIFICATION`. `quarantine_class` is a four-branch classifier.

### 4.1 The author's receipt, reproduced

```
$ python3 -m unittest test_r17_reference -v      # Python 3.11.15
... 13 tests ... OK
$ python3 -OO -m unittest test_r17_reference
... 13 tests ... OK
```

13 groups, all passing, under both interpreters. This confirms the implementation report's
claim. My verbose output is 1,370 bytes hashing to
`bf59fa4f75a60c9cf2563f0517db0a8dda9b6e1e30af18e54627130be8c5fe5c` — **byte for byte the
archive's `TESTS_OPTIMIZED.txt`**.

### 4.2 The shipped handoff, verified end to end

`00_HANDOFF_SUMMARY_OPS4.json` recomputes under the object's *own* canonical encoder to its
recorded `payload_sha256` `41e6b541cd1e5c7b216a3859d18745dafcd22d391da6d9602d1fd1d44d9c21bc`.
Its sole declared source is this object, by Drive ID, digest and byte count. Against a matching
live state it returns `FAST_ROUTE_VALID` at the checkpoint hour and `SCOPED_REVALIDATION` a day
later — §2 as written.

---

## 5. Negative controls — 15 mutants, 11 fired, 4 survived

Each control perturbs one constant, flips one inequality, or removes one hypothesis from the
byte-verified `r17_reference.py`, then re-runs the bundled suite.

| # | control | result |
|---|---|---|
| C1 | staleness threshold 7200 s → 3600 s | **SURVIVED** |
| C2 | staleness threshold 7200 s → 1800 s | fired (`test_fresh_and_old_route`) |
| C3 | `dt>7200` → `dt>=7200` | **SURVIVED** |
| C4 | future guard `dt<0` → `dt<=0` | **SURVIVED** |
| C5 | max lease 120 min → 240 min | **SURVIVED** |
| C6 | `min(eligible)` → `max(eligible)` | fired (`test_all_six_contender_orders`) |
| C7 | fork `len(v)>1` → `len(v)>0` | fired (`test_fork_detected_not_prevented`) |
| C8 | drop the coauthor flag | fired (`test_family_deadlock_and_coauthors`) |
| C9 | drop `substantive` from the review gate | fired (`test_family_deadlock_and_coauthors`) |
| C10 | EXACT_DUPLICATE without byte equality | fired (`test_quarantine_not_title_similarity`) |
| C11 | drop the 64-hex source digest check | fired (`test_missing_bad_future`) |
| C12 | stop comparing the payload digest | fired (`test_corruption_and_self_reference`) |
| C13 | allow extra envelope keys | fired (`test_corruption_and_self_reference`) |
| C14 | accept conflicting event-ID reuse | fired (`test_duplicate_event_ids`) |
| C15 | `org_independent` hard-wired True | fired (`test_family_deadlock_and_coauthors`) |

**The four that did not fire are a finding about the suite's sensitivity, not a pass.** C1 and
C2 bracket it: the bundled tests pin the 120-minute staleness threshold only to the interval
**(3600, 10801] seconds**, because they assert FAST at 3600 s and SCOPED at 10801 s and never
probe between. C3 shows the boundary itself is untested; C4 shows age zero is never exercised;
C5 shows no fixture uses a lease longer than 120 minutes, so the §3 lease cap is unpinned from
above. The suite is genuinely adversarial on **structure** — shapes, guards, orderings,
coauthorship, idempotence — and weak on the two **numeric constants** §2 and §3 actually state.

---

## 6. The four cases in the brief

### Case 1 — stale handoff

**Rule (§2).** "A 120-minute-old handoff triggers bounded revalidation, not an automatic
whole-Drive crawl. Missing fields, future timestamps, bad digests or conflicting pointers
invalidate fast continuation." Plus: "No self-referential hash field."

**Handled.** Tampering after hashing → `DIGEST_MISMATCH`. An extra self-referential hash field
→ `INVALID_ENVELOPE`. Uppercase digest, a bool byte count and a negative byte count are each
rejected as `BAD_SOURCE_IDENTITY`. Staleness is evaluated last, so a stale handoff whose world
also changed reports the change (`HEAD_CHANGED`, `CLAIM_INVALID`), not the age — the right
order, and the more useful answer.

**Not handled.**

* **OBJ-05 (MINOR).** §2 says *a 120-minute-old handoff triggers revalidation*. The code uses a
  strict `dt>7200`, so **at exactly 120 minutes the answer is `FAST_ROUTE_VALID`**. Ages 0, 119
  and 120 minutes all take the fast route; 121 does not. Off by one second in the permissive
  direction, and C1/C3 show the suite pins neither the constant nor the inequality. MINOR
  because the guards that actually protect correctness all run first.
* **OBJ-06 (MINOR).** §2 makes *extraction rule* a required source identity. The code checks
  the key is present, never that it says anything: a source with `'extraction': ''` returns
  `FAST_ROUTE_VALID`. The strictness exists elsewhere in the same function and is simply absent
  here.

### Case 2 — same-family coauthor

**Rule (§4).** Same provider is zero organizational independence, not a prohibition. A different
provider alone establishes nothing "if it coauthored the target or reused the same reasoning."
"Existing independent verdicts retain their exact scope; do not award a second credit to the
same lineage."

**Handled.** A fresh same-provider session → `NONAUTHOR_TECHNICAL_REVIEW` with
`org_independent: False` — the deadlock-breaking case R17 exists to permit. The author session
itself → `INTERNAL_VERIFICATION`. A flagged coauthor of a *different* provider →
`INTERNAL_VERIFICATION`, `org_independent: False`. Undisclosed or non-substantive →
`CANNOT_VERIFY`. Controls C8, C9 and C15 all fire, so these are pinned.

**Not handled.**

* **OBJ-03 (MAJOR) — vacuous independence.** Independence is computed as set non-membership
  against a possibly empty provider list, so with an **unknown author lineage it returns
  `org_independent: True`**:
  `review_class([], [], 'fresh', 'Anthropic', True, True) → {'technical': 'NONAUTHOR_TECHNICAL_REVIEW', 'org_independent': True}`.
  **Eleven** of the 24 Review Queue rows are in exactly that state — RV-DQ-005, -017, -020,
  -023, -027, -035, -038, -053, -057, -061 and -096 each carry "Disclose exact-object
  authorship; provider alone is insufficient" in the Author/provider column, and thirteen
  rows carry an unresolved `Body SHA-256`. The bundled suite tests
  the empty-author case only with `substantive=False`, where the gate short-circuits, so this
  branch is never reached. Worth noting: **this repository's own `tools/reviews_check.py` is
  stricter than the protocol's reference implementation here**, refusing any nonzero credit
  against an `unknown` author lineage.
* **OBJ-04 (MAJOR) — two §4 sentences have no implementation.** `review_class` takes no
  prior-credit or lineage argument, so two successive Anthropic reviewers of one OpenAI object
  each receive `org_independent: True` — the second credit to one lineage that §4 forbids. And
  only *coauthorship* has a parameter; **"reused the same reasoning" has none**, so a
  different-provider reviewer replaying the author's derivation is indistinguishable from a
  genuinely separate one.
* **OBJ-07 (MINOR) — provider names are exact case-sensitive strings.** Against author
  `['OpenAI']`, the spellings `'openai'`, `'OPENAI'`, `'OpenAI '` and `'Open AI'` each return
  `org_independent: True`. Only the byte-identical spelling returns False. No normalisation, no
  canonical vocabulary, no bundled test that varies the spelling.

### Case 3 — publication race

**Rule (§3).** "Multiple successors of one old head are a publication conflict: hold only that
target, preserve both branches, and append a reconciliation selecting or merging them with
reasons. Never resolve it by silently overwriting proof bytes." And immediately before moving a
pointer, "recheck source/head/claim."

**Handled.** Two distinct successors of one head are reported as a conflict; a replayed
identical publish is not. Control C7 fires. `winner`'s ordering by confirmed append position is
pinned by C6 across all six permutations of three contenders.

**Not handled.**

* **OBJ-08 (MAJOR) — detection keyed on a self-reported field.** `publication_conflicts` groups
  by the publisher's **self-reported `old_head`**, which is exactly the field a racing or stale
  writer gets wrong. Change only the second event's `old_head` to an incorrect value and the
  real fork **disappears**: `{('X','h0'): ['a','b']}` becomes `{}`. §3 itself insists on "the
  confirmed append position, not a self-reported timestamp"; the same distrust is not applied
  to `old_head`.
* **OBJ-08b (MAJOR) — claims and publications never meet.** No predicate joins `winner()` to
  PUBLISH events. With a valid claim held by `s1`, a lone PUBLISH from `s9` is flagged by
  nothing: `publication_conflicts` returns `{}` and `winner` ignores PUBLISH events entirely.
  §3's "recheck … claim" is unmodelled.
* **OBJ-08c (MINOR) — nothing requires the reconciliation.** There is no `RECONCILE` event type
  and no predicate that a detected fork was ever resolved; a fork plus a reconciliation event is
  reported identically to the unreconciled fork.

**The object is honest about this.** §3 says plainly that Sheets batch updates are "**not a
compare-and-swap lock across clients**", that this is "cooperative coordination, not a hard
distributed mutex", and that "if a strict machine-enforced lock is required, a
transaction-capable service is still needed; none is represented as installed here." The
docstring in the code says "Detect forks; no assertion that a read/write race was prevented."
These are coverage gaps against §3's own sentences, not false claims by the object.

### Case 4 — restoration

**Rule (§6).** Frozen members that cannot be moved get a **logical quarantine keyed by carrier
ID plus relative path and hash**; their bytes and old manifests stay intact; "A new
erratum/successor and fresh review restore eligibility"; "Do not alter a frozen Merkle tree to
make history disappear"; "No permanent deletion in this workflow."

**Handled.** Classification only, and correctly: identical bytes → `EXACT_DUPLICATE`; bytes
differing only by CRLF → `UNVERIFIED` (§6's "same titles, similar topics" trap avoided, and C10
pins it); a concrete counterexample → `DEFECTIVE_SCOPE`; a numbered successor → `SUPERSEDED`.

**Not handled — this is the weakest of the four.**

* **OBJ-02 (MAJOR) — there is no restoration predicate at all.** The module's entire public
  surface is `canonical, digest, envelope, handoff_status, instant, publication_conflicts,
  quarantine_class, review_class, winner`. **Not one** accepts an erratum, a successor, a fresh
  review, a move record, or a carrier-plus-path-plus-hash exclusion key. The single bundled
  quarantine test exercises classification. **The restoration case the brief names is exercised
  by nothing.**
* **OBJ-02b (MAJOR) — the fifth classification is unreachable.** Enumerating `quarantine_class`
  over every argument combination yields exactly four outputs:
  `['DEFECTIVE_SCOPE', 'EXACT_DUPLICATE', 'SUPERSEDED', 'UNVERIFIED']`. **`LEGACY_INSPIRATION`
  is never returned**, and §6's "UNVERIFIED / CONFLICT" collapses to `UNVERIFIED` with no way to
  say which. This corroborates, from the implementation side, the **pre-existing** repository
  finding in `docs/OPEN_PROBLEMS.md` §F that three live Quarantine Index rows use an
  `EXISTING_CONTAINER` class §6 does not define. That observation is not mine; my probe
  rediscovered the same gap from the other direction.
* **OBJ-11 (MINOR) — a present but empty defect is not a defect.** The classifier branches on
  truthiness, so `quarantine_class(b'p', b'p', defect='')` returns `EXACT_DUPLICATE`. §6 requires
  a "concrete failed statement, counterexample or reproducible invalid certificate chain" — a
  content requirement the code does not make.

---

## 7. Finding OBJ-12 — the GIT_ADAPTATION publication-race claim is overstated

`governance/GIT_ADAPTATION.md` asserts:

> Git gives the transactional guarantee the protocol says Sheets cannot: a push that races is
> rejected, never silently overwritten.

and, for the publication-conflict row, "Merge conflict; resolved by a reconciliation commit that
preserves both branches in history | **Same semantics, machine-enforced**."

*This is a finding about a repository document, not about the reviewed object, which claims
nothing about git.*

The narrow kernel is true: a non-fast-forward update of one ref is an atomic compare-and-swap
and is rejected. Three qualifications defeat the claim as written.

1. **The row contradicts the row above it.** The same table prescribes "A branch per claim; the
   PR is the claim." Under branch-per-claim, two contenders push **different refs**. Nothing
   races, nothing is rejected, and the fork R17 §3 describes is created with no git objection at
   all. The CAS fires only in the same-ref case, which this architecture is designed to avoid.
2. **The rejection is defeasible, and here it is not even configured.** `--force`,
   `--force-with-lease`, and delete-then-recreate all overwrite. GIT_ADAPTATION itself says
   branch protection "*should*" forbid force-push on `main` — aspirational, not installed. In
   this repository:
   ```
   receive.denyNonFastForwards = (unset)
   receive.denyDeletes         = (unset)
   .github/                    = pull_request_template.md, workflows/ci.yml
   ```
   No force-push or branch-protection configuration is present.
3. **"Same semantics, machine-enforced" does not hold.** Two successors of one head editing
   *disjoint* regions of a proof body auto-merge with **no conflict raised**:
   ```
   $ git merge-file -p merged.txt base.txt theirs.txt
   lemma A: bound is 2C r^3          <- from one successor
   lemma B: kappa = 1/3
   lemma C: domain is the FULL torus <- from the other
   exit 0
   ```
   The result is a third body that **neither claimant published** — silent combination, in place
   of the hold-and-reconcile §3 requires. Overlapping edits do return exit 1.

**Recommendation:** restate the row as *git strengthens the same-ref case and leaves the
cross-ref and auto-merge cases exactly as cooperative as the protocol says they are.*

**Limit on this finding.** I did **not** execute a live push race: this session was instructed
not to run `git add`, `git commit` or `git push`. The CAS half is assessed from git's documented
semantics plus this repository's actual configuration. The auto-merge half is executed, above,
with read-only plumbing.

---

## 8. Verdict

**`AMEND`** — from the register's own R17 status set.

It is **not** a finding that the object is wrong. Every statement the object makes about itself
that I could check held: the validator and tests exist, are byte-identifiable, and pass; the
shipped handoff satisfies the object's own noncircular digest rule; §3 and §8 deny the very
guarantees a careless reader might infer. Its closing scope clause — "protocol predicates **on
specified inputs**" — is exactly accurate.

AMEND records that **three protocol rules the object itself states have no implementation in the
bundle the object points at** — §6 restoration of eligibility, §4's prohibition on a second
credit to one lineage, and §4's reused-reasoning qualifier — and that **two shipped predicates
admit adversarial inputs they should reject**: fork detection keyed on a self-reported
`old_head`, and independence awarded vacuously against an empty author-provider list.

Amendment means extending the reference implementation and its tests to those cases, **or**
narrowing the object's final paragraph to name which predicates the bundle demonstrates and
which it does not.

### Independence

`independence_credit: 0`, and the reason is unusual for this directory. **The author family is
openai and mine is anthropic — a different family — so R17 §4's same-provider rule is not what
makes this zero.** The credit is zero because the predicate that would license a nonzero one
fails: R17 §4 ("Different provider alone does not establish independence if it coauthored the
target or reused the same reasoning") together with OP-PROT-012 §5, of whose ten items at least
four fail outright — (b) no frozen task specification before execution; (c) I had full access to
the author's implementation report and to this repository's derived port before any freeze of my
own, which is exactly the access (c) forbids; (e) no hash frozen before comparison; (i) lineage
disclosure shows this session is downstream of an Anthropic re-expression of the object.

**The external-independence predicate on this route REMAINS OPEN.** The row reads
`EXTERNAL_REVIEW_OPEN` and is exactly as open after this record as before it. No technical
verdict bears on it. R17 §4: "A task may finish its technical review while an
external-independence predicate remains open."

### Unresolved dependencies

Both carrier archives (I verified three members against recorded digests but never opened either
carrier); all 12 members of `R17_FINAL_VERIFICATION.zip`; `R17_FINAL_CUSTODY_RECEIPT.json`, which
the handoff's `verified_boundary` cites; the move log, quarantine index, review-queue and catalog
CSVs; **every quantitative figure in the implementation report** (the 90.76% reduction, 2,943
catalog entries, 28 memberships, 17 quarantine rows, 11 H5 exclusions); the "installed daily
automation" §5 describes, which I never observed and under which 23 of the 24 routes remain UNASSIGNED; the
live Drive surfaces §1 makes normative; the object's Drive revision history; OP-PROT-011,
OP-GDN-002 and OP-CNS-001; and a live push race. The JSON record lists these in full.

### What this review does not establish

The full statement is the `does_not_establish` field. In brief: no gate moves and no register is
edited; no independence is created and this is not partial progress toward the external-review
obligation; AMEND is not a defect in the policy's reasoning; passing the bundled suite
establishes nothing about Google Drive, since the suite has no Drive call, no Sheets call, no
concurrency and no clock; the four cases were answered only in the reference implementation's
terms, never in the deployed workflow; verifying digests proves which bytes exist, not that any
live worker ever loaded them; nothing mathematical is touched, the 2D upper and 3D lifetime
tracks are not composed anywhere in this work, and the count of original prize problems solved
is unchanged at zero; and age approves nothing.

### Finding OBJ-13 — `docs/OPEN_PROBLEMS.md` §D is off by one

The §D heading reads "Review queue — 24 routes, **all unassigned**". Twenty-three are.
`RV-LM009-MAIN` is not: its `Reviewer / claim` column reads `OPS4 nonauthor / exposed / OpenAI`,
its Aging action is `EXTERNAL ONLY` rather than `ESCALATE`, and its technical status is
`PASS_TECHNICAL`. The register is correct; the prose summary of it is not.

This is worth flagging because the distinction is the one R17 §4 turns on. That row already
records a same-provider nonauthor technical pass with its organizationally-distinct verdict
still open — precisely the state this record is in. An "all unassigned" summary erases the
difference between a route nobody has reviewed and a route with a zero-credit technical verdict
already attached, which is the same erasure R17 §5 warns against when it says the July items are
"*unresolved review obligations*, not necessarily work no one has ever reviewed".

Reported, not repaired: `docs/` is outside the paths this review owns.

### A note on filenames

The task assigned `reviews/records/RV-OPS-R17*.json`, but `review_record.schema.json` requires
`review_id` to match `^REV-` and `tools/reviews_check.py` requires the filename stem to equal
`review_id`. No name satisfies both. I followed the machine-checked schema — `REV-OPS-R17-001` —
and touched nothing else under `reviews/`. The `route_key` field carries the register's actual
key, `RV-OPS-R17`, so the record still binds to the right row.

```
$ python3 tools/reviews_check.py
reviews_check: 1 record(s) in reviews/records, 0 problem(s).
reviews_check: form only. No mathematics verified, no independence credit awarded, no gate moved.
EXIT=0
```

---

## 9. Reproduction

Everything above is reproducible from these three scripts plus the two byte-verified members.
`extract.py` recovers a sealed member from the connector's saved response and **exits non-zero
unless the recovered bytes hash to the member digest**, so a failed recovery cannot be mistaken
for a successful one.

### `extract.py` — recover an exact archive member, digest-verified

Used for `r17_reference.py` and `test_r17_reference.py`. `TESTS_NORMAL.txt` needs the
blank-line-collapsing variant described in §3.1 — that is finding OBJ-10.

```python
#!/usr/bin/env python3
"""Extract an exact archive-member body from an ACCESS READING VOLUME export.

Usage: extract.py <tool-result-json> <member-sha256> <out-path>
Extraction rule (derived, then digest-verified):
  base64-decode result["content"] -> UTF-8 text with BOM and CRLF
  -> locate 'CONTENT START <sha>-part<N>' .. 'CONTENT END <sha>-part<N>'
  -> join parts, normalize CRLF->LF, strip trailing blank lines,
  -> append exactly two LF.
Exits non-zero unless sha256(out) == <member-sha256>.
"""
import sys, json, base64, hashlib, re

res, want, out = sys.argv[1], sys.argv[2], sys.argv[3]
raw = base64.b64decode(json.load(open(res))["content"]).decode("utf-8-sig")
lines = raw.replace("\r\n", "\n").split("\n")
parts, cur, n = [], None, 0
for l in lines:
    if l.startswith("CONTENT START " + want):
        cur = []
    elif cur is not None and l.startswith("CONTENT END " + want):
        parts.append(cur); cur = None; n += 1
    elif cur is not None:
        cur.append(l)
if not parts:
    sys.exit("no CONTENT block for " + want)
body = [l for p in parts for l in p]
while body and body[-1] == "":
    body.pop()
blob = ("\n".join(body) + "\n\n").encode("utf-8")
got = hashlib.sha256(blob).hexdigest()
open(out, "wb").write(blob)
print(f"parts={n} bytes={len(blob)} sha256={got} want={want} {'MATCH' if got==want else 'MISMATCH'}")
sys.exit(0 if got == want else 1)
```

### `negative_controls.py` — the 15-mutant battery of §5

```python
#!/usr/bin/env python3
"""Negative controls for RV-OPS-R17: mutate r17_reference.py, re-run the
bundled adversarial suite, and record which controls FIRE (suite fails) and
which SURVIVE (suite still passes -> the suite does not pin that behaviour)."""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = open(os.path.join(HERE, "r17_reference.py"), encoding="utf-8").read()
TEST = os.path.join(HERE, "test_r17_reference.py")

MUTANTS = [
 ("C1 constant: 120-min staleness threshold 7200 -> 3600",
  [("if dt>7200 else", "if dt>3600 else")]),
 ("C2 constant: staleness threshold 7200 -> 1800",
  [("if dt>7200 else", "if dt>1800 else")]),
 ("C3 inequality: dt>7200 -> dt>=7200 (boundary inclusive)",
  [("'SCOPED_REVALIDATION' if dt>7200", "'SCOPED_REVALIDATION' if dt>=7200")]),
 ("C4 inequality: future-checkpoint dt<0 -> dt<=0",
  [("if dt<0:return 'FUTURE_CHECKPOINT'", "if dt<=0:return 'FUTURE_CHECKPOINT'")]),
 ("C5 constant: max lease 120 min -> 240 min (both sites)",
  [("datetime.timedelta(minutes=120)", "datetime.timedelta(minutes=240)")]),
 ("C6 inequality: earliest append position min() -> max()",
  [("return min(eligible)[1] if eligible else None", "return max(eligible)[1] if eligible else None")]),
 ("C7 inequality: fork detection len(v)>1 -> len(v)>0",
  [("if len(v)>1", "if len(v)>0")]),
 ("C8 hypothesis removed: drop the coauthor flag from review_class",
  [("if coauthor or reviewer_session in author_sessions",
    "if reviewer_session in author_sessions")]),
 ("C9 hypothesis removed: drop 'substantive' from the review gate",
  [("if not disclosed or not substantive", "if not disclosed")]),
 ("C10 hypothesis removed: EXACT_DUPLICATE without byte equality",
  [("if b is not None and a==b", "if b is not None")]),
 ("C11 hypothesis removed: drop the source sha256 format check",
  [("if not re.fullmatch('[0-9a-f]{64}',s['sha256']) or type(s['bytes']) is not int or s['bytes']<0",
    "if type(s['bytes']) is not int or s['bytes']<0")]),
 ("C12 hypothesis removed: stop comparing payload digest to envelope digest",
  [("if digest(p)!=obj['payload_sha256']:return 'DIGEST_MISMATCH'", "pass")]),
 ("C13 hypothesis removed: allow extra envelope keys (self-referential hash)",
  [("if set(obj)!={'payload','payload_sha256'}:return 'INVALID_ENVELOPE'",
    "if not {'payload','payload_sha256'}<=set(obj):return 'INVALID_ENVELOPE'")]),
 ("C14 inequality: duplicate-ID equality check inverted to accept conflicts",
  [("if canonical(e)!=seen[ident]:raise ValueError('conflicting event ID')", "pass")]),
 ("C15 constant: independence by provider -> always independent",
  [("'org_independent':reviewer_provider not in author_providers", "'org_independent':True")]),
]

def run(src):
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "r17_reference.py"), "w", encoding="utf-8").write(src)
        shutil.copy(TEST, os.path.join(d, "test_r17_reference.py"))
        p = subprocess.run([sys.executable, "-m", "unittest", "test_r17_reference"],
                           cwd=d, capture_output=True, text=True)
        return p.returncode, (p.stderr or "").strip().splitlines()
    finally:
        shutil.rmtree(d)

rc, _ = run(SRC)
print(f"baseline (unmutated): {'PASS' if rc==0 else 'FAIL'}\n")
fired = survived = 0
for name, subs in MUTANTS:
    m = SRC
    for a, b in subs:
        if a not in m:
            print(f"  !! pattern not found: {a!r}"); sys.exit(2)
        m = m.replace(a, b)
    assert m != SRC
    rc, err = run(m)
    if rc != 0:
        fired += 1
        why = next((l for l in err if l.startswith(("FAIL:", "ERROR:"))), "")
        print(f"FIRED    {name}\n         first failing test: {why}")
    else:
        survived += 1
        print(f"SURVIVED {name}\n         bundled suite still passes -> behaviour not pinned")
print(f"\n{fired} fired, {survived} survived, {len(MUTANTS)} controls")
```

### `adversarial_cases.py` — the four cases of §6

```python
#!/usr/bin/env python3
"""RV-OPS-R17 adversarial probes for the four cases named in the review brief:
stale handoff, same-family coauthor, publication race, restoration.
Each probe states the protocol rule, constructs the scenario, and prints what
the byte-verified reference implementation actually returns."""
import importlib.util, os, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r17", os.path.join(HERE, "r17_reference.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)

def hdr(n, t): print(f"\n{'='*70}\nCASE {n}: {t}\n{'='*70}")
def p(label, got, expect=None):
    verdict = "" if expect is None else ("  <-- as protocol requires" if got == expect else "  <-- DIVERGES from protocol text")
    print(f"  {label:<58} {got!r}{verdict}")

PAY = {'protocol_version':'R17','checkpoint_utc':'2026-09-17T11:00:00Z','task_id':'T',
       'claim_id':'A','target_key':'X',
       'sources':[{'drive_id':'d','sha256':'a'*64,'bytes':3,'extraction':'raw'}],
       'output_head':'old','verified_boundary':'b','next_action':'n'}
LIVE = {**PAY, 'claim_valid': True}

hdr(1, "STALE HANDOFF  (R17 sec.2: 'A 120-minute-old handoff triggers bounded "
        "revalidation'; 'Missing fields, future timestamps, bad digests or "
        "conflicting pointers invalidate fast continuation')")
env = R.envelope(PAY)
for mins, now in [(0,'2026-09-17T11:00:00Z'), (119,'2026-09-17T12:59:00Z'),
                  (120,'2026-09-17T13:00:00Z'), (121,'2026-09-17T13:01:00Z')]:
    p(f"age {mins} min", R.handoff_status(env, LIVE, now))
print("  PROBE 1.1 the 120-minute boundary itself is inside the FAST route:")
print("            at exactly 120 min the reference returns FAST_ROUTE_VALID,")
print("            i.e. 'a 120-minute-old handoff' does NOT trigger revalidation.")

print("\n  PROBE 1.2 staleness is evaluated LAST, so a stale handoff whose world")
print("            also changed reports the change, not the age:")
p("stale AND head moved", R.handoff_status(env, {**LIVE,'output_head':'new'}, '2026-09-18T11:00:00Z'), 'HEAD_CHANGED')
p("stale AND claim dead", R.handoff_status(env, {**LIVE,'claim_valid':False}, '2026-09-18T11:00:00Z'), 'CLAIM_INVALID')

print("\n  PROBE 1.3 self-referential / tampered envelopes:")
bad = R.envelope(PAY); bad['payload']['next_action'] = 'tampered'
p("payload edited after hashing", R.handoff_status(bad, LIVE, '2026-09-17T12:00:00Z'), 'DIGEST_MISMATCH')
bad = R.envelope(PAY); bad['self_hash'] = 'x'
p("extra self-referential hash field", R.handoff_status(bad, LIVE, '2026-09-17T12:00:00Z'), 'INVALID_ENVELOPE')

print("\n  PROBE 1.4 identity smuggling through a source entry:")
for label, src in [("uppercase digest", {'drive_id':'d','sha256':'A'*64,'bytes':3,'extraction':'raw'}),
                   ("bytes given as bool True", {'drive_id':'d','sha256':'a'*64,'bytes':True,'extraction':'raw'}),
                   ("negative byte count", {'drive_id':'d','sha256':'a'*64,'bytes':-1,'extraction':'raw'}),
                   ("empty extraction rule", {'drive_id':'d','sha256':'a'*64,'bytes':3,'extraction':''})]:
    q = {**PAY, 'sources':[src]}
    p(label, R.handoff_status(R.envelope(q), {**LIVE,'sources':[src]}, '2026-09-17T12:00:00Z'))
print("  NOTE: an EMPTY extraction rule is accepted. R17 sec.2 makes 'extraction rule'")
print("        a required source identity; the reference checks presence, not content.")

hdr(2, "SAME-FAMILY COAUTHOR  (R17 sec.4: same provider is zero organizational "
        "independence; a different provider alone establishes nothing if it "
        "coauthored the target or reused the same reasoning; 'do not award a "
        "second credit to the same lineage')")
p("fresh OpenAI session reviewing an OpenAI object",
  R.review_class(['author'],['OpenAI'],'fresh','OpenAI',True,True))
p("the author session itself",
  R.review_class(['author'],['OpenAI'],'author','OpenAI',True,True))
p("foreign provider, flagged coauthor",
  R.review_class(['author'],['OpenAI'],'foreign','Anthropic',True,True,True))
p("foreign provider, not a coauthor",
  R.review_class(['author'],['OpenAI'],'foreign','Anthropic',True,True))

print("\n  PROBE 2.1 UNRESOLVED AUTHORSHIP. 4 of the 24 Review Queue rows carry")
print("            'Disclose exact-object authorship; provider alone is insufficient'.")
print("            Model that as an empty author list and ask for independence:")
r = R.review_class([], [], 'fresh', 'Anthropic', True, True)
p("author lineage unknown, substantive+disclosed", r)
print(f"            org_independent == {r['org_independent']}: independence is awarded")
print("            VACUOUSLY, because 'Anthropic' is not in the empty provider list.")

print("\n  PROBE 2.2 SECOND CREDIT TO ONE LINEAGE (R17 sec.4 forbids it):")
a = R.review_class(['auth'],['OpenAI'],'anthropic-session-1','Anthropic',True,True)
b = R.review_class(['auth'],['OpenAI'],'anthropic-session-2','Anthropic',True,True)
p("first Anthropic reviewer", a['org_independent'])
p("second Anthropic reviewer, same lineage", b['org_independent'])
print("            Both True. review_class takes no prior-credit argument, so the")
print("            'do not award a second credit to the same lineage' rule has no")
print("            representation in the reference implementation at all.")

print("\n  PROBE 2.3 PROVIDER NAME IS AN EXACT, CASE-SENSITIVE STRING:")
for prov in ['OpenAI', 'openai', 'OPENAI', 'OpenAI ', 'Open AI']:
    p(f"author=['OpenAI'], reviewer provider {prov!r}",
      R.review_class(['a'],['OpenAI'],'fresh',prov,True,True)['org_independent'])
print("            Any spelling drift silently converts zero independence into one.")

print("\n  PROBE 2.4 'REUSED THE SAME REASONING' has no parameter. A reviewer of a")
print("            different provider who replays the author's own derivation is")
print("            indistinguishable from a genuinely independent one:")
p("different provider, reused reasoning, not flagged coauthor",
  R.review_class(['a'],['OpenAI'],'fresh','Anthropic',True,True)['org_independent'])

hdr(3, "PUBLICATION RACE  (R17 sec.3: 'Multiple successors of one old head are a "
        "publication conflict: hold only that target, preserve both branches, and "
        "append a reconciliation'; 'recheck source/head/claim' before a pointer move)")
A = dict(type='PUBLISH', target='X', old_head='h0', new_head='a')
B = dict(type='PUBLISH', target='X', old_head='h0', new_head='b')
p("two successors of one head", R.publication_conflicts([A,B]), {('X','h0'):['a','b']})
p("the same publish replayed", R.publication_conflicts([A,A]), {})

print("\n  PROBE 3.1 MISREPORTED old_head DEFEATS DETECTION. The detector keys on the")
print("            publisher's SELF-REPORTED old_head, the one thing a racing writer")
print("            gets wrong. Second writer raced from h0 but records h_wrong:")
Bx = {**B, 'old_head':'h_wrong'}
p("real fork, second old_head misreported", R.publication_conflicts([A,Bx]), {('X','h0'):['a','b']})
print("            R17 sec.3 says to order contenders by 'the confirmed append position,")
print("            not a self-reported timestamp'. The same distrust is not applied")
print("            to the self-reported old_head, and the fork disappears.")

print("\n  PROBE 3.2 A PUBLISHER WITH NO CLAIM IS NEVER FLAGGED:")
claim = dict(id='C1', session='s1', target='X', type='CLAIM',
             utc='2026-09-17T11:00:00Z', expiry='2026-09-17T13:00:00Z')
print(f"            winner() = {R.winner([claim],'X','2026-09-17T12:00:00Z')!r} (session s1 holds the claim)")
rogue = dict(type='PUBLISH', target='X', old_head='h0', new_head='rogue', session='s9')
p("lone publish by a session holding no claim", R.publication_conflicts([rogue]), "flagged: publisher holds no claim")
print("            No predicate joins winner() to PUBLISH events, so the sec.3 rule")
print("            'recheck ... claim' immediately before a pointer move is unmodelled.")

print("\n  PROBE 3.3 NOTHING REQUIRES THE RECONCILIATION. There is no RECONCILE event")
print("            type and no predicate that a detected fork was ever resolved:")
rec = dict(type='RECONCILE', target='X', old_head='h0', new_head='a')
p("fork plus a reconciliation event", R.publication_conflicts([A,B,rec]), "reconciled: no longer outstanding")
print("            The fork is reported identically before and after reconciliation.")

hdr(4, "RESTORATION  (R17 sec.6: logical quarantine keyed by carrier ID + relative "
        "path + hash; 'A new erratum/successor and fresh review restore "
        "eligibility'; 'No permanent deletion in this workflow')")
p("identical bytes", R.quarantine_class(b'proof', b'proof'), 'EXACT_DUPLICATE')
p("bytes differing only by CRLF", R.quarantine_class(b'proof', b'proof\r\n'), 'UNVERIFIED')
p("concrete counterexample", R.quarantine_class(b'p', defect='counterexample'), 'DEFECTIVE_SCOPE')
p("numbered successor", R.quarantine_class(b'p', superseded=True), 'SUPERSEDED')

print("\n  PROBE 4.1 THE FIFTH CLASSIFICATION IS UNREACHABLE. R17 sec.6 tabulates five")
print("            classifications; quarantine_class can return only four:")
outs = set()
for b_ in (None, b'p', b'q'):
    for d_ in (None, '', 'defect'):
        for s_ in (False, True):
            outs.add(R.quarantine_class(b'p', b_, d_, s_))
print(f"            reachable outputs over every argument combination: {sorted(outs)}")
print("            LEGACY_INSPIRATION is never returned, and 'UNVERIFIED / CONFLICT'")
print("            collapses to 'UNVERIFIED' with no way to say which.")

print("\n  PROBE 4.2 THERE IS NO RESTORATION PREDICATE. The module's public surface:")
print(f"            {sorted(n for n in vars(R) if not n.startswith('_') and callable(vars(R)[n]))}")
print("            Nothing accepts an erratum, a successor, a fresh review, a move")
print("            record, or a carrier-ID+path+hash exclusion key. Restoration of")
print("            eligibility, the sec.6 rule the brief asks about, is not implemented,")
print("            and no bundled test exercises it.")

print("\n  PROBE 4.3 AN EMPTY-STRING DEFECT IS NOT A DEFECT:")
p("defect='' (present but falsy)", R.quarantine_class(b'p', b'p', defect=''), 'DEFECTIVE_SCOPE')
print("            A recorded-but-empty defect field silently downgrades to EXACT_DUPLICATE.")
```

### Running them

```bash
# with r17_reference.py and test_r17_reference.py (digest-verified) in the cwd
python3 -m unittest test_r17_reference -v     # 13 groups OK
python3 -OO -m unittest test_r17_reference    # 13 groups OK
python3 negative_controls.py                  # 11 fired, 4 survived
python3 adversarial_cases.py                  # the four cases

# in the repository
python3 tools/drive_index.py id 1hBph5Fpxd5dVrolNkvzU8nb7xpxUiUdc
python3 tools/drive_index.py archive R17_NAVIGATION_REVIEW_AND_ROLLBACK
sha256sum governance/protocols/OP-PROT-019-v1.1_R17.md   # efcfdd5c… — NOT the object
python3 tools/reviews_check.py
```
