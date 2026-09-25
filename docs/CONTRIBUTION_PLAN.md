# Where I can contribute

Written after reading the full Drive corpus: the 4,456-item source map, the 44-tab register export (2026-09-18 xlsx; the
2026-09-17 rendering had 42 tabs, seven of them truncated), the governance canon, the frozen Q0 core's navigation
structure, the D1 assembly layers, the RN3 and RN5 releases, the LPW fold
dispositions, the prize registry, and the 2026-09-17 accessibility publication.

Ordered by value, not by ease. Every item names the exact object it touches.

---

## 1. Turn the discipline into machine-checked invariants (started here)

The program's integrity currently rests on prose repeated across hundreds of
documents: *packaging is not premise discharge*, *display is not a certified
enclosure*, *do not compose 2D and 3D*, *session CLOSE is not lemma close*,
*0 prizes solved*. That prose is excellent, and it cannot fail a build.

`claims/graph.json` plus `tools/claims_check.py` encode the claim/premise
structure and five firewalls as data and assertions. `tests/test_claims.py`
carries eight negative controls proving the checker rejects each violation —
including the exact promotion the HOLD checklist forbids. Writing those
controls immediately found a real bug in my own checker, which is the point.

**Next:** extend the graph to the LM003…LM013 lemma stack so `RV-LM011`'s
synthesis precondition ("first obtain LM003, LM004-v1.1, LM006, LM009,
LM010-v1.1, LM012-v1.1, LM013 Carrier-B-v1.1 plus the joint stack") is a
computed, not remembered, fact.

## 2. Convert each found defect into a permanent regression test

**Status 2026-09-18.** Two of the three "next" items below are done and pinned:
the LPW headline mutation is `research/lpw/headline.py` (the delivered
`6.239e−44` is admissible as an upper bound and the proposed `6.238e−44` is
not; under an equality headline nothing rounded is admissible because the
expansion never terminates), and the false identity is
`research/identities/gaussian_moments.py` (`12 + 32/π`, one dropped cross term
worth `16/π`). The H5 totals rung-scope test is still open. See
`docs/FINDINGS_2026-09-18.md` §2.

`OP-GDN-002 §8` asks for exactly this: every serious error should be examined
for a reusable control. The corpus has several structural error classes that
recurred, and none of them is currently pinned by an executable test.

Done here: **RN5's determinant-moment defect.**
`research/rn/moment_envelope.py` reimplements the correct Hölder(4, 4, 2)
envelope, the defective `envelope_v` expression, and the typed Gaussian
counterexample over exact rationals, deriving the values from the moment
formulas rather than copying RN5's numbers — and reproducing them:

| | |
|---|---|
| typed expectation | ≥ 0.207675035568 |
| defective `envelope_v` | ≈ 0.062504062490 |
| correct Hölder(4, 4, 2) | ≈ 0.250004625009 |

`tests/test_rn_moment_envelope.py` asserts the separation by exact rational
comparison of fourth powers. The wrong-power bound cannot come back silently.

**Same treatment, status:**

* the **H5 totals contamination** — a rung-unscoped merge of `r = 0.025` patches
  into `r = 0.05` corrupted `I_lo` in v2. A test that fails when a totals merge
  crosses rung scope would have caught it at the merge, not at the errata.
* the **LPW headline mutation** — done, pinned by
  [`research/lpw/headline.py`](../research/lpw/headline.py). The delivered
  `6.239e−44` is admissible as an upper bound and the proposed `6.238e−44` is
  not; under an equality headline nothing rounded is admissible because the
  expansion never terminates. This is a regression test only. It does not
  discharge the R05 review and does not freeze the headline.
* the **false identity** `E(|ξ|+|η|)⁴ = 12 + 16/π` — done, **REFUTED**. For iid
  `N(0,1)` the exact value is `12 + 32/π`, because one of the two equal `16/π`
  cross terms was dropped; that exact value is the counterexample. Derivation:
  [`research/identities/gaussian_moments.py`](../research/identities/gaussian_moments.py)
  (see [`research/identities/README.md`](../research/identities/README.md)),
  source-bound at `5306b9dce91352e84effe34f61ff7d9aee605d77`.
  Owning record: the `LPW_CONSTANT` row, quoted under
  [What the sources say](../research/lpw/README.md#what-the-sources-say).
  The historical carrier
  [`engine/carriers/blobs/e258322cfbb71dbb__lpw_constant.py`](../engine/carriers/blobs/e258322cfbb71dbb__lpw_constant.py)
  is frozen and unedited. Complement: `12 + 16/π` remains a valid upper budget
  only for the different variable `ρ = √(ξ²+η²)`, where `E ρ⁴ = 8`
  ([`docs/LPW_AMPLITUDE.md`](LPW_AMPLITUDE.md)).

## 3. Close `OBL-H5-JETMOD` — the named blocking proof step

This is the chart-side lead and the sources name the proof step precisely:
evaluate the lattice sums with `r` as an **interval over the band**, yielding
G12-band enclosures, hence `Î(r)/r³ ≤ F(G12-band)` for the whole band — a finite
computation per band, never a fitted exponent. The falsifier is equally precise:
a band enclosure wider than the claimed modulus.

I can implement the interval-`r` lattice-sum evaluator for the full 24-jet set
with certified tails, produce per-band enclosure certificates, and ship the
falsifier alongside. The output is either the certificates or an honest fail
receipt naming the band that resists — both are progress, and the second is more
informative.

Adjacent, cheaper: finish the rung ladder at `r = 0.0177` (42/70) and
`r = 0.0125` (21/70). Engineering, not premise discharge, and the sources say so.

## 4. Build the missing annulus driver for `D3-LEMMA-RN-UNIF` Piece 2

`LANE_RN_UNIF` records Piece 2's Riemann-sum driver as **unwritten**, and RN5
gives the exact recipe for what replaces it: a complete non-overlapping spatial
cover of `0.1 ≤ |y| ≤ 5`, retaining boundary-area bounds and every rejected
cell, summing area × corrected cell supremum, verifying no cell remains pending,
then reassembling the remote budget. RN5's ten boxes are explicitly *not* a
coverage certificate, and the near-axis refinement cost has to be faced.

I can write that driver against the corrected envelope, with the
accept/refine/reject ledger as a first-class output so partial coverage is never
mistakable for full coverage. Also on this lane: the whitened `env_form`
orders 2–4 runnable smoke, which the lane brief lists as missing.

## 5. Perform the overdue nonauthor technical reviews

**Status 2026-09-18.** Six records in `reviews/records/`, all three READY routes
covered (`RV-P15` split into four scoped reviews as proposed), every record at
`independence_credit: 0` with `gate_status_after: UNCHANGED`, which
`tools/reviews_check.py` refuses to accept any other value for. Two AMEND
verdicts found defects in this repository's own migration (§1 of the findings
record). The prediction below — that I cannot supply organizational
independence — held, but for a sharper reason than same-provider: on the two
routes whose author family is `openai`, the credit is zero because
`OP-PROT-012` §5's predicate fails on (b), (c), (e) and (i), the reviewer having
read this repository's derived port before any freeze of its own. The remaining
18 routes are untouched.

Nineteen of the 24 review-queue routes are 51–54 days old and at ESCALATE. R17
§4 permits a fresh nonauthor session of any provider to perform technical
review, with organizational independence recorded separately and, for a
same-provider reviewer, at zero.

I am a fresh nonauthor session for every one of these objects. I can take them
in the order the queue implies — the component verdicts before `RV-LM011`'s
synthesis — and produce reviews in the exact form R17 §4 requires: source
ID/hash/bytes and extraction rule, exposure disclosure, precise hypotheses,
reconstructed argument, executed negative controls, findings by criterion,
unresolved dependencies, verdict, reproducible output.

The three `READY` routes are the best first targets: `RV-RN3` (replay the bundle
and challenge every uniform bound), `RV-P15` (split into four scoped reviews),
`RV-OPS-R17` (challenge stale handoff, same-family coauthor, publication race
and restoration cases using the bundled tests).

**What I cannot supply:** organizational independence. Where the predicate
demands a distinct family, my verdict is a technical pass with zero independence
credit, and the gate stays open. I will always record it that way.

## 6. Resolve the identifier collisions the registers carry

**Status 2026-09-18.** Proposed, not repaired: `registers/COLLISION_PROPOSAL.md`
covers all 16 findings with 13 successor identifiers under `OP-CNS-001` §2's
preserve-and-disambiguate remedy, and `tools/collision_proposal_check.py`
asserts the proposal is additive and the exported registers byte-unchanged.
For the three `EXISTING_CONTAINER` rows both permitted options are laid out
with consequences and a recommendation. Applying any of it is an operator
action.

Six duplicate artifact IDs and seven duplicate transition IDs, each pair holding
*different* status text — the exact failure class OP-CNS-001 §2 exists to
prevent, now present in the collision registry's own source. Mechanical to fix
at the source with successor IDs, and `tools/registers_check.py` already flags
them so they cannot quietly grow. See `registers/KNOWN_FINDINGS.json`.

Also: three Quarantine Index rows use `EXISTING_CONTAINER`, a class the protocol
does not define. Either add `CONTAINER_POINTER` to OP-PROT-019 §6 or reclassify.

## 7. Recover the eight empty native bodies and five read failures

**Done, and the headline guess in this section was wrong.** Ledger:
[`recovery/LEDGER.json`](../recovery/LEDGER.json), 31 records —
**15 RECOVERED** (every one digest-corroborated), 1 CANDIDATE, 15 UNRECOVERABLE
(current record per exception; 13 / 3 / 15 on 2026-09-18, before the two Markdown
candidates were recovered under a display rule found on 2026-09-19 — see
`recovery/README.md`).

What came back: three archive members resolved through the member index
(`SHA256SUMS`, `arithmetic_ledgers.fresh.txt`,
`first_variation_derivation.fresh.txt`), eight `ENCODED_BLOCK_FAILURE` base64
chunks re-extracted from `LS-DATA-009-v1.1`, and two gzip result payloads from
`CL-AUD-084` — several of these only because the Drive has moved since the
2026-09-17 snapshot, so a read that failed at audit time succeeds now.

What did not, and why this section's claim was wrong. I wrote that the TB-G2
capsules "are recoverable from Drive revision history, from the archive carriers,
or by regeneration from their sources." **Drive revision history is not available
for them.** All three `LS-DATA-015` shells report `createdTime == modifiedTime`
(2026-07-25 22:19:38.365Z, 22:20:40.640Z, 22:21:59.144Z), so no post-creation
revision exists to recover. Their bodies are still a bare UTF-8 BOM on fresh
2026-09-18 bytes, independently reproducing the audit's own finding. And the
corpus states **no digest for any of the eight**, so even a recovered body could
not have been corroborated.

Worse for the premise: the three "Corrected" shells were created minutes after
`LS-DATA-015-v1.0`, whose own capsule document
(`1jCJcC_dCgNSi6oYJOR9IGI5r49a2aa-qKZLvOInxRRI`) is titled *"MALFORMED PAYLOAD …
DO NOT USE"*. Recovering the original would restore the object the corpus itself
tells you not to use, not the corrected result.

So the eight remain UNRECOVERABLE, recorded with exactly what is missing. That is
the audit's own discipline: missing content is not invented.

## 8. Keep the git side honest as the Drive moves

The source map is a snapshot. `tools/drive_index.py` plus a modified-time delta
gives a scoped refresh that updates only affected derived copies, which is what
the accessibility report asks future maintenance to do. I can run that refresh,
diff the inventory, and open a PR per delta — so the Drive and the repository
never silently diverge.

---

## What I will not do

* Promote any premise, discharge any obligation, or relabel any status without
  the exact predicate that licenses it. Where a gate needs a distinct provider
  family, I am not that family and I will say so every time.
* Compose the 2D and 3D tracks, or let the prize track into the q0 dependency
  graph. CI now fails on both.
* Treat a display, a Monte Carlo estimate, a fitted exponent, a session CLOSE,
  a smoke test, or a registration as a proof.
* Edit a frozen body in place, or edit an exported register to make a check pass.
* Claim that any original prize problem is solved.
