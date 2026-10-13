import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text(encoding='utf-8')
def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

# identity rows and mentions
rep("SHA-256 `e4f76f7cd7f4731c7e3ff8c77d39fc3259e7832d2b7f92539be8dc913ad3135d`, 28898 bytes; imports",
    "SHA-256 `df9d3645282ca526b67eca3d5db691b622092a14b5fddee9be21013e39666184`, 29590 bytes; imports")
rep("| Registered bytes | SHA-256 `e4f76f7cd7f4731c7e3ff8c77d39fc3259e7832d2b7f92539be8dc913ad3135d`, 28898 bytes, 445 lines |",
    "| Registered bytes | SHA-256 `df9d3645282ca526b67eca3d5db691b622092a14b5fddee9be21013e39666184`, 29590 bytes, 453 lines |")
rep("owner-authored `OrdinaryH0Controls.lean` (e4f76f7c…) is added as module five",
    "owner-authored `OrdinaryH0Controls.lean` (df9d3645…) is added as module five")

# finding 5: the C254 naturality exclusion keeps its copied words; the relabel is a dated companion note
rep("""- naturality comparing the displayed component maps with singular-H0 generators beyond the
  functor laws actually proved: C254 alone did not establish this; C264's `OrdinaryH0.lean`
  establishes the stated ordinary-H0 naturality only (`ordinaryH0_iso_naturality`,
  `ordinaryH0_naturalIso`, `ordinaryH0_filtration_naturality`), and nothing beyond it;
""", """- naturality comparing the displayed component maps with singular-H0 generators beyond the
  functor laws actually proved; (companion note, 2026-10-12, Math- #428 comment 6116225936, not
  part of the copied exclusion: C254 alone did not establish this; C264's `OrdinaryH0.lean`
  establishes the stated ordinary-H0 naturality only (`ordinaryH0_iso_naturality`,
  `ordinaryH0_naturalIso`, `ordinaryH0_filtration_naturality`), and nothing beyond it)
""")

# finding 7a: premise lemmas name no C264 theorem
rep("""or axiom states a naturality, merger or separation fact. Each statement applies one of the C264
theorems above to a case whose answer is known independently.
""", """or axiom states a naturality, merger or separation fact. The premise lemmas (coefficient identities,
explicit paths, discrete separations, the parabola levels) are about the fixtures alone and are
proved directly, naming no C264 theorem. The other statements concern C264's point classes, its
natural isomorphism and its component coproduct (or Mathlib's H0 through them) on cases whose answer
is known independently; fourteen of them apply one of the C264 theorems above by name, seven
distinct ones (`ordinaryH0_pointClass_augmentation`, `ordinaryH0_pointClass_ne_zero`,
`ordinaryH0_sameComponent_difference_zero`, `ordinaryH0_pointClass_ne_of_component_ne`,
`ordinaryH0_pointClass_map`, `ordinaryH0_difference_map_eq_zero_iff`,
`ordinaryH0_filtration_difference_zero_iff_joinedIn`); the module names no other C264 theorem.
""")

# finding 2 (and 1, 6): the closing paragraph of "What is NOT established"
rep("""source review (6113277173) read the handoff bytes, not the successors. Until 2026-10-12 01:32Z no
nonauthor review of the compiled successor bytes existed (so this paragraph read at the
registration commit `940dffc9`); since then Math- #428 review 5485883117 (Muse Spark / Meta, first
read, PASS source-scoped for each module; the reviewer did not run Lean and retains FORMAL HOLD;
count corrected in comment 6116328276) is one, with the engineering readback of
`4c33d8cd..940dffc9` in review 5485829137 (PASS). Both handoffs keep formal acceptance on HOLD;
its disposition is for the C263/C264 handoff owners and nothing in this companion moves it. No
review of the `OrdinaryH0Controls` bytes is recorded.
""", """source review (6113277173) read the handoff bytes, not the successors. At the registration commit
`940dffc9` (2026-10-11) no nonauthor review of the compiled successor bytes existed, and this
paragraph said so. Until 2026-10-12 01:32Z no nonauthor exact-source (mathematical) review of them
existed; the nonauthor engineering registration readback of `4c33d8cd..940dffc9` (Math- #428 review
5485829137, 2026-10-12 01:16Z, PASS) read those bytes for identity and registration only, and records
organizational-independence credit 0. Since 01:32Z Math- #428 review 5485883117 (Muse Spark / Meta,
first read, PASS source-scoped for each module; count corrected in comment 6116328276) is such a
review, source-scoped only: its reviewer did not run Lean and keeps Lean execution as an open
condition under FORMAL HOLD, and it states no organizational-independence credit and was posted from
the same project account. Both handoffs keep formal acceptance on HOLD on their own listed
conditions (6113033545 item 4; 6113372506; 6113277485 item 5), among them a distinct review of the
source-bound execution output and source-bound hosted evidence and custody, which they say a source
review alone does not meet; its disposition is for the C263/C264 handoff owners and nothing in this
companion moves it. Alignment still needs a reviewer with nonzero organizational-independence
credit. No review of the `OrdinaryH0Controls` bytes is recorded.
""")

# compile-fixes paragraph on the controls module bytes
rep("""SHA-256 `e825837f3b0e43553a3a2f48ca48e4150182ad160eb0d7e1a10c141f9c395056`, 28503 bytes, 440 lines). The registered
bytes (`e4f76f7c…`, 28898 bytes, 445 lines) differ from that form only in one module-docstring paragraph, which now
records that the nonauthor exact-source review 5485883117 exists and that FORMAL HOLD is unchanged; no declaration
changed, and the module was recompiled, its axioms and its census re-run after the edit (see
""", """SHA-256 `e825837f3b0e43553a3a2f48ca48e4150182ad160eb0d7e1a10c141f9c395056`, 28503 bytes, 440 lines). The registered
bytes (`df9d3645…`, 29590 bytes, 453 lines) differ from that form only in one module-docstring paragraph, which now
records that FORMAL HOLD rests on the handoffs' own listed conditions, that the source-scoped review 5485883117
(Lean execution kept as an open condition, no credit stated) and the engineering readback 5485829137 (credit 0)
exist, and that FORMAL HOLD is unchanged; no declaration changed, and the module was recompiled, its axioms and its
census re-run after the edit (see
""")

# finding 4: hosted completion of the registered set
rep("""lowering it buys nothing. One hosted completion is recorded here; every later run's receipt stands
on its own, and a receipt is a record of a computation, not evidence of alignment.""",
"""lowering it buys nothing. One hosted completion is measured here; every later run's receipt stands
on its own, and a receipt is a record of a computation, not evidence of alignment.""")
rep("""locally; at the registration commit the registered set had no hosted completion, and this README
records none since. The item-4 set""", """locally. At the registration commit the registered set had no hosted completion. Hosted run
[38187870416](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38187870416) of `940dffc9` (created
2026-10-12T00:24:18Z) later completed with conclusion success, as reported by the engineering readback
5485829137, which inspected its artifact 11719939176 and receipt (GitHub's run record, read by this lane on
2026-10-12, shows the same status, conclusion and head); this lane has not downloaded or re-inspected that
artifact, and this README does not adopt its receipt or its timings. The item-4 set""")
rep("""a full local replay took 279 s); no hosted completion of it is recorded here.""",
    """a full local replay took 279 s); the item-4 commit has no hosted completion recorded here.""")

# change-log bullet
rep("""  eleven controls). The same commit time-scopes the
  statements that no review of the compiled successors existed (true at `940dffc9`; review 5485883117 exists since
  2026-10-12 01:32Z) and relabels the C254 naturality exclusion (Math- #428 comment 6116225936); no label moves.
""", """  eleven controls). The same commit time-scopes the
  statements that no review of the compiled successors existed (true at `940dffc9`; the engineering registration
  readback 5485829137 exists since 2026-10-12 01:16Z, credit 0, and the source-scoped review 5485883117 since 01:32Z,
  no credit stated, Lean execution kept open), restates FORMAL HOLD as resting on the handoffs' own listed conditions
  rather than on one review, records that hosted run 38187870416 of `940dffc9` completed (as reported, not adopted),
  and marks the relabel of the C254 naturality exclusion (Math- #428 comment 6116225936) as a dated companion note;
  no label moves.
""")

# registration section intro (findings 1, 6)
rep("""gate; it moves no status label. Both handoffs keep their own label, **FORMAL HOLD**, which stands
until a nonauthor exact-source review of the compiled successors exists; the C263 (6113033351)
and C264 (6113277173) nonauthor source reviews read the handoff bytes, not the successors, and
this lane (compile fix, repair, registration) is the same project account as the companion,
organizational-independence credit 0. (This section describes the registration commit
`940dffc9` of 2026-10-11. Since 2026-10-12 01:32Z a nonauthor exact-source review of the compiled
successors exists: Math- #428 review 5485883117, Muse Spark / Meta, first read, PASS source-scoped
for each module, the reviewer did not run Lean and retains FORMAL HOLD, count corrected in comment
6116328276; the engineering readback of `4c33d8cd..940dffc9` is review 5485829137, PASS. FORMAL
HOLD is unchanged and its disposition is for the C263/C264 handoff owners; no label below moves,
and the record `provenance.successors.*.formal_status` keeps the handoffs' own text.)
""", """gate; it moves no status label. Both handoffs keep their own label, **FORMAL HOLD**, which rests
on the handoffs' own listed conditions (6113033545 item 4; 6113372506; 6113277485 item 5): among
them a nonauthor exact-source review of the compiled successors, a distinct review of the
source-bound execution output and source-bound hosted evidence and custody; the handoffs say a
source review alone does not meet them. The C263 (6113033351) and C264 (6113277173) nonauthor
source reviews read the handoff bytes, not the successors, and this lane (compile fix, repair,
registration) is the same project account as the companion, organizational-independence credit
0. (This section describes the registration commit `940dffc9` of 2026-10-11. At that commit this
paragraph said that FORMAL HOLD stands until a nonauthor exact-source review of the compiled
successors exists, which names only one of the handoffs' conditions; corrected on 2026-10-12. On
2026-10-12 the nonauthor engineering readback of `4c33d8cd..940dffc9`, Math- #428 review 5485829137
(01:16Z, PASS, identity and registration only, organizational-independence credit 0 as it records),
was posted, and since 01:32Z a nonauthor exact-source review of the compiled successors exists,
source-scoped only: Math- #428 review 5485883117, Muse Spark / Meta, first read, PASS source-scoped
for each module, count corrected in comment 6116328276; its reviewer did not run Lean and keeps Lean
execution as an open condition under FORMAL HOLD, and it states no organizational-independence
credit and was posted from the same project account. FORMAL HOLD is unchanged and its disposition
is for the C263/C264 handoff owners; no label below moves, and the record
`provenance.successors.*.formal_status` keeps the text this commit transcribed.)
""")

# Status row (finding 1)
rep("""| Status | FORMAL HOLD (handoff's own label) until a nonauthor exact-source review of the compiled successor | FORMAL HOLD, same |""",
    """| Status | FORMAL HOLD (handoff's own label), on the handoff's own listed conditions (6113033545 item 4); `provenance.successors` records it as `FORMAL_HOLD until a nonauthor exact-source review of the compiled successor; registration is not review`, this commit's transcription, which names one of those conditions only | FORMAL HOLD, same (conditions: 6113372506; 6113277485 item 5) |""")

# registration "does not establish": review and hosted receipt
rep("""review of the compiled successors (none existed at this commit, 2026-10-11: this lane compiled, repaired and registered
them and has not reviewed them; since 2026-10-12 the nonauthor review 5485883117 and the engineering readback
5485829137 exist, recorded above); any
organizational-independence credit; any hosted receipt for the registered set (a green hosted
run of this commit would produce one, and a receipt is a record of a computation). The compile fix""",
"""review of the compiled successors (none existed at this commit, 2026-10-11: this lane compiled, repaired and registered
them and has not reviewed them; since 2026-10-12 the engineering registration readback 5485829137, 01:16Z, credit 0,
and the source-scoped review 5485883117, 01:32Z, no credit stated, Lean execution kept open, exist, recorded above);
any organizational-independence credit; any hosted receipt for the registered set (none existed at this commit; hosted
run 38187870416 of `940dffc9` later completed with conclusion success, as reported by the engineering readback
5485829137; this lane has not re-inspected it and this README does not adopt its receipt; a receipt is a record of a
computation). The compile fix""")

# item-4 section intro (finding 7, same overclaim)
rep("""`OrdinaryH0`: they instantiate its statements on named spaces and coefficient modules with every
premise proved, so that each statement is seen in use, and its nonzero-coefficient hypothesis seen
to be needed, on a case whose answer is known independently. The FORMAL HOLD of `SuperlevelBar` and
`OrdinaryH0` is unchanged (the review 5485883117 that now exists retains it, and its disposition is
for the C263/C264 handoff owners); alignment stays `PENDING_INDEPENDENT_REVIEW`; organizational-independence
credit 0 (the same project account wrote, compiled and registered the module); scientific effect NONE.
""", """`OrdinaryH0`: they apply seven of its 20 theorems and three of its seven definitions
(`ordinaryH0_pointClass`, `ordinaryH0_naturalIso`, `ordinaryH0_componentCoproductFunctor`) to named
spaces and coefficient modules with every premise proved, so that those statements are seen in use,
and the nonzero-coefficient hypothesis seen to be needed, on cases whose answer is known
independently. The FORMAL HOLD of `SuperlevelBar` and `OrdinaryH0` is unchanged (it rests on the
handoffs' own listed conditions; the source-scoped review 5485883117 that now exists retains it and
keeps Lean execution as an open condition; its disposition is for the C263/C264 handoff owners);
alignment stays `PENDING_INDEPENDENT_REVIEW`; organizational-independence credit 0 (the same project
account wrote, compiled and registered the module); scientific effect NONE.
""")
rep("""| First compiled form | SHA-256 `e825837f3b0e43553a3a2f48ca48e4150182ad160eb0d7e1a10c141f9c395056`, 28503 bytes, 440 lines; the registered bytes change only the module-docstring paragraph on review status (review 5485883117 exists, FORMAL HOLD unchanged); no declaration changed |""",
    """| First compiled form | SHA-256 `e825837f3b0e43553a3a2f48ca48e4150182ad160eb0d7e1a10c141f9c395056`, 28503 bytes, 440 lines; the registered bytes change only the module-docstring paragraph on review status (FORMAL HOLD rests on the handoffs' listed conditions; the source-scoped review 5485883117, no credit stated, and the engineering readback 5485829137, credit 0, exist; FORMAL HOLD unchanged); no declaration changed |""")

# item-4 "does not establish" (finding 7b) plus the unexercised theorems
rep("""attainment theorem, interval decomposition, elder correspondence, count identity, Field adapter,
stochastic law or scientific-status promotion; a first merger level or threshold attainment; any
statement about connected (as opposed to path) components beyond the discrete cases; alignment""",
"""attainment theorem, interval decomposition, elder correspondence, count identity, Field adapter,
stochastic law or scientific-status promotion; a first merger level or threshold attainment; any
statement about connected (as opposed to path) components; a fixture for any of the 13 C264
theorems the module does not name (among them `ordinaryH0_iso_naturality` and
`ordinaryH0_filtration_naturality`); alignment""")
p.write_text(s, encoding='utf-8')
print('README edits applied')
