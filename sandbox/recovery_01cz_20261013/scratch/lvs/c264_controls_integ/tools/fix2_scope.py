import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text(encoding='utf-8')
def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

rep("""decides. The C263 and C264 modules are registered under the handoffs' own FORMAL HOLD, which
the handoffs tied to a nonauthor exact-source review of the compiled successor bytes; registration
is not review. At the registration commit `940dffc9` (2026-10-11) no such review existed; since
2026-10-12 01:32Z Math- #428 review 5485883117 (Muse Spark / Meta, first read, PASS source-scoped
for each module; the reviewer did not run Lean and retains FORMAL HOLD; count corrected in comment
6116328276) is one, with the engineering readback of `4c33d8cd..940dffc9` in review 5485829137
(PASS). FORMAL HOLD is unchanged; its disposition is for the C263/C264 handoff owners, and this
note moves no label. `OrdinaryH0Controls` instantiates the C264 statements on concrete fixtures; it
is not a handoff module, not a review of `OrdinaryH0`, and lifts nothing.""",
"""decides. The C263 and C264 modules are registered under the handoffs' own FORMAL HOLD, which
rests on the handoffs' own listed conditions (Math- #428 comments 6113033545 item 4, 6113372506
and 6113277485 item 5), among them a nonauthor exact-source review of the compiled successor
bytes, a distinct review of the source-bound execution output and source-bound hosted evidence and
custody; the handoffs say a source review alone does not meet them. Registration is not review. At
the registration commit `940dffc9` (2026-10-11) no nonauthor exact-source review of the successor
bytes existed. On 2026-10-12 the nonauthor engineering registration readback of
`4c33d8cd..940dffc9`, Math- #428 review 5485829137 (01:16Z, PASS, identity and registration only),
was posted and records organizational-independence credit 0; since 01:32Z Math- #428 review
5485883117 (Muse Spark / Meta, first read, PASS source-scoped for each module; count corrected in
comment 6116328276) is an exact-source review, source-scoped only: its reviewer did not run Lean and
keeps Lean execution as an open condition under FORMAL HOLD, and it states no
organizational-independence credit and was posted from the same project account. FORMAL HOLD is
unchanged; its disposition is for the C263/C264 handoff owners, alignment still needs a reviewer
with nonzero organizational-independence credit, and this note moves no label. `OrdinaryH0Controls`
applies seven of the 20 C264 theorems (and three of its definitions) to concrete fixtures; it is not
a handoff module, not a review of `OrdinaryH0`, and lifts nothing.""")

rep("""  not a successor): SHA-256 `e4f76f7cd7f4731c7e3ff8c77d39fc3259e7832d2b7f92539be8dc913ad3135d`
  (28898 bytes; first compiled form `e825837f…`, 28503 bytes, differing only in the""",
"""  not a successor): SHA-256 `df9d3645282ca526b67eca3d5db691b622092a14b5fddee9be21013e39666184`
  (29590 bytes; first compiled form `e825837f…`, 28503 bytes, differing only in the""")

rep("""organizational-independence credit records otherwise. For `SuperlevelBar` and `OrdinaryH0` the
handoffs' own FORMAL HOLD stands in addition. The handoffs tied it to a nonauthor exact-source
review of the compiled successor bytes (346eea94…, 3d6b0d6b…); until 2026-10-12 01:32Z none
existed (so this paragraph read at the registration commit `940dffc9`), and since then review
5485883117 is one, whose reviewer retains FORMAL HOLD; its disposition is for the C263/C264 handoff
owners. The registration of 2026-10-11 is not that review, and the compiler, repairer and
registrar of both successors, and the author and registrar of `OrdinaryH0Controls` (e4f76f7c…), is
the companion's own lane at organizational-independence credit 0. Aging approves nothing.""",
"""organizational-independence credit records otherwise. For `SuperlevelBar` and `OrdinaryH0` the
handoffs' own FORMAL HOLD stands in addition, on the handoffs' own listed conditions (6113033545
item 4; 6113372506; 6113277485 item 5), among them a nonauthor exact-source review of the compiled
successor bytes (346eea94…, 3d6b0d6b…), a distinct review of the source-bound execution output and
source-bound hosted evidence and custody, which they say a source review alone does not meet. At the
registration commit `940dffc9` no nonauthor review of those bytes existed. The engineering
registration readback 5485829137 (2026-10-12 01:16Z, identity and registration only, credit 0) and,
since 2026-10-12 01:32Z, the source-scoped review 5485883117 (its reviewer did not run Lean and keeps
Lean execution as an open condition under FORMAL HOLD; no organizational-independence credit stated,
posted from the same project account) exist; neither is a nonzero-credit review, and FORMAL HOLD is
unchanged, its disposition being for the C263/C264 handoff owners. The registration of 2026-10-11 is
not a review, and the compiler, repairer and registrar of both successors, and the author and
registrar of `OrdinaryH0Controls` (df9d3645…), is the companion's own lane at
organizational-independence credit 0. Aging approves nothing.""")

rep("""specification: the corrected CONTROL_OBLIGATIONS of comment 6113277173) instantiates the C264
statements on named spaces and coefficient modules with every premise proved in the fixture: the""",
"""specification: the corrected CONTROL_OBLIGATIONS of comment 6113277173) applies seven of the 20
C264 theorems and three of its definitions to named spaces and coefficient modules with every premise
proved in the fixture (its thirteen premise lemmas name no C264 theorem): the""")
rep("""item 4 is closed (no label moves); any Morse chart, critical-level attainment, interval
decomposition, elder correspondence, count identity, Field adapter or stochastic law; any
organizational-independence credit (0).""",
"""item 4 is closed (no label moves); any Morse chart, critical-level attainment, interval
decomposition, elder correspondence, count identity, Field adapter or stochastic law; any statement
about connected (as opposed to path) components; a fixture for the 13 C264 theorems it does not
name; any organizational-independence credit (0).""")
p.write_text(s, encoding='utf-8')
print('SCOPE edits applied')
