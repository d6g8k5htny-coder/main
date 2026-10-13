"""Fix round, second pass: a credit statement beside the two remaining review pointers (README item-4 intro and hosted
paragraph) and in the check.py SUCCESSORS comment. Applied to the companion directory given as argv[1]."""
import sys
from pathlib import Path
C = Path(sys.argv[1])
def rep(path, old, new):
    s = path.read_text(encoding='utf-8'); n = s.count(old)
    assert n == 1, (path.name, n, old[:100]); path.write_text(s.replace(old, new), encoding='utf-8')
rep(C / 'README.md', """handoffs' own listed conditions; the source-scoped review 5485883117 that now exists retains it and
keeps Lean execution as an open condition; its disposition is for the C263/C264 handoff owners);""",
"""handoffs' own listed conditions; the source-scoped review 5485883117 that now exists, which states no
organizational-independence credit and was posted from the same project account, retains it and keeps
Lean execution as an open condition; its disposition is for the C263/C264 handoff owners);""")
rep(C / 'README.md', """2026-10-12T00:24:18Z) later completed with conclusion success, as reported by the engineering readback
5485829137, which inspected its artifact 11719939176 and receipt""",
"""2026-10-12T00:24:18Z) later completed with conclusion success, as reported by the engineering readback
5485829137 (credit 0), which inspected its artifact 11719939176 and receipt""")
rep(C / 'check.py', """# them. `formal_status` below is the registration commit's transcription, naming one of those conditions; it is kept
# unchanged, and no review recorded since (2026-10-12: engineering readback 5485829137, source-scoped review
# 5485883117) moves it. Registration is not review, not acceptance; organizational-independence credit 0 (same project
# account).
""", """# them. `formal_status` below is the registration commit's transcription, naming one of those conditions; it is kept
# unchanged, and no review recorded since moves it (2026-10-12: engineering readback 5485829137, credit 0;
# source-scoped review 5485883117, no organizational-independence credit stated, same project account). Registration
# is not review, not acceptance; organizational-independence credit 0 (same project account).
""")
print('fix3 applied')
