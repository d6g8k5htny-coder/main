"""Single-point mutants of tests/test_active_board_routing.py helpers (main#487 at 0a0f60bd), plus data mutants.
Usage: python3 -I mut487.py TREE OUT.json  (TREE = git archive of the PR head)."""
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
TREE = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2])
T = "tests/test_active_board_routing.py"
CODE = [
 ("M01 image guard removed", r'(?<!!)\[', r'\['),
 ("M02 CANONICAL case-sensitive", 'CANONICAL = re.compile(r"Agent Message Board #([1-9][0-9]*)", re.I)', 'CANONICAL = re.compile(r"Agent Message Board #([1-9][0-9]*)")'),
 ("M03 ACTIVE accepts only 'active'", '(?:active|current) Agent', '(?:active) Agent'),
 ("M04 ACTIVE case-sensitive", 'main #([1-9][0-9]*)", re.I)', 'main #([1-9][0-9]*)")'),
 ("M05 fence close ignores length", ' and len(token[1]) >= len(fence)', ''),
 ("M06 fence close ignores char", 'token and token[1][0] == fence[0]\n', 'token\n'),
 ("M07 fence close ignores info string", ' and not token[2].strip()):', '):'),
 ("M08 fences never open", '        if token:\n            fence = token[1]\n            continue\n', ''),
 ("M09 fence indent unbounded", r'^ {0,3}(`{3,}|~{3,})(.*)$', r'^ *(`{3,}|~{3,})(.*)$'),
 ("M10 no whitespace normalization", '" ".join(match[1].split())', 'match[1]'),
 ("M11 canonical count >=1 allowed", 'if len(canonical) != 1:', 'if len(canonical) < 1:'),
 ("M12 canonical target check removed", '    if target != expected:\n        return [f"AGENTS.md:{line}: canonical label/target mismatch"]\n', ''),
 ("M13 active number check removed", 'if active[1] != canonical[0][1][1] or target != expected:', 'if target != expected:'),
 ("M14 active target check removed", 'if active[1] != canonical[0][1][1] or target != expected:', 'if active[1] != canonical[0][1][1]:'),
 ("M15 vacuous success allowed", '    if not count:\n        problems.append("docs/RESEARCH_INDEX.md: no qualifying active board links")\n', ''),
 ("M16 fullmatch->search on ACTIVE", 'active = ACTIVE.fullmatch(label)', 'active = ACTIVE.search(label)'),
 ("M17 fullmatch->search on CANONICAL (filter)", 'if CANONICAL.fullmatch(label)]', 'if CANONICAL.search(label)]'),
 ("M18 LINK target allows newline-free only -> any", r'\(([^)\n]*)\)', r'\(([^)]*)\)'),
 ("M19 problems collapse to first only", '            problems.append(f"docs/RESEARCH_INDEX.md:{line}: active board mismatch")', '            problems.append(f"docs/RESEARCH_INDEX.md:{line}: active board mismatch"); break'),
 ("M20 canonical first-of-many", '    if len(canonical) != 1:\n        return ["AGENTS.md: expected exactly one canonical board link"]\n', '    if not canonical:\n        return ["AGENTS.md: expected exactly one canonical board link"]\n'),
]
def data_mut(name, path, old, new, count=1):
    return (name, path, old, new, count)
DATA = [
 ("D1 RESEARCH_INDEX active link back to #419", "docs/RESEARCH_INDEX.md", "issues/475", "issues/419"),
 ("D2 AGENTS canonical board -> #419 (label+target)", "AGENTS.md", None, None),
 ("D3 AGENTS second canonical link appended", "AGENTS.md", None, None),
 ("D4 RESEARCH_INDEX active label renumbered to #419 only", "docs/RESEARCH_INDEX.md", None, None),
]
def run(root, opt):
    cmd = [sys.executable, "-B", "-S"] + (["-O"] if opt else []) + ["-m", "unittest", "discover", "-s", "tests", "-p", "test_active_board_routing.py"]
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=120)
    tail = [l for l in r.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]
    return r.returncode, " ".join(tail)
rows = []
def record(name, apply):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "t"
        shutil.copytree(TREE, root, ignore=shutil.ignore_patterns(".git"))
        note = apply(root)
        n, o = run(root, False), run(root, True)
    v = "BASE_OK" if name == "BASE" and n[0] == 0 and o[0] == 0 else ("KILLED" if n[0] and o[0] else ("SURVIVED" if not n[0] and not o[0] else "MIXED"))
    rows.append({"mutant": name, "normal": n[1], "optimized": o[1], "verdict": v, "note": note})
    print(json.dumps(rows[-1]), flush=True)
record("BASE", lambda root: None)
for name, old, new in CODE:
    def apply(root, old=old, new=new):
        p = root / T; t = p.read_text()
        c = t.count(old); assert c >= 1, (name, old)
        p.write_text(t.replace(old, new, 1)); return f"occurrences={c}"
    record(name, apply)
def d1(root):
    p = root / "docs/RESEARCH_INDEX.md"; t = p.read_text()
    m = list(re.finditer(r"\[(?:active|current) Agent Message Board, main #475\]\(https://github\.com/d6g8k5htny-coder/main/issues/475\)", t, re.I))
    assert m, "no active link"
    s = m[0]; p.write_text(t[:s.start()] + s.group(0).replace("475", "419") + t[s.end():]); return f"active links={len(m)}"
def d2(root):
    p = root / "AGENTS.md"; t = p.read_text()
    m = list(re.finditer(r"\[Agent Message Board #475\]\(https://github\.com/d6g8k5htny-coder/main/issues/475\)", t, re.I)); assert len(m) == 1, len(m)
    s = m[0]; p.write_text(t[:s.start()] + s.group(0).replace("475", "419") + t[s.end():]); return "canonical->419"
def d3(root):
    p = root / "AGENTS.md"; p.write_text(p.read_text() + "\nSee [Agent Message Board #419](https://github.com/d6g8k5htny-coder/main/issues/419).\n"); return "appended"
def d4(root):
    p = root / "docs/RESEARCH_INDEX.md"; t = p.read_text()
    m = list(re.finditer(r"\[((?:active|current) Agent Message Board, main #)475\]", t, re.I)); assert m
    s = m[0]; p.write_text(t[:s.start()] + "[" + s.group(1) + "419]" + t[s.end():]); return "label only"
for (name, *_), fn in zip(DATA, (d1, d2, d3, d4)):
    record(name, fn)
OUT.write_text(json.dumps(rows, indent=1) + "\n")
