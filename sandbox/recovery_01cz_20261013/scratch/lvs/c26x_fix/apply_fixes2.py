"""Final fix pass on the registration commit after the delta verification (refuted on two remaining overclaims)."""
import pathlib, re
def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8'); print('edited', path, len(pairs))
edit('SCOPE.md', [
    ("injected admission, every honesty label and\nschema field of `SOURCE_FILES.json` one at a time, wrong",
     "injected admission, every honesty label and\nschema field of `SOURCE_FILES.json` outside the successor record one at a time (the successor record's\ncoverage is stated at the end of this list), wrong"),
    ("handoff or successor hash, successor or original byte count, a changed census line,",
     "handoff or successor hash, a successor byte count or a C26x entry of `provenance.original_bytes`, a changed census line,"),
])
edit('test_torus_filtration.py', [
    ('    """Every honesty label and schema rule of SOURCE_FILES.json is enforced through the CLI, one field at a time."""',
     '    """The honesty labels and schema rules of SOURCE_FILES.json are enforced through the CLI, one field at a time,\n'
     '    except seven fields of the successor record (handoff byte count and status, lane, authors, namespace, imports,\n'
     '    change summary), which are refused only through the whole-record comparison in check.py metadata()."""'),
])
edit('replay.sh', [
    ("# Replay of the torus-filtration companion (the C254 modules and the registered C263/C264 successors). It runs neither formal/gate.py nor the formal\n# tests:",
     "# Replay of the torus-filtration companion: TorusTopology and ClosedSuperlevel (C254) and the registered\n# SuperlevelBar (C263) and OrdinaryH0 (C264) successors. It runs neither formal/gate.py nor the formal tests:"),
    ("# leanchecker over the five modules peaked at about 8.4 GB resident at two threads and exceeded a 13.36 GB memory\n# limit at four (2026-10-11, this lane);",
     "# leanchecker over the five modules peaked at about 8.4-8.7 GiB resident at two threads and exceeded a 13.36 GiB\n# memory limit at four (2026-10-11, this lane);"),
    ("# .oleans (30 s at 2 threads locally for the C254 pair).", "# .oleans (30 s at 2 threads locally for the three C254 modules, 53 s for the five)."),
])
edit('README.md', [
    ("registered bytes) with a peak of about 8.4 GB resident (sampled every 2 s), no output on success",
     "registered bytes) with a peak of about 8.4 to 8.7 GiB resident (two runs, sampled every 2 s), no output on success"),
    ("13.36 GB memory limit and was killed (twice, 2026-10-11)", "13.36 GiB memory limit and was killed (twice, 2026-10-11)"),
    ("the registered set (five builds, eight controls, 54 files) adds two `lean` compiles of about 10\nand 12 s and two control runs of about 10 s each at `LEAN_NUM_THREADS=2`, measured locally, and\nhas no hosted completion yet.",
     "the registered set (five builds, the census, eight controls, 54 files) adds, relative to that run, the\ncensus `lean` run (from the C262 AMEND), two `lean` compiles of about 10 and 12 s, two control runs of\nabout 10 s each and about 23 s more of `leanchecker` (30 s to 53 s), at `LEAN_NUM_THREADS=2`, measured\nlocally; the registered set has no hosted completion yet."),
    ("`_proof_1`–`_proof_4` of the point class", "`_proof_1`–`_proof_4` of the simplicial point class"),
    ("root `01a0bbb5` design, custody and control design, `c264_h0_author/api_support` read-only API research,",
     "root `01a0bbb5` supplied the target, custody and the nonzero-coefficient/projection control design, `c264_h0_author/api_support` did read-only pinned-API research and proposed the cycle/projection naturality rewrite chain (the step that failed at the pin and was repaired),"),
])
edit('check.py', [
    ("'source_author': 'OpenAI/Codex lane (C264, c264_h0_author; root 01a0bbb5 design, custody and control design, '\n"
     "                         'c264_h0_author/api_support read-only API research, as disclosed in Math- #428 comment 6113276830), '",
     "'source_author': 'OpenAI/Codex lane (C264, c264_h0_author; root 01a0bbb5 supplied the target, custody and the '\n"
     "                         'nonzero-coefficient/projection control design; c264_h0_author/api_support did read-only pinned-API '\n"
     "                         'research and proposed the cycle/projection naturality rewrite chain; as disclosed in Math- #428 '\n"
     "                         'comment 6113276830), '"),
])
