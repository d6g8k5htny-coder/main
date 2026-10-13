"""Fix the documentation slips the two adversarial verifiers found in the C263/C264 registration commit 41397c23
(one blocking: a phrase implying the successors were reviewed by their compiler). Exact string edits; each target must
occur exactly once. Run from the companion directory."""
import pathlib
def edit(path, pairs):
    p = pathlib.Path(path); s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8'); print('edited', path, len(pairs))

PROPOSAL_OLD = "the C263 handoff's own `FALSE_CONTROL_PROPOSAL`"
PROPOSAL_NEW = ("the proposition of the C263 handoff's `FALSE_CONTROL_PROPOSAL` (Math- #428 comment 6113033351), whose tactic "
                "`rw [superlevelBar_death_const 1 (-1) (0 : superlevelBar_E 1)]; norm_num` is replaced here by "
                "`norm_num [superlevelBar_death_const]`")
edit('README.md', [
    # blocking: no review happened
    ("stochastic law; a\nreview of the compiled successors by anyone other than their compiler and repairer; any",
     "stochastic law; any\nreview of the compiled successors (none exists: this lane compiled, repaired and registered them and has not reviewed them); any"),
    # disclosed open obligation + original_status scope
    ("decides. Local evidence for this commit",
     "decides. The concrete ordinary-H0 controls the C264 handoff asks for (item 4 of its owner task, Math- #428 comment\n"
     "6113277485: empty, point, two-component, merge/non-merge and zero-coefficient cases with premises proved in concrete\n"
     "fixtures) are not provided here: this commit adds one false control (`RejectReducedPointClass`) and no positive\n"
     "fixture, and that obligation stays open. `provenance.original_status` (`NOT_COMPILED`) records the two C254 handoff\n"
     "modules at delivery; the C263 and C264 handoffs' statuses are `provenance.successors.*.handoff_status`. Local evidence for this commit"),
    # counts and wording
    ("The module docstring gained five lines recording the handoff hash and this fix.",
     "The module docstring gained six lines (a blank separator and five text lines) recording the handoff hash and this fix."),
    ("- The three declaration names ending in `ι` are spelled",
     "- The three declaration names containing `ι` (two ending in it; `ordinaryH0_componentι_isoInv` has it inside) are spelled"),
    ("- Fourteen docstring lines recording the above. No statement changed; the 20 theorem and 7 definition names are the\n  handoff's, in the successor's file order.",
     "- Fourteen docstring lines (a blank separator and thirteen text lines) recording the above. No statement changed; the\n  20 theorem and 7 definition names are the handoff's except the three ASCII `iota` respellings above, in the successor's\n  file order."),
    ("(three module builds, six controls, 44 evidence files)", "(three module builds, six controls, 41 evidence files, 42 with the receipt)"),
    ("The out-of-tree generator refuses a registered module whose bytes are not the\n  recorded successor.",
     "The protocol tests (`InventoryTests`, run by the replay) refuse a registered module whose bytes are not the\n  recorded successor; the out-of-tree generator that writes `SOURCE_FILES.json` checks the same but is not part of this\n  tree, so no reader can audit it here."),
    ("OpenAI/Codex lane (`c264_h0_author`), delegated AI work for Dylan Roy |",
     "OpenAI/Codex lane (`c264_h0_author`; root `01a0bbb5` design, custody and control design, `c264_h0_author/api_support` read-only API research, as disclosed in Math- #428 comment 6113276830), delegated AI work for Dylan Roy |"),
    ("`sigmaConst.eq_1`, two `eq_1`,", "`sigmaConst.eq_1`, three `eq_1`,"),
    (PROPOSAL_OLD, PROPOSAL_NEW),
])
edit('SCOPE.md', [
    (PROPOSAL_OLD, PROPOSAL_NEW),
    ("module name outside `MODULES`, and every field of the successor record one at a time: a changed\nhandoff or successor hash or byte count, a changed census line, a nonzero credit, a lifted or\nself-awarded `formal_status`, a dropped comment pointer, a dropped or added module entry).",
     "module name outside `MODULES`, and single-field changes to the successor record (a changed\nhandoff or successor hash, successor or original byte count, a changed census line, a nonzero credit, a lifted or\nself-awarded `formal_status`, a dropped comment pointer, a dropped or added module entry); its remaining fields (handoff\nbyte count and status, lane, authors, namespace, imports, change summary) are refused through the whole-record\ncomparison in `metadata`, not by a control of their own)."),
])
edit('check.py', [
    ("# protocol tests and the out-of-tree generator require the tracked module bytes",
     "# protocol tests (InventoryTests, run by the replay) and the out-of-tree generator require the tracked module bytes"),
    ("five module-docstring lines recording the handoff hash and the fix; no other byte",
     "six module-docstring lines (a blank separator and five text lines) recording the handoff hash and the fix; no other byte"),
    ("the three names ending in iota spelled in ", "the three names containing iota spelled in "),
    ("'source_author': 'OpenAI/Codex lane (C264, c264_h0_author), delegated AI work for Dylan Roy',",
     "'source_author': 'OpenAI/Codex lane (C264, c264_h0_author; root 01a0bbb5 design, custody and control design, '\n"
     "                         'c264_h0_author/api_support read-only API research, as disclosed in Math- #428 comment 6113276830), '\n"
     "                         'delegated AI work for Dylan Roy',"),
    ("death 0 by superlevelBar_death_const, and 0 ≤ -1 is false; the C263 handoff's own FALSE_CONTROL_PROPOSAL);",
     "death 0 by superlevelBar_death_const, and 0 ≤ -1 is false; the proposition of the C263 handoff's FALSE_CONTROL_PROPOSAL,\n"
     "# Math- #428 comment 6113033351, with its rw-then-norm_num tactic replaced by norm_num [superlevelBar_death_const]);"),
])
edit('replay.sh', [
    ("# Replay of the C254 torus-filtration companion. It runs",
     "# Replay of the torus-filtration companion (the C254 modules and the registered C263/C264 successors). It runs"),
])
