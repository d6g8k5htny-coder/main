

_old_backticked = lambda ss: ', '.join('`' + s + '`' for s in ss[:-1]) + ' and `' + ss[-1] + '`'
_old_plain = lambda ss: ', '.join(ss[:-1]) + ' and ' + ss[-1]
for mods in (T4_MODULES, T5_MODULES):
    assert backticked(stems(mods)) == _old_backticked(stems(mods)) and plain(stems(mods)) == _old_plain(stems(mods))
MODS_TXT, MODS_PLAIN = backticked(stems(T4_MODULES)), plain(stems(T4_MODULES))
MODS5_TXT, MODS5_PLAIN = backticked(stems(T5_MODULES)), plain(stems(T5_MODULES))
MODS6_TXT, MODS6_PLAIN = backticked(stems(T6_MODULES)), plain(stems(T6_MODULES))
assert (MODS6_TXT, MODS6_PLAIN) == ('`Side24Images`', 'Side24Images')

# ----------------------------------------------------------------------------- receipt phrases of rounds 1-4 (render_docs11/13 verbatim)
HEADER_BIND6 = (f'{len(RECEIPTED52)} kernel-checked by hosted run {RUN2}: PR head {C2}, actual tested merge {M2[:8]}… with base {BASE[:8]}…; the {N3} of P15Interpolation by '
                f'run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base {BASE2[:8]}… (the head of PR #422); same trees as the heads')
HEADER_BIND9 = HEADER_BIND6 + (f'; the {N4} of {MODS_PLAIN} by run {RUN8}: PR head {C8}, actual tested merge {M8[:8]}… with base {BASE3[:8]}… (the head of PR #423), '
                               f'the merge having the same tree as PR head {C8}; receipt theorem_count 73, artifact sha256 {SHA8[:8]}…')
PEND3_TXT = (f'; the {N3} theorems of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4} with base {BASE}, receipt theorem_count 52), '
             f'are kernel-checked by the hosted run {RUN6} on PR head {C6} (actual tested merge {M6} with base {BASE2}, the head of PR #422; receipt theorem_count 60, '
             f'leanchecker PASS) and were first receipted by run {RUNA} on PR head {C5A} (actual tested merge {M5A} with the same base); same trees as the heads')
REC4_TXT = (f'; the {N4} theorems of {MODS_TXT}, added after PR head {C7} (run {RUN7}, actual tested merge {M7} with base {BASE2}; receipt theorem_count 60, '
            f'corroborating the {len(RECEIPTED60)} theorems above on that head; artifact lean-specification-proofs-{RUN7}-1, sha256 {SHA7}), are kernel-checked by '
            f'the hosted run {RUN8} on PR head {C8} (actual tested merge {M8} with base {BASE3}, the head of PR #423; receipt theorem_count 73, leanchecker PASS; '
            f'artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8}) and were first receipted by that run, which receipts all 73 theorems of that head; '
            'same tree as the head')


# templates of the receipted tier forms (render_docs11's, with the base description, the previous run's relation and the tree
# clause as parameters), pinned below to the committed round-4 forms (REC4_*) and, in step 0, to the committed round-5 forms
def hdr_tail(n, mods_plain, run, head, merge, base, base_desc, count, sha, hdr_tree, **_):
    return (f'; the {n} of {mods_plain} by run {run}: PR head {head}, actual tested merge {merge[:8]}… with base {base[:8]}… ({base_desc}), '
            f'{hdr_tree}; receipt theorem_count {count}, artifact sha256 {sha[:8]}…')


def rec_txt(n, mods_txt, prev_head, prev_run, prev_merge, prev_base, prev_count, relation, prev_sha, run, head, merge, base, base_desc, count, sha,
            tree_txt, **_):
    return (f'; the {n} theorems of {mods_txt}, added after PR head {prev_head} (run {prev_run}, actual tested merge {prev_merge} with base {prev_base}; receipt '
            f'theorem_count {prev_count}, {relation}; artifact lean-specification-proofs-{prev_run}-1, sha256 {prev_sha}), '
            f'are kernel-checked by the hosted run {run} on PR head {head} (actual tested merge {merge} with base {base}, {base_desc}; receipt '
            f'theorem_count {count}, leanchecker PASS; artifact lean-specification-proofs-{run}-1, sha256 {sha}) and were first receipted by that run, which '
            f'receipts all {count} theorems of that head; {tree_txt}')


def rec_short(n, mods_txt, prev_head, prev_run, prev_merge, run, head, merge, base, base_desc, short_tree, **_):
    return (f'; the {n} of {mods_txt}, added after PR head {prev_head} (run {prev_run}, actual tested merge {prev_merge[:8]}…), by run {run}:\n'
            f'PR head {head}, actual tested merge {merge[:8]}… with base {base[:8]}…, {base_desc}, {short_tree}')


def rec_bullet(n, mods_txt, prev_head, prev_run, prev_merge, prev_count, run, head, merge, base, base_desc, count, bullet_tree, **_):
    return (f'; the {n} theorems\n'
            f'  of {mods_txt}, added after PR head {prev_head} (run {prev_run}, actual tested merge {prev_merge[:8]}…, receipt theorem_count {prev_count}), are '
            f'receipted by run {run}: PR head {head},\n'
            f'  actual tested merge {merge[:8]}… with base {base[:8]}…, {base_desc}{bullet_tree}, receipt theorem_count {count}, and were first receipted by that run')


def rec_scope_end(n, mods_txt, run, head, merge, base, base_desc, count, sha, scope_tree, **_):
    return (f'; the {n} of {mods_txt} by run {run}: PR head {head}, actual tested merge {merge[:8]}… with base\n'
            f'{base[:8]}…, {base_desc}{scope_tree}; receipt theorem_count {count}; artifact lean-specification-proofs-{run}-1, sha256 {sha[:8]}…')


SAME_TREE_R45 = dict(tree_txt='same tree as the head', short_tree='same tree as the head', bullet_tree='', scope_tree='')
R4 = dict(n=N4, mods_txt=MODS_TXT, mods_plain=MODS_PLAIN, prev_head=C7, prev_run=RUN7, prev_merge=M7, prev_base=BASE2, prev_count=60,
          relation=f'corroborating the {len(RECEIPTED60)} theorems above on that head', prev_sha=SHA7, run=RUN8, head=C8, merge=M8, base=BASE3,
          base_desc='the head of PR #423', count=73, sha=SHA8, hdr_tree=f'the merge having the same tree as PR head {C8}', **SAME_TREE_R45)
R5 = dict(n=N5, mods_txt=MODS5_TXT, mods_plain=MODS5_PLAIN, prev_head=C9, prev_run=RUN9, prev_merge=M9, prev_base=BASE3, prev_count=73,
          relation=f'corroborating the {len(RECEIPTED73)} theorems above on that head', prev_sha=SHA9, run=RUN10, head=C10, merge=M10, base=BASE4,
          base_desc='the head of PR #434', count=75, sha=SHA10, hdr_tree=f'the merge having the same tree as PR head {C10}', **SAME_TREE_R45)
# round 6: base = Math- main at the merge of PR #440 (PR #441 is not stacked); THE MERGE IS NOT THE HEAD'S TREE, it differs only
# outside both packets; the previous run 38195810463 receipted the 2 round-5 theorems for the first time, so 'corroborating the 75'
# would be false of them: render_docs13's own round-6 relation 'naming the 75 theorems above on that head and not these' is kept
BASE_DESC6 = 'Math- main at the merge of PR #440'
DOC_TREE_LONG = ('the merge differs from its head only outside both packets (main\'s changes from PRs #428 and #440: 13 files under '
                 '`companions/torus_filtration_20261011/` and `.github/workflows/torus-filtration.yml`; this packet and the proof packet are '
                 'byte-identical between head and merge)')
DOC_TREE_SHORT = 'the merge differing from the head only outside both packets'
R6 = dict(n=N6, mods_txt=MODS6_TXT, mods_plain=MODS6_PLAIN, prev_head=C10, prev_run=RUN10, prev_merge=M10, prev_base=BASE4, prev_count=75,
          relation=f'naming the {len(RECEIPTED75)} theorems above on that head and not these', prev_sha=SHA10, run=RUN11, head=C11, merge=M11, base=BASE5,
          base_desc=BASE_DESC6, count=78, sha=SHA11, hdr_tree=f'the merge differing from PR head {C11} only outside both packets',
          tree_txt=DOC_TREE_LONG, short_tree=DOC_TREE_SHORT, bullet_tree=f' ({DOC_TREE_SHORT})', scope_tree=f', {DOC_TREE_SHORT}')
assert HEADER_BIND9 == HEADER_BIND6 + hdr_tail(**R4) and REC4_TXT == rec_txt(**R4)
HEADER_BIND11 = HEADER_BIND9 + hdr_tail(**R5)
REC5_TXT = rec_txt(**R5)
HEADER_BIND14 = HEADER_BIND11 + hdr_tail(**R6)
REC6_TXT = rec_txt(**R6)

# prose phrases (render_docs11/13 verbatim): tier-3/4 phrases; the round-5 receipted ones and the round-6 receipted ones from the templates
PEND3_SHORT = (f'; the {N3} of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4[:8]}…), by run {RUN6}:\n'
               f'PR head {C6}, actual tested merge {M6[:8]}… with base {BASE2[:8]}…, the head of PR #422, same tree as the head')
REC4_SHORT = (f'; the {N4} of {MODS_TXT}, added after PR head {C7} (run {RUN7}, actual tested merge {M7[:8]}…), by run {RUN8}:\n'
              f'PR head {C8}, actual tested merge {M8[:8]}… with base {BASE3[:8]}…, the head of PR #423, same tree as the head')
PEND3_BULLET = (f'; the {N3} theorems\n'
                f'  of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4[:8]}…), are receipted by run {RUN6}: PR head {C6}, actual tested\n'
                f'  merge {M6[:8]}… with base {BASE2[:8]}…, the head of PR #422, and first by run {RUNA}: PR head {C5A}, actual tested merge {M5A[:8]}…, same base')
REC4_BULLET = (f'; the {N4} theorems\n'
               f'  of {MODS_TXT}, added after PR head {C7} (run {RUN7}, actual tested merge {M7[:8]}…, receipt theorem_count 60), are receipted by run {RUN8}: PR head {C8},\n'
               f'  actual tested merge {M8[:8]}… with base {BASE3[:8]}…, the head of PR #423, receipt theorem_count 73, and were first receipted by that run')
PEND3_SCOPE_END = (f'; the {N3} of `P15Interpolation` by run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base\n'
                   f'{BASE2[:8]}…, the head of PR #422')
REC4_SCOPE_END = (f'; the {N4} of {MODS_TXT} by run {RUN8}: PR head {C8}, actual tested merge {M8[:8]}… with base\n'
                  f'{BASE3[:8]}…, the head of PR #423; receipt theorem_count 73; artifact lean-specification-proofs-{RUN8}-1, sha256 {SHA8[:8]}…')
assert (REC4_SHORT, REC4_BULLET, REC4_SCOPE_END) == (rec_short(**R4), rec_bullet(**R4), rec_scope_end(**R4))   # the shape is pinned
REC5_SHORT, REC5_BULLET, REC5_SCOPE_END = rec_short(**R5), rec_bullet(**R5), rec_scope_end(**R5)
REC5_SCOPE = ''           # render_docs11: the boundary sentence needs no receipt clause once nothing is marked
REC6_SHORT, REC6_BULLET, REC6_SCOPE_END = rec_short(**R6), rec_bullet(**R6), rec_scope_end(**R6)
REC6_SCOPE = ''           # likewise for round 6


# the PENDING tier forms of render_docs13 (verbatim), whose committed instances (step 1) are what is replaced
def pend_hdr(head, run, merge, **_):
    return f'; † = added after PR head {head} (run {run}, actual tested merge {merge[:8]}…), no hosted receipt yet'


def pend_txt(n, mods_txt, head, last, run, merge, base, count, relation, sha, **_):
    return (f'; the {n} theorems marked † ({mods_txt}) were added after PR head {head}, {last} (run {run}, actual tested merge {merge} '
            f'with base {base}; receipt theorem_count {count}, {relation}; artifact lean-specification-proofs-{run}-1, '
            f'sha256 {sha}), and have no hosted receipt yet: the next hosted `lean-specification-proofs.yml` run is their first receipt')


def pend_short(n, mods_txt, head, last, run, merge, **_):
    return (f'; the {n} of {mods_txt}, added after PR head {head}, {last} (run {run}, actual tested merge {merge[:8]}…),\n'
            'have no hosted receipt yet and are marked † in the tables')


def pend_bullet(n, mods_txt, head, run, merge, count, **_):
    return (f'; the {n} theorems\n'
            f'  of {mods_txt}, added after PR head {head} (run {run}, actual tested merge {merge[:8]}…, receipt theorem_count {count}), have no hosted receipt yet, marked †')


def pend_scope(head, run, merge, **_):
    return (f', where † marks\n'
            f'a theorem added after PR head {head} (run {run}, actual tested merge {merge[:8]}…) with no hosted receipt yet')


def pend_scope_end(head, run, merge, base, **_):
    return (f'; theorems added after PR head {head} (run {run}, actual tested merge {merge[:8]}… with base {base[:8]}…) await the next hosted\n'
            '`lean-specification-proofs.yml` run, their first receipt')


LAST6 = 'the last head whose receipt the register binds'
P6 = dict(n=N6, mods_txt=MODS6_TXT, head=C10, last=LAST6, run=RUN10, merge=M10, base=BASE4, count=75,
          relation=f'naming the {len(RECEIPTED75)} theorems above on that head and not these', sha=SHA10)
assert P6['relation'] == R6['relation']
HEADER_BIND13 = HEADER_BIND11 + pend_hdr(**P6)
PEND6_TXT = pend_txt(**P6)
PEND6_SHORT, PEND6_BULLET, PEND6_SCOPE, PEND6_SCOPE_END = pend_short(**P6), pend_bullet(**P6), pend_scope(**P6), pend_scope_end(**P6)
REC6_FORMS = {'HEADER_BIND14_tail': hdr_tail(**R6), 'REC6_TXT': REC6_TXT, 'REC6_SHORT': REC6_SHORT, 'REC6_BULLET': REC6_BULLET,
              'REC6_SCOPE_END': REC6_SCOPE_END}
for label, p in REC6_FORMS.items():   # every receipted round-6 form binds head, merge and base, names the tree difference and never 'same tree'
    assert RUN11 in p and C11 in p and M11[:8] in p and BASE5[:8] in p and BASE_DESC6 in p, label
    assert 'only outside both packets' in p and 'same tree' not in p and 'same trees' not in p, label
    assert '†' not in p and 'no hosted receipt' not in p and 'await the' not in p and 'first receipt' not in p.replace('first receipted', ''), label
    assert RUN9 not in p and RUN_STMT11 not in p and RUN_AGG11 not in p, label
    assert p.count('kernel-checked') == (1 if label == 'REC6_TXT' else 0), label     # a receipt word, only in the receipt sentence


