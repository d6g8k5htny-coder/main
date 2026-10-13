

# ----------------------------------------------------------------------------- 0. PIN THE RECEIPTED SHAPE on the committed round-5 receipted docs (4afcca6a)
repro5, _, st5 = render_all(reg5['claims'], set(), RECEIPTED75, HEADER_BIND11, PEND3_TXT + REC4_TXT + REC5_TXT, COMMITTED_R5)
for name in ('README.md', 'SCOPE.md'):
    assert repro5[name] == COMMITTED_R5[name], f'{name}: the renderer does not reproduce the committed 4afcca6a text; refusing to render'
for name, phrases in (('README.md', (REC5_SHORT, REC5_BULLET)), ('SCOPE.md', (REC5_SCOPE_END,))):
    for p in phrases:
        assert COMMITTED_R5[name].count(p) == 1, (name, p[:60])
    assert COMMITTED_R5[name].count(HEADER_BIND11) == len(MODULES) and COMMITTED_R5[name].count(REC5_TXT) == 2, name
    assert '†' not in COMMITTED_R5[name] and 'no hosted receipt' not in COMMITTED_R5[name], name
print('step 0: 4afcca6a README.md and SCOPE.md (round-5 receipted tier) reproduced byte-identically; the receipted templates '
      'reproduce REC4_* (asserted at definition) and REC5_* (found verbatim in the committed text)')

# ----------------------------------------------------------------------------- 1. REPRODUCE the committed 9e305ea5 docs from the committed register
repro, old_bodies, old_stats = render_all(old_reg['claims'], T6, RECEIPTED75, HEADER_BIND13, PEND3_TXT + REC4_TXT + REC5_TXT + PEND6_TXT, COMMITTED)
for name in ('README.md', 'SCOPE.md'):
    assert repro[name] == COMMITTED[name], f'{name}: the renderer does not reproduce the committed 9e305ea5 text; refusing to render'
assert (old_stats['NP'], old_stats['NT'], old_stats['n_def']) == (OLD_NP, OLD_NT, OLD_NDEF)
assert old_stats['entries']['total'] == UPDATE14['before']['statements'] == 357 and old_stats['entries']['proved_total'] == UPDATE14['before']['proved'] == 160
for name, phrases in (('README.md', (PEND6_SHORT, PEND6_BULLET)), ('SCOPE.md', (PEND6_SCOPE, PEND6_SCOPE_END))):
    for p in phrases:
        assert COMMITTED[name].count(p) == 1, (name, p[:60])
    assert COMMITTED[name].count(HEADER_BIND13) == len(MODULES) and COMMITTED[name].count(PEND6_TXT) == 2, name
print('step 1: 9e305ea5 README.md and SCOPE.md (round-6 pending tier, †) reproduced byte-identically from the committed register; '
      'PEND6_* forms pinned')

# ----------------------------------------------------------------------------- 2. RENDER the round-14 state (no pending theorem)
rendered, bodies, st = render_all(new_reg['claims'], set(), RECEIPTED78, HEADER_BIND14, PEND3_TXT + REC4_TXT + REC5_TXT + REC6_TXT, COMMITTED)
NP, NT, n_def = st['NP'], st['NT'], st['n_def']
assert (NP, NT, n_def) == (OLD_NP, OLD_NT, OLD_NDEF) == (77, 78, 141), (NP, NT, n_def)   # FACTS: theorem_count 78, covered_props 77; no count moves
assert st['dups'] == old_stats['dups'] == ['UniversalLaw.Spec.P15.TriangleNeedsThree']
assert st['entries'] == old_stats['entries'] and st['entries']['total'] == UPDATE14['after']['statements'] == 357
assert st['entries']['proved_total'] == UPDATE14['after']['proved'] == 160
assert st['claim_formal_progress'] == old_stats['claim_formal_progress'] == {'specified': 33, 'none': 138}

README_TOP_OLD = (f'Props ({NP} of the {n_def} Props, {NT} theorems); the register lists them under the claim rows as\n'
                  f'statement entries of kind `theorem` with label `proved` ({len(RECEIPTED52)} of them kernel-checked by the hosted run {RUN2}: PR head {C2},\n'
                  f'actual tested merge {M2[:8]}… with base {BASE[:8]}…, same tree as the head{PEND3_SHORT}{REC4_SHORT}{REC5_SHORT}{PEND6_SHORT}),\n')
README_BULLET_OLD = (f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run {RUN2} receipted the 52\n'
                     f'  theorems of PR head {C2} and run {RUN1} the first 38; each receipt\'s checked_commit is the synthetic merge of the PR head into the\n'
                     f'  base that a pull_request run checks out, not the head itself: run {RUN2} PR head {C2}, actual tested merge {M2[:8]}…; run {RUN1}\n'
                     f'  PR head {C1}, actual tested merge {M1[:8]}…; both with base {BASE[:8]}… and each with the same tree as its head{PEND3_BULLET}{REC4_BULLET}{REC5_BULLET}{PEND6_BULLET}).\n')
SCOPE_BOUNDARY_OLD = (f'`specifications/lean_statement_proofs_20261011` ({NT} theorems typed exactly as {NP} of the {n_def} Props;\n'
                      f'the last column of the declaration tables and the `proved (proof packet …)` entries of the claims table{PEND6_SCOPE});\n')
SCOPE_END_OLD = (f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
                 f'hosted run {RUN2} receipted the 52 theorems of PR head {C2}: actual tested merge {M2[:8]}… with base {BASE[:8]}…, the\n'
                 f'synthetic merge a pull_request run checks out, same tree as the head{PEND3_SCOPE_END}{REC4_SCOPE_END}{REC5_SCOPE_END}{PEND6_SCOPE_END}).')
PROSE = {   # each new form is its old form with exactly one round-6 pending clause replaced by its receipted successor
    'README.md': [
        (README_TOP_OLD, once(README_TOP_OLD, f'{REC5_SHORT}{PEND6_SHORT}),\n', f'{REC5_SHORT}{REC6_SHORT}),\n')),
        (README_BULLET_OLD, once(README_BULLET_OLD, f'{REC5_BULLET}{PEND6_BULLET}).\n', f'{REC5_BULLET}{REC6_BULLET}).\n')),
    ],
    'SCOPE.md': [
        (SCOPE_BOUNDARY_OLD, once(SCOPE_BOUNDARY_OLD, f'entries of the claims table{PEND6_SCOPE});\n', f'entries of the claims table{REC6_SCOPE});\n')),
        (SCOPE_END_OLD, once(SCOPE_END_OLD, f'{REC5_SCOPE_END}{PEND6_SCOPE_END}).', f'{REC5_SCOPE_END}{REC6_SCOPE_END}).')),
    ],
}
REC6_PROSE = {'README.md': (REC6_SHORT, REC6_BULLET), 'SCOPE.md': (REC6_SCOPE_END,)}
for name in PROSE:   # every replacement keeps every earlier binding of its anchor and drops every pending form
    for old, new in PROSE[name]:
        assert new != old, (name, old[:60])
        for phrase in (PEND3_SHORT, REC4_SHORT, REC5_SHORT, PEND3_BULLET, REC4_BULLET, REC5_BULLET, PEND3_SCOPE_END, REC4_SCOPE_END, REC5_SCOPE_END,
                       RUN2, RUN1, RUN6, RUNA, RUN7, RUN8, RUN9, RUN10):
            if phrase == RUN10 and old == SCOPE_BOUNDARY_OLD:      # named there only by the dagger legend, which goes (render_docs11 did the same for RUN9)
                assert old.replace(PEND6_SCOPE, '').count(RUN10) == 0 and RUN10 not in new
                continue
            assert phrase not in old or phrase in new, (name, phrase[:60])
        assert '†' not in new and 'no hosted receipt' not in new and 'await the' not in new and 'their first receipt' not in new, (name, new[:60])
        assert new.count('same tree') == old.count('same tree'), (name, new[:60])      # no new same-tree claim
    assert sum(new.count(p) for _, new in PROSE[name] for p in REC6_PROSE[name]) == len(REC6_PROSE[name]), name

PAIRS = tuple((run, merge[:8]) for run, merge in RUN_MERGE)
NONE_YET_REGISTER = sum(r['next_step'].count('none yet') + r['not_established'].count('none yet') for r in new_reg['claims'])   # 'proof lane: none yet', not a receipt statement
assert NONE_YET_REGISTER == 1 and 'Hosted receipt: none yet' not in json.dumps(new_reg) and 'no hosted receipt' not in json.dumps(new_reg)
assert '†' not in json.dumps(new_reg) and 'await the next hosted' not in json.dumps(new_reg)
T6_SHORTS = [t.split('.', 3)[-1] for t in T6_EXPECTED.values()]                     # ImageBound, EntrywiseCovarianceBound, CovarianceOrdering
T6_CELLS_OLD = {t.split('.', 3)[-1]: '`' + p.split('.', 2)[-1] + '†`' for p, t in T6_EXPECTED.items()}   # Prop short name -> committed proofs cell
T6_CELLS_NEW = {t.split('.', 3)[-1]: '`' + p.split('.', 2)[-1] + '`' for p, t in T6_EXPECTED.items()}    # ... and the receipted one
old_by_id = {r['claim_id']: r for r in old_reg['claims']}
new_by_id = {r['claim_id']: r for r in new_reg['claims']}
# the register's receipted round-6 phrases (update_register14) as the SCOPE claims table transcribes them
REG6 = UPDATE14['phrases']
REG6_PLANNED = [cell(REG6['TIER6']), cell(REG6['RC_ITEM6']), cell(REG6['RC_TAIL_NEW'])]


def check_section_lines(name, idx, old_body, new_body):
    """Every changed line of a regenerated section is enumerated, and differs from the committed line by exactly the round-6 binding."""
    ol, nl = old_body.split('\n'), new_body.split('\n')
    assert len(ol) == len(nl), (name, idx, len(ol), len(nl))
    seen = Counter()
    for o, n in zip(ol, nl):
        if o == n:
            continue
        if o.startswith('| declaration |'):                                         # table header: the pending legend -> the round-6 receipt
            assert once(o, HEADER_BIND13, HEADER_BIND14) == n, (name, o[-120:])
            seen['header'] += 1
        elif idx == 0 and o == old_stats['intro']:                                  # declarations intro
            assert once(o, PEND6_TXT, REC6_TXT) == n == st['intro'], name
            seen['intro'] += 1
        elif idx == 1 and o == old_stats['claim_intro']:                            # claims intro
            assert once(o, PEND6_TXT, REC6_TXT) == n == st['claim_intro'], name
            seen['claim_intro'] += 1
        elif idx == 0 and any(o.startswith(f'| `{s}` | def_prop |') for s in T6_SHORTS):   # the three Props' rows: the dagger goes
            s = next(s for s in T6_SHORTS if o.startswith(f'| `{s}` | def_prop |'))
            assert o.endswith(' | ' + T6_CELLS_OLD[s] + ' |') and n == o[:-len(T6_CELLS_OLD[s] + ' |')] + T6_CELLS_NEW[s] + ' |', (name, o[:80])
            seen['prop_row'] += 1
        elif idx == 1 and any(o.startswith(f'| `{cid}` |') for cid in CHANGED_ROWS):   # the claim rows the register update changed
            cid = next(c for c in CHANGED_ROWS if o.startswith(f'| `{c}` |'))
            e = o
            if cid == T6_ROW:
                for s in T6_SHORTS:
                    e = once(e, T6_CELLS_OLD[s], T6_CELLS_NEW[s])
            if name == 'SCOPE.md':                                                    # register cells transcribed verbatim
                for f in ('not_established', 'next_step'):
                    if old_by_id[cid][f] != new_by_id[cid][f]:
                        e = once(e, ' | ' + cell(old_by_id[cid][f]) + ' |', ' | ' + cell(new_by_id[cid][f]) + ' |')
            assert e == n, (name, cid)
            seen['claim_row:' + cid] += 1
        else:
            raise AssertionError(f'{name} section {idx}: unexpected changed line: {o[:120]!r}')
    return seen


summary = {'scientific_effect': 'NONE', 'reproduction_of_round5_receipted_docs_4afcca6a': 'byte-identical',
           'reproduction_of_committed_docs_9e305ea5': 'byte-identical', 'hand_edits_folded_in': [], 'files': {}}
for name in ('README.md', 'SCOPE.md'):
    orig = COMMITTED[name]
    if '†' not in orig or 'no hosted receipt yet' not in orig:
        sys.exit(f'REFUSED: {name} carries no pending (†) form: the committed docs are not the round-13 ones')
    text = rendered[name]
    for old, new in PROSE[name]:
        assert text.count(old) == 1, (name, old[:60])
        text = text.replace(old, new)
    # byte-for-byte preservation outside the regenerated sections, modulo exactly the PROSE edits
    expect = outside(orig, HEADINGS[name])
    for old, new in PROSE[name]:
        assert expect.count(old) == 1, (name, 'prose anchor inside a regenerated section', old[:60])
        expect = expect.replace(old, new)
    assert outside(text, HEADINGS[name]) == expect, name
    # inside the regenerated sections: every changed line enumerated and exactly edited
    seen = Counter()
    for idx in (0, 1):
        seen.update(check_section_lines(name, idx, old_bodies[name][idx], bodies[name][idx]))
    exp_seen = {'header': len(MODULES), 'intro': 1, 'claim_intro': 1, 'prop_row': 3, 'claim_row:' + T6_ROW: 1}
    if name == 'SCOPE.md':
        exp_seen.update({'claim_row:' + c: 1 for c in CHANGED_ROWS if c in [r['claim_id'] for r in st['other_rows']] and c != T6_ROW
                         and any(old_by_id[c][f] != new_by_id[c][f] for f in ('not_established', 'next_step'))})
        assert 'claim_row:math.side24-coefficient' in exp_seen and len(exp_seen) == 6, exp_seen
    assert dict(seen) == exp_seen, (name, dict(seen), exp_seen)
    assert not re.search(r'<<[^<>\n]*>>', text), name
    for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW', PEND3_TXT, REC4_TXT, REC5_TXT, REC6_TXT, HEADER_BIND9, HEADER_BIND11,
                   HEADER_BIND14, RUN11, M11, SHA11, BASE5, RUN10, M10, SHA10, BASE4, RUN9, M9) + REC6_PROSE[name]:
        assert phrase in text, (name, phrase[:60])
    # the pending forms are gone: no dagger, no 'no hosted receipt', no 'await the ...', no receipt-'none yet'
    for phrase in (PEND6_TXT, PEND6_SHORT, PEND6_BULLET, PEND6_SCOPE, PEND6_SCOPE_END, HEADER_BIND13, LAST6, 'whose receipt this register binds',
                   'Hosted receipt: none yet', 'the last head with a hosted receipt', '†', 'no hosted receipt', 'await the', 'their first receipt',
                   'first receipt for them'):
        assert phrase not in text, (name, phrase[:60])
    # 'not evidence about any commit': the README's own local-build sentences stay; the register's PENDING6 cells (SCOPE) go
    _nec = 'not evidence about any commit'
    _nec_cells = sum(cell(old_by_id[c][f]).count(_nec) for c in [r['claim_id'] for r in st['other_rows']] for f in ('not_established', 'next_step'))
    assert sum(cell(new_by_id[c][f]).count(_nec) for c in new_by_id for f in ('not_established', 'next_step')) == 0
    assert text.count(_nec) == orig.count(_nec) - (_nec_cells if name == 'SCOPE.md' else 0), (name, text.count(_nec), orig.count(_nec))
    assert (name, _nec_cells) != ('SCOPE.md', 0) and _nec_cells == 2, _nec_cells
    assert text.count('none yet') == (NONE_YET_REGISTER if name == 'SCOPE.md' else 0), (name, text.count('none yet'))   # SCOPE renders next_step cells
    # no new same-tree claim anywhere; the tree difference is stated in every receipted round-6 form
    assert text.count('same tree') == orig.count('same tree') and text.count('same trees') == orig.count('same trees'), name
    # C422-E1: every mention of a hosted run binds its actual tested merge, per paragraph and per table cell; runs 38180705952,
    # 38187538744, 38195810463 and 38277458954 also name their PR head wherever they appear; every mention of 38277458954 names its
    # base and the tree difference; every mention of 38277458954 is a planned phrase; no unlisted run id; no companion run cited
    for para in re.split(r'\n\s*\n', text):
        for run, merge in PAIRS:
            if run in para:
                assert merge in para, (name, run, para[:100])
        for run in (RUN8, RUN9, RUN10, RUN11):
            if run in para:
                assert RUN_HEAD[run] in para, (name, run, para[:100])
        if RUN11 in para:
            assert BASE5[:8] in para and 'only outside both packets' in para, (name, para[:100])
            stripped = para
            for p in sorted(list(REC6_FORMS.values()) + REG6_PLANNED, key=len, reverse=True):
                stripped = stripped.replace(p, '')
            assert RUN11 not in stripped, (name, stripped[stripped.index(RUN11) - 200:stripped.index(RUN11) + 100])
    for line in text.split('\n'):
        if line.startswith('|'):
            for c in line.split(' | '):
                for run, merge in PAIRS:
                    assert run not in c or merge in c, (name, run, c[-160:])
                for run in (RUN8, RUN9, RUN10, RUN11):
                    assert run not in c or RUN_HEAD[run] in c, (name, run, c[-200:])
                if RUN11 in c:     # a header cell carries the receipted legend (with the tree difference); any other cell is a
                    assert BASE5[:8] in c, (name, c[-200:])      # register cell transcribed verbatim, naming the run only in planned phrases
                    if line.startswith('| declaration |'):
                        assert hdr_tail(**R6) in c and 'only outside both packets' in c, (name, c[-200:])
                    else:
                        assert name == 'SCOPE.md', (name, c[:120])
                        stripped = c
                        for p in sorted(REG6_PLANNED, key=len, reverse=True):
                            stripped = stripped.replace(p, '')
                        assert RUN11 not in stripped, (name, c[:120])
    for found in set(re.findall(r'\b38\d{9}\b', text)):
        assert found in RUNS or found == WORKSPACE_LANDING_RUN, (name, found)
    for run in (RUN_STMT, RUN_AGG, RUN_STMT10, RUN_AGG10, RUN_STMT11, RUN_AGG11):
        assert run not in text, (name, run)
    # `kernel-checked` stays a receipt word: added only by REC6_TXT (the two intros) and, in SCOPE, by the register's receipted tier phrase
    cells_delta = sum(cell(new_by_id[c][f]).count('kernel-checked') - cell(old_by_id[c][f]).count('kernel-checked')
                      for c in [r['claim_id'] for r in st['other_rows']] for f in ('not_established', 'next_step')) if name == 'SCOPE.md' else 0
    assert text.count('kernel-checked') - orig.count('kernel-checked') == 2 * REC6_TXT.count('kernel-checked') + cells_delta, name
    assert cells_delta == (2 * REG6['TIER6'].count('kernel-checked') if name == 'SCOPE.md' else 0), (name, cells_delta)
    # every round-6 theorem is listed exactly as the round-3/4/5 ones are: backticked name, no mark
    decl_body, claim_body = bodies[name]
    assert '†' not in decl_body and '†' not in claim_body
    for t in T6:
        sh = t.split('.', 2)[-1]
        assert decl_body.count('`' + sh + '`') == 1 and claim_body.count('`' + sh + '`') == 1, (name, t)
        assert text.count('`' + sh + '`') == orig.count('`' + sh + '†`') and ('`' + sh + '`') not in orig, (name, t)
    for t in RECEIPTED75:     # every earlier theorem keeps its committed form and count
        sh = t.split('.', 2)[-1]
        assert ('`' + sh + '`') in decl_body and text.count('`' + sh + '`') == orig.count('`' + sh + '`'), (name, t)
    assert text.count(HEADER_BIND14) == len(MODULES) and orig.count(HEADER_BIND13) == len(MODULES), name
    assert text.count(REC6_TXT) == 2 and orig.count(PEND6_TXT) == 2, name        # declarations intro and claims intro
    (DOCS_OUT / name).write_text(text, encoding='utf-8')
    summary['files'][name] = {'bytes': len(text.encode()), 'bytes_before': len(orig.encode()), 'dagger_marks': text.count('†'),
                              'dagger_marks_before': orig.count('†'), 'changed_lines_in_regenerated_sections': dict(seen),
                              'prose_edits': len(PROSE[name]), 'changed': text != orig, 'written_to': str(DOCS_OUT / name),
                              'sha256': hashlib.sha256(text.encode()).hexdigest(), 'git_blob': git_blob(text.encode())}
    print(name, len(text.encode()), 'bytes', f'({text.count("†")} † marks; {orig.count("†")} before)')
summary.update({'props_with_proof': NP, 'proof_theorems': NT, 'def_props': n_def, 'pending_receipt': 0, 'receipted_round6': sorted(T6),
                'round6_modules': list(T6_MODULES), 'round6_statement_entries': N_T6_ENTRIES,
                'committed_register_counts': {'props': OLD_NP, 'theorems': OLD_NT, 'def_props': OLD_NDEF}, 'duplicates': st['dups'],
                'tiers': {'T1_T2_receipted_52': len(RECEIPTED52), 'T3_P15Interpolation': N3, 'T4_run_38180705952': N4, 'T5_run_38195810463': N5,
                          'T6_run_38277458954': N6},
                'entries': st['entries'], 'claim_rows': st['claim_rows'], 'claim_formal_progress': st['claim_formal_progress'],
                'changed_register_rows': CHANGED_ROWS,
                'run_merge_bindings_asserted': dict(RUN_MERGE), 'run_head_bindings_asserted': RUN_HEAD,
                'receipt': {'run': RUN11, 'pr': 441, 'pr_head': EXPECTED_HEAD, 'checked_commit_merge': M11, 'base': BASE5, 'base_description': BASE_DESC6,
                            'merge_tree_differs_from_head_only_outside_both_packets': True, 'theorem_count': 78,
                            'artifact_sha256_github_digest': SHA11},
                'template_parameters_round6_vs_round5': {k: [R5[k], R6[k]] for k in ('base_desc', 'relation', 'hdr_tree', 'tree_txt', 'short_tree',
                                                                                       'bullet_tree', 'scope_tree')},
                'phrases': {**REC6_FORMS, 'REC6_SCOPE': REC6_SCOPE}})
(OUT / 'render_summary14.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'phrases'}, indent=1, ensure_ascii=False))
