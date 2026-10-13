

OLD_NDEF, OLD_NP, OLD_NT = counts(collect(old_reg['claims']))
assert (OLD_NDEF, OLD_NP, OLD_NT) == (141, 77, 78), (OLD_NDEF, OLD_NP, OLD_NT)
assert counts(collect(reg5['claims'])) == (141, 74, 75)
assert counts(collect(new_reg['claims'])) == (141, 77, 78)
new_decls = collect(new_reg['claims'])
for t, prop in T6_EXPECTED.items():    # each round-6 theorem proves exactly its registered Prop, as committed
    holders = [k for k in new_decls if t in new_decls[k]['proofs']]
    assert [k[1] for k in holders] == [prop] and new_decls[holders[0]]['proofs'] == [t], (t, holders)
    assert collect(reg5['claims'])[holders[0]]['proofs'] == []
pp_new = [s for r in new_reg['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE]
T3 = {s['declaration'] for s in pp_new if s['module'] == T3_MODULE}
T4 = {s['declaration'] for s in pp_new if s['module'] in T4_MODULES}
T5 = {s['declaration'] for s in pp_new if s['module'] in T5_MODULES}
T6 = {s['declaration'] for s in pp_new if s['module'] in T6_MODULES}
assert len(T3) == 8 and len(T4) == 13 and len(T5) == 2 and T6 == set(T6_EXPECTED)
assert not (T3 & T4 or T4 & T5 or T3 & T5 or T6 & (T3 | T4 | T5)) and not ((T3 | T4 | T5 | T6) & RECEIPTED52)
RECEIPTED60 = RECEIPTED52 | T3
RECEIPTED73 = RECEIPTED60 | T4
RECEIPTED75 = RECEIPTED73 | T5
RECEIPTED78 = RECEIPTED75 | T6        # after this binding every theorem of the committed packet has a hosted receipt
assert len(RECEIPTED73) == 73 and len(RECEIPTED75) == 75 and len(RECEIPTED78) == 78
assert {s['declaration'] for r in reg5['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE} == RECEIPTED75
assert {s['declaration'] for r in old_reg['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE} == RECEIPTED78
assert {s['declaration'] for s in pp_new} == RECEIPTED78
assert UPDATE14['receipt']['theorem_count'] == 78 and UPDATE14['receipt']['covered_props'] == 77 and UPDATE14['receipt']['run'] == RUN11
assert UPDATE14['receipt']['checked_commit_merge'] == M11 and UPDATE14['receipt']['base'] == BASE5 and UPDATE14['receipt']['pr_head'] == EXPECTED_HEAD
assert UPDATE14['receipt']['artifact_sha256_github_digest'] == SHA11 and UPDATE14['receipt']['both_packets_identical_between_head_and_merge'] is True
N3, N4, N5, N6 = len(T3), len(T4), len(T5), len(T6)
N_T6_ENTRIES = sum(1 for s in pp_new if s['declaration'] in T6)
assert N_T6_ENTRIES == 3
stems = lambda mods: [PurePosixPath(m).stem for m in mods]


