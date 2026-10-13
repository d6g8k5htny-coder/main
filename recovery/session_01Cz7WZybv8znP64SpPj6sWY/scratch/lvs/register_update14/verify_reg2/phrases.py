import json, re
o=json.load(open('REGISTER.committed.json')); n=json.load(open('REGISTER.new.json'))
R='38277458954'; H='9e305ea5'; M='1376175ce5536a6ac74cb03706d18e65fdb695fa'; B='a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc'
SHA='1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79'; ART=f'lean-specification-proofs-{R}-1'
CAV=', the tested merge differing from the head only outside both packets'
M5='1320c5b217e491d301ea140e5cc25b07ff5a21b7'; B5='be01479435f98f34b802258ee9ec3c3578f37169'; SHA5='d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
TREE6="the merge differs from its head only outside both packets (main's changes from PRs #428 and #440: 13 files under companions/torus_filtration_20261011/ and .github/workflows/torus-filtration.yml; specifications/lean_statements_20261010 and specifications/lean_statement_proofs_20261011 are byte-identical between head and merge)"
names3='Side24Images.imageBound, Side24Images.entrywiseCovarianceBound, Side24Images.covarianceOrdering'
names2='Side24Covariance.referenceCovarianceLowerBound, Side24ConeMoment.coneMomentD2Identity'
# round-5 tier phrase as it sits in the new (and 32fc) note
T5=(f'2 ({names2}) were added after PR head be014794 and kernel-checked by the hosted run 38195810463 (PR head c6ce0256; actual tested merge {M5} with base {B5}) '
    f'(job proof-evidence: fresh build, leanchecker PASS, `#print axioms` within {{propext, Classical.choice, Quot.sound}}, receipt formalization_status kernel-checked, theorem_count 75, worktree CLEAN; '
    f'artifact lean-specification-proofs-38195810463-1, sha256 {SHA5}); they were first receipted by that run (PR #439, event pull_request); base {B5} is the head of PR #434, the base of PR #439, and the merge has the same tree as its head.')
nn=n['claims'][0]['kernel_evidence'][1]['note']
assert nn.count(T5)==1
T6=(T5.replace(f'2 ({names2}) were added after PR head be014794', f'3 ({names3}) were added in round 6 on top of Math- main 4afcca6a, the merge of PR #439, after PR head c6ce0256')
      .replace('38195810463', R).replace('c6ce0256;', f'{H};').replace(M5, M).replace(B5, B).replace('theorem_count 75', 'theorem_count 78').replace(SHA5, SHA)
      .replace('PR #439, event', 'PR #441, event').replace('is the head of PR #434, the base of PR #439, and the merge has the same tree as its head.', f'is Math- main at the merge of PR #440, the base of PR #441, and {TREE6}.'))
print('tier phrase in note:', nn.count(T6))
# not_established form: lowercase '3 of them (' and ends ')' w/o period
T6ne = T6.replace(f'3 ({names3})', f'3 of them ({names3})')[:-1]
print('NE0 form', n['claims'][0]['not_established'].count(T6ne), 'NE16', n['claims'][16]['not_established'].count(T6ne))
# WF tier
rw=n['claims'][0]['kernel_evidence'][1]['receipt_or_workflow']
W5T=(f'the theorems added after PR head be014794 (modules Side24Covariance, Side24ConeMoment) are receipted by run 38195810463 (PR head c6ce0256; actual tested merge {M5} with base {B5}) '
     f'(artifact lean-specification-proofs-38195810463-1, sha256 {SHA5}; receipt formalization_status kernel-checked, leanchecker PASS, theorem_count 75, worktree CLEAN) and were first receipted by that run; base {B5} is the head of PR #434, the base of PR #439; the merge has the same tree as its head')
assert rw.count(W5T)==1
W6T=(W5T.replace('the theorems added after PR head be014794 (modules Side24Covariance, Side24ConeMoment)', 'the theorems added in round 6 after PR head c6ce0256 (module Side24Images)')
       .replace('38195810463', R).replace('c6ce0256;', f'{H};').replace(M5, M).replace(B5, B).replace('theorem_count 75', 'theorem_count 78').replace(SHA5, SHA)
       .replace('is the head of PR #434, the base of PR #439; the merge has the same tree as its head', f'is Math- main at the merge of PR #440, the base of PR #441; {TREE6}'))
print('WF tier phrase in rw:', rw.count(W6T))
N6=f'All 78 theorems of PR head {H}, including every theorem of the tiers above, are receipted on that head by run {R} (actual tested merge {M} with base {B}{CAV}; receipt theorem_count 78; artifact {ART}, sha256 {SHA}).'
W6=f'all 78 theorems of PR head {H} receipted on that head by run {R} (actual tested merge {M} with base {B}{CAV}; receipt theorem_count 78; artifact {ART}, sha256 {SHA})'
RCI=f'the 3 added in round 6 after PR head c6ce0256 ({names3}) first by run {R} (PR head {H}; actual tested merge {M} with base {B}{CAV}) (receipt theorem_count 78)'
RCT=f'each of those 22 theorems receipted on PR head {H} by run {R} (actual tested merge {M} with base {B}{CAV}; receipt theorem_count 78)'
PH=[T6, T6ne, W6T, N6, W6, RCI, RCT]
for p in PH:
    assert H in p and M in p and B in p and (CAV in p or TREE6 in p) and 'same tree' not in p and 'the merge differing' not in p
def leaves(x, p=()):
    if isinstance(x, dict):
        for k,v in x.items(): yield from leaves(v, p+(k,))
    elif isinstance(x, list):
        for i,v in enumerate(x): yield from leaves(v, p+(i,))
    elif isinstance(x, str): yield p, x
tot=0; uncovered=0; per={}
for p, s in leaves(n):
    c=s.count(R)
    if not c: continue
    tot+=c
    rest=s
    for ph in sorted(PH, key=len, reverse=True):
        k=rest.count(ph)
        if k: per[ph[:40]]=per.get(ph[:40],0)+k
        rest=rest.replace(ph,'\x00')
    if R in rest:
        uncovered+=rest.count(R); print('UNCOVERED', p)
print('mentions', tot, 'uncovered', uncovered); print(per)
# counts
def cnt(r, t): return sum(s.count(t) for _, s in leaves(r))
for t in [CAV, TREE6, 'only outside both packets', 'the merge differing', 'every theorem of this row receipted on PR head 9e305ea5', 'each of those 22 theorems', 'same tree', 'the merge has the same tree as its head', 'kernel-checked', R, H, M, B, SHA, '1376175c', 'every theorem of this row']:
    print(f'{cnt(o,t):5d} -> {cnt(n,t):5d}  {t[:80]!r}')
