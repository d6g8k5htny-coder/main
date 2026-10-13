import json
exec(open('phrases.py').read().split('PH=[')[0])   # reuse independently built T6, T6ne, W6T, N6, W6 definitions
P_NOTE=(f'3 ({names3}) were added in round 6 on top of Math- main 4afcca6a, the merge of PR #439, after PR head c6ce0256 (the last head whose receipt this register binds: run 38195810463, actual tested merge {M5} with base {B5}, receipt theorem_count 75): '
        'locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {propext, Classical.choice, Quot.sound} on an unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them.')
P_NE=(f'3 of them ({names3}) were added in round 6 on top of Math- main 4afcca6a, the merge of PR #439, after PR head c6ce0256 (the last head whose receipt this register binds: run 38195810463, actual tested merge {M5} with base {B5}, receipt theorem_count 75): '
        'locally `lake env lean -DwarningAsError=true` exit 0 and `#print axioms` within {propext, Classical.choice, Quot.sound} on an unverified worktree, not evidence about any commit; the next hosted lean-specification-proofs.yml run is the first receipt for them)')
P_RW=(f'the theorems added in round 6 after PR head c6ce0256 (module Side24Images; run 38195810463 (PR head c6ce0256; actual tested merge {M5} with base {B5}), receipt theorem_count 75, does not name them) have no hosted receipt yet: the next hosted lean-specification-proofs.yml run is their first receipt')
oc=o['claims']; nc=n['claims']
on=oc[0]['kernel_evidence'][1]['note']; assert on.count(P_NOTE)==1
print('note full reconstruction:', on.replace(P_NOTE, T6+' '+N6)==nc[0]['kernel_evidence'][1]['note'])
ow=oc[0]['kernel_evidence'][1]['receipt_or_workflow']; assert ow.count(P_RW)==1 and ow.endswith(P_RW)
print('rw full reconstruction:', ow.replace(P_RW, W6T+'; '+W6)==nc[0]['kernel_evidence'][1]['receipt_or_workflow'])
for ci in (0,16):
    s=oc[ci]['not_established']; assert s.count(P_NE)==1, ci
    exp=s.replace(P_NE, T6ne+')')
    print('NE', ci, 'full reconstruction:', exp==nc[ci]['not_established'])
