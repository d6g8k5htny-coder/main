import json
o=json.load(open('REGISTER.committed.json')); n=json.load(open('REGISTER.new.json'))
H='9e305ea5'; M='1376175ce5536a6ac74cb03706d18e65fdb695fa'; B='a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc'; R='38277458954'
SHA='1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79'; ART=f'lean-specification-proofs-{R}-1'
CAV=', the tested merge differing from the head only outside both packets'
M5='1320c5b217e491d301ea140e5cc25b07ff5a21b7'; B5='be01479435f98f34b802258ee9ec3c3578f37169'; SHA5='d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff'
N5=f'All 75 theorems of PR head c6ce0256, including every theorem of the tiers above, are receipted on that head by run 38195810463 (actual tested merge {M5} with base {B5}; receipt theorem_count 75; artifact lean-specification-proofs-38195810463-1, sha256 {SHA5}).'
N6=f'All 78 theorems of PR head {H}, including every theorem of the tiers above, are receipted on that head by run {R} (actual tested merge {M} with base {B}{CAV}; receipt theorem_count 78; artifact {ART}, sha256 {SHA}).'
W5=f'all 75 theorems of PR head c6ce0256 receipted on that head by run 38195810463 (actual tested merge {M5} with base {B5}; receipt theorem_count 75; artifact lean-specification-proofs-38195810463-1, sha256 {SHA5})'
W6=f'all 78 theorems of PR head {H} receipted on that head by run {R} (actual tested merge {M} with base {B}{CAV}; receipt theorem_count 78; artifact {ART}, sha256 {SHA})'
rows={2:0,5:1,6:1,7:1,8:1,17:0,25:1,26:1,54:0}
for ci,ki in rows.items():
    oe=o['claims'][ci]['kernel_evidence'][ki]; ne=n['claims'][ci]['kernel_evidence'][ki]
    on, nn = oe['note'], ne['note']
    assert on.count(N5)==1, (ci,'N5')
    exp = on.replace(N5, N5+' '+N6)
    ok_note = exp==nn
    ow, nw = oe['receipt_or_workflow'], ne['receipt_or_workflow']
    assert ow.count(W5)==1, (ci,'W5')
    expw = ow.replace(W5, W5+'; '+W6)
    ok_rw = expw==nw
    # what follows W5 in old?
    print(ci, o['claims'][ci]['claim_id'], ki, 'note ok', ok_note, 'rw ok', ok_rw, 'old W5 at end:', ow.endswith(W5), repr(on[on.find(N5)+len(N5):on.find(N5)+len(N5)+40]))
# side24 row: the corroboration part
ne=n['claims'][0]['kernel_evidence'][1]
print('side24 note has N6 once:', ne['note'].count(N6), 'rw has W6 once:', ne['receipt_or_workflow'].count(W6), 'rw endswith W6', ne['receipt_or_workflow'].endswith(W6))
