"""Own reconstruction of the receipts' five mutations; each copy must exit 1 at the named check, the restored tree must exit 0."""
import os, shutil, subprocess, sys
O = sys.argv[1]
SRC = os.path.join(O, 'stage')
def run(tree):
    d = os.path.join(tree, 'K3_SIDE24_LB/UPPER2D/D1_assembly')
    p = subprocess.run(['/usr/bin/python3', '-E', '-s', 'd1_falsify_v4.py'], cwd=d, capture_output=True, text=True)
    fails = [l for l in p.stdout.splitlines() if l.startswith('FAIL ')]
    return p.returncode, (fails[0] if fails else p.stdout.splitlines()[-1])
def edit(tree, rel, old, new, count=1):
    p = os.path.join(tree, 'K3_SIDE24_LB/UPPER2D', rel)
    s = open(p, encoding='utf-8').read()
    assert s.count(old) >= 1, (rel, old)
    open(p, 'w', encoding='utf-8').write(s.replace(old, new, count))
def premlist_insert(tree, text):
    p = os.path.join(tree, 'K3_SIDE24_LB/UPPER2D/D1_assembly/D1_ASSEMBLY_v2_3_DRAFT.md')
    s = open(p, encoding='utf-8').read()
    i = s.find('VALIDITY premises'); j = s.find('\n', i)
    open(p, 'w', encoding='utf-8').write(s[:j + 1] + text + '\n' + s[j + 1:])
MUTS = {
    'M1_slipped_bracket': ('C-bracket-recompute-EXACT', lambda t: edit(t, 'D1_assembly/d1_falsify_v4.py', 'B_REMOTE_EXACT = Decimal("21.927883016")', 'B_REMOTE_EXACT = Decimal("21.92788306")')),
    'M2_retired_premise_reinserted': ('F-only-two-premises', lambda t: premlist_insert(t, '- **PERC-DECAY** (reinserted)')),
    'M3_retracted_LPW_token': ('F-retracted-LPW-token-absent', lambda t: edit(t, 'D1_assembly/D1_ASSEMBLY_v2_3_DRAFT.md', 'END_FROZEN_BODY', '1.1357762846347807215698804\nEND_FROZEN_BODY')),
    'M4_B4LOC_body_tampered': ('A-body B4LOC_damline/B4LOC_DAMLINE.md', lambda t: edit(t, 'B4LOC_damline/B4LOC_DAMLINE.md', 'BEGIN_FROZEN_BODY\n', 'BEGIN_FROZEN_BODY\nX')),
    'M5_identification_reasserted': ('F-identification-not-carried', lambda t: premlist_insert(t, 'ASSERTED, NOT ESTABLISHED')),
}
for name, (tag, fn) in MUTS.items():
    t = os.path.join(O, 'mut_' + name)
    shutil.rmtree(t, ignore_errors=True); shutil.copytree(SRC, t)
    fn(t)
    rc, first = run(t)
    print(f'{name}: exit {rc}; first FAIL = {first!r}; expected tag {tag!r}: {"OK" if rc == 1 and first == "FAIL " + tag else "DIFFERS"}')
    shutil.rmtree(t)
rc, last = run(SRC)
print('restored:', rc, last)
