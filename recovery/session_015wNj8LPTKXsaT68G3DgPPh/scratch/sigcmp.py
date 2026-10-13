import re,subprocess,sys
def src(ref,path): return subprocess.check_output(['git','show',f'{ref}:{path}'],text=True)
def blocks(t):
    t=re.sub(r'/-.*?-/','',t,flags=re.S); t=re.sub(r'--.*','',t)
    out={}
    for m in re.finditer(r'theorem\s+(\S+)(.*?)(?=\ntheorem|\nend Research)',t,flags=re.S):
        body=m.group(2); i=body.find(':=')
        out[m.group(1)]=(' '.join(body[:i].split()),' '.join(body[i:].split()))
    return out
new=blocks(src('pr279','formal/ResearchFormalCoreR1/MomentGenerality.lean'))
p6=blocks(src('pr277','reviews/lean_cycle2_20261005/agent6/MomentTail_finite_MT_N2.lean.txt'))
p7=blocks(src('pr278','reviews/lean_cycle2_20261005/agent7/MomentTail_family_interval_MT_N1.lean.txt'))
old=blocks(src('pr279','formal/ResearchFormalCoreR1/MomentTail.lean'))
for n,(s,p) in new.items():
    prop=p6.get(n) or p7.get(n)
    print(n,'TYPE==proposal:',prop[0]==s,'PROOF==proposal:',prop[1]==p)
# compare with predecessors
print('tail_finite vs moment40_tail type differs only in instance:', new['p02_lm009_moment40_tail_finite'][0].replace('IsFiniteMeasure','IsProbabilityMeasure')==old['p02_lm009_moment40_tail'][0].replace('p02_lm009_moment40_tail ',''))
print(repr(new['p02_lm009_moment40_tail_finite'][0][:80])); print(repr(old['p02_lm009_moment40_tail'][0][:80]))
print('tail_of_finite type == old tail type:', new['p02_lm009_moment40_tail_of_finite'][0]==old['p02_lm009_moment40_tail'][0])
print('family_of_interval type == old family type:', new['p02_lm009_moment40_family_r3_of_interval'][0]==old['p02_lm009_moment40_family_r3'][0])
print('finite proof == old tail proof:', new['p02_lm009_moment40_tail_finite'][1]==old['p02_lm009_moment40_tail'][1])
print('interval proof vs old family proof:'); print(new['p02_lm009_moment40_family_r3_interval'][1]); print(old['p02_lm009_moment40_family_r3'][1])
