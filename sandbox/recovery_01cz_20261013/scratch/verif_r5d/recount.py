import re,glob,os
root='<SCRATCH>/repos/math_r5/specifications/'
props=set()
for f in glob.glob(root+'lean_statements_20261010/Specifications/*.lean'):
    src=open(f).read()
    mod=os.path.basename(f)[:-5]
    for m in re.finditer(r'^def (\w+)\b[^:=]*?(?:\([^)]*\)\s*)*:\s*Prop\s*:=',src,re.M):
        props.add(mod+'.'+m.group(1))
# simpler robust: lines starting with def and containing ': Prop :='
props2=set()
for f in glob.glob(root+'lean_statements_20261010/Specifications/*.lean'):
    mod=os.path.basename(f)[:-5]
    src=open(f).read()
    for m in re.finditer(r'^def (\w+)(.*?):=',src,re.M|re.S):
        sig=m.group(2)
        if re.search(r':\s*Prop\s*$',sig.strip()+' ') or sig.strip().endswith(': Prop'):
            props2.add(mod+'.'+m.group(1))
print('props',len(props),len(props2))
thms=[];covered=set()
mods=sorted(glob.glob(root+'lean_statement_proofs_20261011/SpecificationProofs/*.lean'))
for f in mods:
    src=open(f).read()
    # strip Aux namespace blocks
    out=[];depth=None
    lines=src.split('\n');inaux=False
    for ln in lines:
        if re.match(r'^namespace Aux\b',ln): inaux=True; continue
        if re.match(r'^end Aux\b',ln): inaux=False; continue
        if not inaux: out.append(ln)
    s='\n'.join(out)
    for m in re.finditer(r'^theorem (\w+)(.*?):=',s,re.M|re.S):
        sig=m.group(2)
        t=re.search(r'UniversalLaw\.Spec\.(\w+)\.(\w+)',sig)
        thms.append((os.path.basename(f),m.group(1),t.group(1)+'.'+t.group(2) if t else None))
        if t: covered.add(t.group(1)+'.'+t.group(2))
print('modules',len(mods),'theorems',len(thms),'covered',len(covered),'covered subset of props',covered<=props2)
from collections import Counter
print(Counter(p for _,_,p in thms).most_common(2))
print([t for t in thms if 'Side24Cov' in t[0] or 'Side24Cone' in t[0]])
print([t for t in thms if t[2] is None])
