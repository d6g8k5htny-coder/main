import json, subprocess, difflib, re
REPO='<SCRATCH>/repos/math_r5'
S='specifications/lean_statements_20261010/REGISTER.json'
new=json.load(open(f'{REPO}/{S}',encoding='utf-8'))
old=json.loads(subprocess.run(['git','-C',REPO,'show','9e305ea5:'+S],capture_output=True).stdout)
r5=json.loads(subprocess.run(['git','-C',REPO,'show','4afcca6a:'+S],capture_output=True).stdout)
g=lambda r,c:{x['claim_id']:x for x in r['claims']}[c]
def wd(a,b):
    ta,tb=re.findall(r'\S+|\s+',a),re.findall(r'\S+|\s+',b)
    out=[]
    for t,a1,a2,b1,b2 in difflib.SequenceMatcher(None,ta,tb,autojunk=False).get_opcodes():
        if t=='equal':
            s=''.join(ta[a1:a2]); out.append(s if len(s)<300 else s[:120]+' ..... '+s[-120:])
        else:
            if a2>a1: out.append('[-'+''.join(ta[a1:a2])+'-]')
            if b2>b1: out.append('{+'+''.join(tb[b1:b2])+'+}')
    return ''.join(out)
import sys
which=sys.argv[1]
if which=='ns':
    print(wd(g(old,'side24-coefficient')['next_step'], g(new,'side24-coefficient')['next_step']))
    print('\n=== R5 -> NEW next_step')
    print(wd(g(r5,'side24-coefficient')['next_step'], g(new,'side24-coefficient')['next_step']))
elif which=='ke':
    for c in new['claims']:
        o=g(old,c['claim_id'])
        for i,(ke_o,ke_n) in enumerate(zip(o['kernel_evidence'],c['kernel_evidence'])):
            for k in ke_n:
                if ke_n[k]!=ke_o.get(k):
                    print(f"--- {c['claim_id']} ke[{i}].{k}")
                    print(wd(ke_o[k],ke_n[k])[-1800:])
