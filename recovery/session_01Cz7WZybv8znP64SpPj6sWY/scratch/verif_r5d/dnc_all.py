import json,re
base='<SCRATCH>/repos/math_r5/specifications/lean_statement_proofs_20261011/'
n=json.load(open('<SCRATCH>/verif_r5d/new.json'))
cnt=0;bad=0
for c in n['claims']:
    for s in c['statements']:
        if 'lean_statement_proofs_20261011' not in s['package']: continue
        cnt+=1
        src=open(base+s['module']).read()
        thm=s['declaration'].split('.')[-1]
        m=re.search(r'/--((?:(?!/--).)*?)-/\s*theorem '+re.escape(thm)+r'\b',src,re.S)
        doc=m.group(1)
        f=' '.join(doc.split('Does not establish:',1)[1].split())
        ok = f==s['does_not_claim']
        if not ok:
            bad+=1; print('MISMATCH',c['claim_id'],s['declaration'])
        # anchor check: compare with statement entry of same Prop
print(cnt,'proof entries; mismatches',bad)
