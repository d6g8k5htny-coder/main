import re
S='/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/rev181/'
body=open(S+'review_A.md').read()
lo=open(S+'checkA.log').read(); hi=open(S+'checkA_hi.log').read()
def parse(line):
    d={}
    for key,pat in (('S',r'S=([\d.]+)'),('E',r'E=([\d.]+)'),('O2',r'O2=([\d.]+)'),('inner',r'inner=([\d.]+)'),('Ei',r'E-inner=([\d.]+)')):
        m=re.search(pat,line); d[key]=float(m.group(1))
    return d
rows=[]
alpha={0.5:(1.11821,0.0347454),1.0:(1.32906,0.0255091),2.0:(2.1621,0.0169281)}
mc={0.5:(1.11457,0.034938),1.0:(1.32721,0.0257064),2.0:(2.16004,0.0173858)}
for line in lo.splitlines():
    if line.startswith('k='):
        k=float(line.split(':')[0][2:]); d=parse(line); rows.append((k,'32×32',d))
for line in hi.splitlines():
    if line.startswith('k='):
        parts=line.split()
        k=float(parts[0][2:]); lvl=parts[1].rstrip(':'); d=parse(line); rows.append((k,lvl.replace('x','×'),d))
rows.sort(key=lambda r:(r[0],r[1]))
tab=['| `k` | shape grid | `∫_S` | `α₁+2α₂` (#168 GH60) | `∫_E` | `α₁+α₂` | `∫_{O2}` | `∫_{S2_inner}` | `α₂` (GH60 / MC) | `∫_E − ∫_{S2_inner}` | `α₁` |','|---|---|---|---|---|---|---|---|---|---|---|']
for k,lvl,d in rows:
    a1,a2=alpha[k]; m1,m2=mc[k]
    tab.append(f'| {k:g} | {lvl} | {d["S"]:.4f} | {a1+2*a2:.4f} | {d["E"]:.4f} | {a1+a2:.4f} | {d["O2"]:.5f} | {d["inner"]:.5f} | {a2:.5f} / {m2:.5f} | {d["Ei"]:.4f} | {a1:.4f} |')
body=body.replace('MASS_TABLE','\n'.join(tab))
script=open(S+'checkA.py').read()
body=body.replace('CHECK_SCRIPT',script.replace('```','` ` `'))
open(S+'review_A_final.md','w').write(body)
print(len(body)); print('\n'.join(tab))
