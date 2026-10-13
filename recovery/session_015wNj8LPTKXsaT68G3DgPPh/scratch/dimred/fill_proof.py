import json, re
P='/home/user/wt_dimred/frontiers/c6_hard_direction_factorization_20260930/'
r=json.load(open(P+'RESULTS.json')); t=r['table']; mc=r['monte_carlo']
rows=[]
for d in (2,3,4,5):
    for b in (0.0,1.0):
        key='d=%d'%d; bk='b=%s'%b
        m=mc['m_d'].get('d=%d,b=%s'%(d,b))
        mcs=f"{m['value']:.4f} ± {m['se']:.4f}" if m else '(exact `m_(2,b)`)'
        rows.append(f"| {d} | {b:g} | {t['N_d'][key][bk]:.10f} | {t['m_d'][key][bk]:.10f} | {mcs} | {t['R_d'][key][bk]:.10f} |")
tabR='\n'.join(rows)
n3=mc['N_3']; mcn3='; '.join(f"`b = {float(k.split('=')[1]):g}`: `{v['value']:.4f} ± {v['se']:.4f}` against `{t['N_d']['d=3'][k]:.4f}`" for k,v in n3.items())
rowsA=[]
asm=r['assembled_near_coefficients']
def keyf(k):
    ks,bs,ds=k.split(','); return (float(eval(ks.split('=')[1])), float(bs.split('=')[1]), int(ds.split('=')[1]))
for k in sorted(asm, key=keyf):
    kk,b,d=keyf(k); v=asm[k]
    rowsA.append(f"| {k.split(',')[0][2:]} | {b:g} | {d} | {v['R_d']:.6f} | {v['a1']:.5g} | {v['a2']:.5g} | {v['a1_plus_a2']:.5g} | {v['a2_over_a1']:.5f} | {v['z0_d']:.4f} |")
tabA='\n'.join(rowsA)
rowsZ=[]
for k,v in sorted(r['finite_r_normalizer'].items(), key=lambda kv:(kv[1]['d'],kv[1]['b'],-kv[1]['r'])):
    rowsZ.append(f"| {v['d']} | {v['b']:g} | {v['r']} | {v['Z_r_over_r2']:.3f} | {v['se']:.3f} | {v['limit_36k2_m_d']:.4f} |")
tabZ='\n'.join(rowsZ)
s=open(P+'PROOF.template.md').read().replace('TABLE_R',tabR).replace('MC_N3',mcn3).replace('TABLE_A',tabA).replace('TABLE_Z',tabZ)
open(P+'PROOF.md','w').write(s); print(tabR); print(mcn3); print(tabA); print(tabZ)
