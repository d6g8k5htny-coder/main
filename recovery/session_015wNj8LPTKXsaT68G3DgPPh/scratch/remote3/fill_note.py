import json
P='/home/user/wt_remote3/frontiers/c6_remote_coefficient_d3_numerics_20260930/'
r=json.load(open(P+'RESULTS.json')); s=open(P+'NOTE.template.md').read()
ff=r['far_field']; c=r['controls']
rep={'FF0':f"{ff['0.0']['lambda']:.8f}",'FF1':f"{ff['1.0']['lambda']:.8f}",
 'FF0J':'`'+', '.join(f"{v:.6f}" for v in ff['0.0']['lambda_j'])+'`','FF1J':'`'+', '.join(f"{v:.6f}" for v in ff['1.0']['lambda_j'])+'`',
 'EDET':f"{c['E_absdet_G3_quadrature']:.10f}", 'CPDJ':'`'+', '.join(f"{v:.10f}" for v in c['critical_point_density_by_index'])+'`',
 'M30':f"{c['m_3_0']:.10f}",'M31':f"{r['z0']['k=1.0,b=1.0']/36:.10f}",'R30':f"{r['R_3']['0.0']:.10f}",'R31':f"{r['R_3']['1.0']:.10f}"}
rows=[]
for k in (0.5,1.0,2.0):
    for b in (0.0,1.0):
        key=f'k={k},b={b}'; H=r['hole'][key]; T=r['remote_total'][key]; A=r['assembled_nu'][key]
        kk={0.5:'1/2',1.0:'1',2.0:'2'}[k]
        rows.append(f"| `{kk}` | `{b:g}` | {H['hole']:.3f} ± {H['hole_se']:.3f} | {T['L=12']['approx_L3_lambda_inf_plus_hole']:.2f} | {T['L=24']['approx_L3_lambda_inf_plus_hole']:.2f} | {A['a1_d3']:.4g} | {A['nu1_L']['L=24']:.1f} | {A['nu2_over_nu1_L']['L=24']:.2e} |")
rep['TABLE_MAIN']='\n'.join(rows)
hj=[]
for b in (0.0,1.0):
    H=r['hole'][f'k=1.0,b={b}']; hj.append(f"`b = {b:g}`: `"+', '.join(f"{v:.3f} ± {e:.3f}" for v,e in zip(H['hole_j'],H['hole_j_se']))+"`")
rep['HOLEJ']='; '.join(hj)
pr=r['ray_profiles']['k=1.0,b=0.0']
def prof(name, rhos):
    return ', '.join(f"`{row['ratio_to_far']:.3g}` at `{row['rho']:g}`" for row in pr[name] if row['rho'] in rhos)
rep['RATIO_AXIS']=prof('axis',(2.5,3.0,4.0,5.0)); rep['RATIO_TRANS']=prof('transverse',(0.5,1.0,2.0,3.0,4.0)); rep['RATIO_DIAG']=prof('diag45',(1.0,2.0,2.5,3.0,4.0))
rep['FARMC']='`'+', '.join(f"{v:.3f}" for v in c['far_mc_over_quadrature_b0.0'])+'`'
g=r['grid_convergence']['k=1.0,b=0.0']; H4=r['hole']['k=1.0,b=0.0']
rep['HOLE06']=f"{g['hole_h=0.6,Rc=6']['hole']:.3f} ± {g['hole_h=0.6,Rc=6']['hole_se']:.3f}"; rep['HOLE03']=f"{g['hole_h=0.3,Rc=6']['hole']:.3f} ± {g['hole_h=0.3,Rc=6']['hole_se']:.3f}"
rep['HOLE04']=f"{H4['hole']:.3f} ± {H4['hole_se']:.3f}"; rep['HOLERC5']=f"{g['hole_h=0.4,Rc=5']['hole']:.3f} ± {g['hole_h=0.4,Rc=5']['hole_se']:.3f}"
for k_,v in sorted(rep.items(), key=lambda kv:-len(kv[0])): s=s.replace(k_,v)
open(P+'NOTE.md','w').write(s); print(rep['TABLE_MAIN']); print(rep['HOLEJ']); print(rep['HOLE06'],rep['HOLE04'],rep['HOLE03'],rep['HOLERC5']); print(rep['FARMC'])
