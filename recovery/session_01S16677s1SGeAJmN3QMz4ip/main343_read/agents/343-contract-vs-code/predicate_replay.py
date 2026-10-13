# Recompute every gate/classification from the saved endpoints and the frozen module.
import json, sys, hashlib
from fractions import Fraction as Q
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).parent))
import population_certificate as pc
import run_population_certificate as runner
res = Path(sys.argv[2])
cert = runner.load_json(res/'CERTIFICATE.json'); run = runner.load_json(res/'RUN.json')
def I(x): return (Q(x[0]), Q(x[1]))
problems = []
def check(name, cond):
    print(('OK  ' if cond else 'BAD ') + name)
    if not cond: problems.append(name)
# output hashes
for name in ('CERTIFICATE.json','RESULTS.md'):
    raw=(res/name).read_bytes(); check(f'RUN.outputs[{name}] hash/bytes match file', run['outputs'][name]['sha256']==hashlib.sha256(raw).hexdigest() and run['outputs'][name]['bytes']==len(raw))
check('RUN sources_before==sources_after', run['sources_before']==run['sources_after'] and run['source_drift']==[])
check('RUN fixed hashes equal FIXED_HASHES', all(run['sources_before'][k]['sha256']==v for k,v in runner.FIXED_HASHES.items()))
check('canonical re-serialization of CERTIFICATE reproduces bytes', runner.canonical_bytes(json.loads((res/'CERTIFICATE.json').read_text()))==(res/'CERTIFICATE.json').read_bytes())
b = I(cert['coefficient_interval']); check('coefficient_interval replay', pc.coefficient_interval()==b)
tgt = I(cert['poisson_target_interval']); check('poisson_target replay', pc.poisson_target()==tgt)
check('target width <= 2^-16', tgt[1]-tgt[0] <= Q(1,1<<16))
statuses=[]
for row in cert['rows']:
    j=row['j']; n=row['n']; tau=Q(row['tau']); d=row['direct']
    check(f'j{j} schedule', tau==Q(1,1<<(3*j)) and n==4**j and n*Q(1,4**j)==1)
    rlo,rhi = I(d['root_bracket'])
    check(f'j{j} root bracket replay', pc.root_bracket(tau)==(rlo,rhi))
    g=d['gates']
    exp_gates={'root_width': rhi-rlo<=Q(1,1<<96),'root_positive': rlo>0,'root_upper': rhi<Q(1,2),'root_auxiliary_lower': rhi>Q(1,1<<9),'core_scale': n*rlo**2<=Q(1,2),'core_monotonicity': 27-45*rlo**2+2*rlo**4>0,'core_range': Q(3,5)*n*rlo**2<=Q(3,10)}
    for k,v in exp_gates.items(): check(f'j{j} gate {k} recomputed=={g[k]}=={v}', g[k]==v==True)
    co = I(d['tail_coefficient_interval'])
    check(f'j{j} tail_coefficient gate', g['tail_coefficient']==(co[1]<=Q(1,30))==True)
    check(f'j{j} tail_second_derivative gate', g['tail_second_derivative']==(co[1]*Q(1485,16)<=Q(99,32))==True)
    aw=Q(d['arithmetic_width']); comps={k:Q(v) for k,v in d['arithmetic_components'].items()}
    check(f'j{j} arithmetic components sum == arithmetic_width', sum(comps.values(),Q(0))==aw)
    check(f'j{j} arithmetic_width <= 2^-22', aw<=Q(1,1<<22) and g['arithmetic_width'])
    dc={k:Q(v) for k,v in d['discretization_components'].items()}
    check(f'j{j} discretization sum', dc['core']+dc['tail']==Q(d['discretization_width']))
    check(f'j{j} core_discretization <= 3/(10*2^16)', 0<=dc['core']<=Q(3,10*65536) and g['core_discretization'])
    check(f'j{j} tail_discretization <= 297/512^3', 0<=dc['tail']<=Q(297,512**3) and g['tail_discretization'])
    sl=Q(d['sliver_width']); check(f'j{j} sliver == 3/5 n (rhi-rlo)', sl==Q(3,5)*n*(rhi-rlo))
    ci=I(d['core_interval']); ti=I(d['tail_interval']); ni=I(d['normalized_interval'])
    check(f'j{j} normalized = core+tail(+sliver)', ni==(ci[0]+ti[0], ci[1]+ti[1]+sl))
    check(f'j{j} normalized width < 2^-17', ni[1]-ni[0]<Q(1,1<<17) and Q(d['normalized_width'])==ni[1]-ni[0] and g['normalized_width'])
    check(f'j{j} interval = normalized/n', I(d['interval'])==(ni[0]/n, ni[1]/n))
    octs = 0; x=Q(1); fin=1/rhi
    while x<fin: x=min(2*x,fin); octs+=1
    planned = 65537 + octs*1025 - (octs-1)
    check(f'j{j} evaluations {d["evaluations"]} == unique-node plan {planned} (octaves {octs}) <= 80000', d['evaluations']==planned<=80000 and d['partial']['completed_octaves']==octs and d['partial']['core_nodes']==65537 and d['partial']['tail_nodes']==planned-65537)
    cdf=I(row['cdf_mass']); check(f'j{j} cdf_mass replay', pc.cdf_mass(tau)==cdf)
    cn=I(row['cdf_normalized']); check(f'j{j} cdf_normalized = n*cdf', cn==(n*cdf[0],n*cdf[1]))
    check(f'j{j} cdf width gate', row['cdf_width_gate']==(cn[1]-cn[0]<=Q(1,1<<16))==True)
    obs=I(row['intersection_normalized']); check(f'j{j} intersection', obs==(max(ni[0],cn[0]),min(ni[1],cn[1])) and obs[0]<=obs[1])
    mass=I(row['intersection_mass']); check(f'j{j} mass = obs/n', mass==(obs[0]/n,obs[1]/n))
    e=I(row['error_interval']); check(f'j{j} error replay', pc.error_interval(j)==e and e==(Q(9,8*n)+pc.quarter_power_bounds(Q(1,1<<(7*j)))[0], Q(9,8*n)+pc.quarter_power_bounds(Q(1,1<<(7*j)))[1]))
    env=row['predicted_envelopes']; inner=I(env['inner']); outer=I(env['outer'])
    check(f'j{j} envelopes', inner==(b[1]-e[0],b[0]+e[0]) and outer==(b[0]-e[1],b[1]+e[1]))
    cs = 'PASS' if (obs[0]>=inner[0] and obs[1]<=inner[1]) else ('FALSIFIED' if (obs[1]<outer[0] or obs[0]>outer[1]) else 'INCONCLUSIVE_PRECISION')
    check(f'j{j} coefficient classification {row["coefficient_status"]} == recomputed {cs} == module {pc.classify_coefficient(obs,b,e)}', row['coefficient_status']==cs==pc.classify_coefficient(obs,b,e))
    print(f'    j{j} margins: Llo-innerlo={float(obs[0]-inner[0]):.3e} innerhi-Lhi={float(inner[1]-obs[1]):.3e}  E=[{float(e[0]):.3e},{float(e[1]):.3e}] |L-B|~{float(abs((obs[0]+obs[1])/2-(b[0]+b[1])/2)):.3e}')
    check(f'j{j} outer distance', Q(row['outer_envelope_distance'])==pc.interval_distance(obs,outer))
    vi=I(row['void_interval']); check(f'j{j} void replay', pc.void_interval(mass,n)==vi)
    check(f'j{j} void width gate', row['void_width_gate']==(vi[1]-vi[0]<=Q(1,1<<16))==True and Q(row['void_width'])==vi[1]-vi[0])
    check(f'j{j} target row == global', I(row['target_interval'])==tgt and row['target_width_gate'])
    if j==8:
        fv=row['finite_void']
        check('j8 error_allowance e_hi<=41/524288', e[1]<=Q(41,524288) and fv['gates']['error_allowance'])
        check('j8 mass gate obs_hi<1/3', obs[1]<Q(1,3) and fv['gates']['mass_gate'])
        ed=max(abs(vi[0]-tgt[1]),abs(vi[1]-tgt[0])); check('j8 endpoint distance replay', Q(fv['endpoint_distance'])==ed)
        check('j8 endpoint distance < 2^-13', ed<Q(1,8192)); print('    j8 endpoint distance', float(ed), ' vs 2^-13 =', 1/8192, ' ratio', float(ed*8192))
        check('j8 min distance', Q(fv['minimum_distance'])==pc.interval_distance(vi,tgt))
        check('j8 void classification', fv['status']==pc.classify_void(vi,tgt)=='PASS')
        # Actual |V - exp(-B)| guarantee math: |nm - B| <= E? Check using certified intervals
        print('    j8 n*m interval', [float(x) for x in obs], 'B', [float(x) for x in b], 'max |nm-B|', float(max(abs(obs[0]-b[1]),abs(obs[1]-b[0]))), 'E_hi', float(e[1]))
    check(f'j{j} row status', row['status']=='PASS')
    statuses.append(row['status'])
check('overall', cert['status']==runner._overall(statuses)=='PASS' and run['status']=='PASS' and run['exit_code']==0)
check('negative controls replay (status and gates)', pc.negative_controls()['status']=='PASS' and all(cert['negative_controls']['gates'].values()))
nc = pc.negative_controls()
check('negative_controls JSON equals fresh computation', runner._json_value(nc)==cert['negative_controls'])
check('guards JSON equals fresh computation', runner._json_value(pc.guard_witnesses())==cert['rational_guards'])
check('actual_route_evaluations', run['actual_route_evaluations']==sum(r['direct']['evaluations'] for r in cert['rows']))
print('\nPROBLEMS:', problems)
