"""Exact observed graph selection and complete-field ordinary H0 coupling.

Geometry source: DESIGN_PROOF.md SHA256
5a20d81b57a216212f88940b79317f1e3593456855914f206a031e50dfb7767d.
Unexecuted implementation; Root separately admits frozen program invocations.
"""
from fractions import Fraction as Q
from collections import deque
import field
import exact_h0 as h0
A=Q(1,65536)
SCALE=2**80

def _require(ok,message):
    if not ok: raise ValueError(message)

def _cell(c):
    field.cell_mass(c,8)
    return Q(c['Qa']),Q(c['Qb']),Q(c['Za']),Q(c['Zb']),c['epsilon']

def _f_bounds(x,c):
    qa,qb,za,zb,sign=_cell(c)
    # f_Z=-sign*16*Q^3*Z^7 has fixed sign on the entire control cell.
    # On each Z edge, f_Q=6*Q*(x+sign*Q*(1-Z^8)). Its only possible
    # interior extremum is the exact rational Qstar below. This retains
    # Q-term correlation, unlike separate interval bounds on Q^2 and Q^3.
    candidates=[]
    for z in (za,zb):
        a=1-z**8
        points=[qa,qb]
        if a:
            star=-x/(sign*a)
            if qa<star<qb:points.append(star)
        candidates.extend(-x**3+3*q*q*x+sign*2*q**3*a for q in points)
    return min(candidates),max(candidates)

def _root_box(c,left,right,increasing,rounds):
    fl,fr=_f_bounds(left,c),_f_bounds(right,c)
    if not ((fl[1]<0 and fr[0]>0) if increasing else (fl[0]>0 and fr[1]<0)):return None,0
    lo,hi=left,right
    for _ in range(rounds):
        mid=(lo+hi)/2;f=_f_bounds(mid,c)
        if (f[1]<0 if increasing else f[0]>0):lo=mid
        else:hi=mid
    lower=lo;lo,hi=left,right
    for _ in range(rounds):
        mid=(lo+hi)/2;f=_f_bounds(mid,c)
        if (f[0]>0 if increasing else f[1]<0):hi=mid
        else:lo=mid
    return (lower,hi),2*rounds

def uniform_roots(cell,max_bisections=512):
    qa,qb,za,zb,sign=_cell(cell)
    _require(type(max_bisections) is int and 2<=max_bisections<=512,'Root bisection cap')
    out={'status':'UNRESOLVED_ROOTS','intervals':[],'bisections':0,'max_bisections_per_root':max_bisections}
    if qa==0 or za==0 or zb==1:out['reason']='POSITIVE_MASS_BOUNDARY_CELL';return out
    for left,right,inc in ((Q(-3),-qb,False),(-qa,qa,True),(qb,Q(3),False)):
        box,cost=_root_box(cell,left,right,inc,min(48,max_bisections//2));out['bisections']+=cost
        if box is None:out['reason']='WHOLE_CELL_SIGNS_NOT_SEPARATED';return out
        out['intervals'].append(box)
    out['status']='ROOTS_CERTIFIED'
    if not verify_root_cover(cell,out):out['status']='UNRESOLVED_ROOTS';out['reason']='ROOT_COVER_UNCERTAIN'
    return out

def verify_root_cover(cell,certificate):
    try:
        qa,qb,za,zb,sign=_cell(cell);boxes=certificate['intervals']
        if certificate.get('status')!='ROOTS_CERTIFIED' or len(boxes)!=3:return False
        if not all(type(x) is Q for box in boxes for x in box):return False
        if not all(len(box)==2 and -3<=box[0]<box[1]<=3 and box[1]-box[0]<=1 for box in boxes):return False
        if not boxes[0][1]<boxes[1][0]<boxes[1][1]<boxes[2][0]:return False
        if not (boxes[0][1]<-qb and -qa<boxes[1][0] and boxes[1][1]<qa and qb<boxes[2][0]):return False
        for i,(left,right) in enumerate(boxes):
            fl,fr=_f_bounds(left,cell),_f_bounds(right,cell)
            if i==1:
                if not (fl[1]<0 and fr[0]>0):return False
            elif not (fl[0]>0 and fr[1]<0):return False
        return True
    except (ValueError,TypeError,KeyError,IndexError):return False

def integer_sample(x,y,control,scale_bits=80):
    _require(type(scale_bits) is int and scale_bits==80,'Frozen integer scale80 required')
    v=max(field.scaled_core_value(x,y,control),Q(-16))*SCALE
    return v.numerator//v.denominator

def build_grid(cell,roots,scale_bits=80):
    _require(verify_root_cover(cell,roots),'No uniform three-root cover')
    _require(type(scale_bits) is int and scale_bits==80,'Frozen integer scale80 required')
    qa,qb,za,zb,sign=_cell(cell);c=field.control((qa+qb)/2,(za+zb)/2,sign)
    xs=[Q(-8)]+[x for box in roots['intervals'] for x in box]+[Q(8)]
    ys=list(map(Q,[-16,-8,-4,-2,0,2,4,16]));residuals=[];values=[];division=[]
    for x in xs:
        for y in ys:
            v=field.scaled_core_value(x,y,c);floored=max(v,Q(-16))*SCALE
            quotient,remainder=divmod(floored.numerator,floored.denominator)
            residuals.append(v);values.append(quotient);division.append((floored.numerator,floored.denominator,quotient,remainder))
    return {'n':8,'x':xs,'y':ys,'axis_y_index':4,'scale_bits':80,'control':c,'values':values,'residuals':residuals,'integer_divisions':division,'whole_field_proof_sha256':'5a20d81b57a216212f88940b79317f1e3593456855914f206a031e50dfb7767d'}

def _neighbors(v,n):
    x,y=divmod(v,n)
    return (((x-1)%n)*n+y,((x+1)%n)*n+y,x*n+(y-1)%n,x*n+(y+1)%n)

def _reach(values,n,start,level):
    if values[start]<level:return set(),{}
    seen={start};parents={start:None};queue=deque([start])
    while queue:
        v=queue.popleft()
        for w in _neighbors(v,n):
            if w not in seen and values[w]>=level:seen.add(w);parents[w]=v;queue.append(w)
    return seen,parents

def _maxima(values,n):
    seen=set();out=[]
    for start in range(n*n):
        if start in seen:continue
        plateau={start};seen.add(start);queue=deque([start]);higher=False
        while queue:
            v=queue.popleft()
            for w in _neighbors(v,n):
                if values[w]>values[start]:higher=True
                if values[w]==values[start] and w not in seen:seen.add(w);plateau.add(w);queue.append(w)
        if not higher:out.append({'vertex':min(plateau),'value':values[start],'vertices':sorted(plateau)})
    return out

def _older(a,b,values):return values[a]>values[b] or (values[a]==values[b] and a<b)

def verify_merge_witness(values,n,barcode,witness):
    try:
        if h0.verify_by_connectivity(values,n,barcode) is False:return False
        v=witness['birth_vertex'];older=witness['older_vertex'];death=witness['death_level']
        records=dict(zip(barcode['birth_vertices'],barcode['intervals']))
        if v not in records or records[v][1]!=death or values[v]!=records[v][0]:return False
        maxima={p['vertex'] for p in _maxima(values,n)}
        if older not in maxima or not _older(older,v,values):return False
        path=witness['path']
        if not path or path[0]!=v or path[-1]!=older:return False
        if not all(type(x) is int and 0<=x<n*n and values[x]>=death for x in path):return False
        if not all(b in _neighbors(a,n) for a,b in zip(path,path[1:])):return False
        reachable,_=_reach(values,n,v,death+1)
        if witness['cut_vertices']!=sorted(reachable):return False
        if any(_older(w,v,values) for w in maxima&reachable):return False
        return True
    except (ValueError,TypeError,KeyError,IndexError):return False

def observe_grid(values,n):
    barcode=h0.compute(values,n)
    _require(h0.verify_by_connectivity(values,n,barcode) is not False,'UF/BFS disagreement')
    plateaus=_maxima(values,n);witnesses=[]
    for v,(birth,death) in zip(barcode['birth_vertices'],barcode['intervals']):
        reachable,parents=_reach(values,n,v,death)
        old=[p['vertex'] for p in plateaus if p['vertex'] in reachable and _older(p['vertex'],v,values)]
        _require(bool(old),'Observed death has no older plateau')
        older=min(old,key=lambda w:(-values[w],w));path=[older]
        while path[-1]!=v:path.append(parents[path[-1]])
        path.reverse();cut,_=_reach(values,n,v,death+1)
        witness={'birth_vertex':v,'death_level':death,'older_vertex':older,'path':path,'cut_vertices':sorted(cut)}
        _require(verify_merge_witness(values,n,barcode,witness),'Invalid observed merge witness');witnesses.append(witness)
    return {'barcode':barcode,'independent_bfs_verified':True,'regional_maximum_plateaus':plateaus,'merge_witnesses':witnesses}

def _couple(cell,roots,grid,observation,mode):
    _require(mode in ('tight','conservative'),'Unknown error mode')
    if not verify_root_cover(cell,roots):return {'status':'UNRESOLVED','reason':'ROOTS'}
    if grid!=build_grid(cell,roots):return {'status':'IMPLEMENTATION_FAILURE','reason':'GRID_SOURCE_MISMATCH'}
    _require(field.realization_descriptor()['whole_field_defined'],'Complete field missing')
    values=grid['values'];barcode=observation['barcode']
    if h0.verify_by_connectivity(values,8,barcode) is False:return {'status':'IMPLEMENTATION_FAILURE','reason':'UF_BFS'}
    if len(barcode['intervals'])!=1 or len(barcode['essential'])!=1:return {'status':'UNRESOLVED','reason':'OBSERVED_PAIR_COUNT'}
    qa,qb,za,zb,sign=_cell(cell);boxes=roots['intervals'];h=max(r-l for l,r in boxes);dQ=(qb-qa)/2;dZ=(zb-za)/2
    X=max(abs(x) for box in boxes for x in box);M2=3*qb**2+3*X**2;MQ=3*qb*X**2+6*qb**2*(1-za**8)*X;MZ=16*qb**3*zb**7*X
    error=(M2*h*h/2+MQ*dQ+MZ*dZ+Q(1,SCALE)) if mode=='tight' else (15*h*h+45*dQ+48*dZ+Q(1,SCALE))
    birth,death=barcode['intervals'][0];essential=barcode['essential'][0]
    if Q(birth-death,SCALE)<=2*error or Q(essential-birth,SCALE)<=2*error:return {'status':'UNRESOLVED','reason':'ROBUST_GAPS','endpoint_error_scaled':error,'refinement_terms':{'Q':MQ*dQ,'Z':MZ*dZ}}
    v=barcode['birth_vertices'][0];old=barcode['essential_vertices'][0];i,j=divmod(v,8);oi,oj=divmod(old,8)
    if j!=4 or oj!=4 or not ((i in (1,2) and oi in (5,6)) or (oi in (1,2) and i in (5,6))):return {'status':'IMPLEMENTATION_FAILURE','reason':'OUTER_PEAK_LOCALIZATION'}
    witnesses=observation['merge_witnesses']
    if len(witnesses)!=1 or not verify_merge_witness(values,8,barcode,witnesses[0]):return {'status':'IMPLEMENTATION_FAILURE','reason':'ACTUAL_DEATH_WITNESS'}
    witness=dict(witnesses[0]);path=witness['path'];middle=(28,36);bottleneck=min(middle,key=lambda x:(values[x],x))
    if death<=-16*SCALE or death!=values[bottleneck] or bottleneck not in path:return {'status':'IMPLEMENTATION_FAILURE','reason':'MIDDLE_BOTTLENECK_OR_FLOOR'}
    columns=[x//8 for x in path]
    if not (3 in columns and 4 in columns) or 0 in columns or 7 in columns:return {'status':'IMPLEMENTATION_FAILURE','reason':'PATH_STRIP_OR_SEAM'}
    birth_index=0 if i in (1,2) else 2;mark=field.physical_mark(boxes[birth_index],boxes[1])
    if not mark['resolved']:return {'status':'UNRESOLVED','reason':'PHYSICAL_LOCATION'}
    witness.update(axis_path=columns,bottleneck_vertex=bottleneck,bottleneck_columns=[3,4],saddle_root_index=1)
    return {'status':'RESOLVED','verified':True,'mode':mode,'endpoint_error_scaled':error,'error_constants':{'h':h,'dQ':dQ,'dZ':dZ,'X':X,'M2':M2,'MQ':MQ,'MZ':MZ,'rounding':Q(1,SCALE)},'refinement_terms':{'Q':MQ*dQ,'Z':MZ*dZ},'birth_root_index':birth_index,'saddle_root_index':1,'actual_merge_witness':witness,'physical_mark':mark,'lifetime_scaled':(Q(birth-death,SCALE)-2*error,Q(birth-death,SCALE)+2*error),'centered_birth_scaled':(Q(birth,SCALE)-error,Q(birth,SCALE)+error),'separation_over_A':tuple(x/A for x in mark['separation']),'zero_hidden_positive_bars_basis':'Reviewed complete-field fiber equivalence and full cyclic critical inventory'}

def couple_cell(cell,roots,grid,observation,mode='tight'):return _couple(cell,roots,grid,observation,mode)

def verify_coupling_certificate(cell,roots,grid,observation,coupling):
    try:
        expected=_couple(cell,roots,grid,observation,coupling['mode'])
        return expected.get('status')=='RESOLVED' and coupling==expected
    except (ValueError,TypeError,KeyError,IndexError):return False

def certify_cell(cell,limits):
    _cell(cell);_require(limits.get('tensor_side',8)==8 and limits.get('total_vertices',64)>=64,'64-node budget required')
    roots=uniform_roots(cell,limits.get('root_bisections',512));costs={'attempted_fields':1,'sampled_vertices':0,'root_bisections':roots['bisections']}
    out={'status':'UNRESOLVED','cell':dict(cell),'roots':roots,'costs':costs}
    if roots['status']!='ROOTS_CERTIFIED':out['reason']=roots.get('reason','ROOTS');return out
    grid=build_grid(cell,roots);costs['sampled_vertices']=len(grid['values']);observation=observe_grid(grid['values'],8)
    coupling=couple_cell(cell,roots,grid,observation,limits.get('error_mode','tight'));out.update(grid=grid,observation=observation,coupling=coupling,status=coupling['status'])
    if coupling['status']=='RESOLVED':
        _require(verify_coupling_certificate(cell,roots,grid,observation,coupling),'Coupling replay mismatch')
        for key in ('lifetime_scaled','centered_birth_scaled','separation_over_A','refinement_terms'):out[key]=coupling[key]
    elif 'refinement_terms' in coupling:out['refinement_terms']=coupling['refinement_terms']
    return out
