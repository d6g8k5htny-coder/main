"""Independent analytic prediction; actual D02 predict/validate RED precedes authorship.
No field, observation roots, UF, BFS, matching error, IO or subprocess is used.
Outputs require separate source/runtime/freeze/terminal and artifact qualification.
"""
from fractions import Fraction as Q
from math import isqrt
import hashlib
import json
import artifact as exact_artifact

A=Q(1,65536)
ANALYTIC_SHA256='e1dd749117f141a0f05e9c6c36bcac154efa137164177a0304f741632796a3c5'
KS=(8,12,15,16)
FORBIDDEN={'observed_cells','uf_results','uniform_root_certificates','matching_epsilon'}

def _q(x):
    if type(x) not in (Q,int,str):
        raise ValueError('Exact rational required, not bool/float')
    return Q(x)

def _clip(x,lo=Q(0),hi=Q(1)):
    a,b=max(lo,x[0]),min(hi,x[1])
    if a>b:
        raise ValueError('Inconsistent domain intersection')
    return a,b

def _add(x,y):
    return x[0]+y[0],x[1]+y[1]

def _sub(x,y):
    return x[0]-y[1],x[1]-y[0]

def _mul(x,y):
    v=(x[0]*y[0],x[0]*y[1],x[1]*y[0],x[1]*y[1])
    return min(v),max(v)

def _div(x,y):
    if y[0]<=0:
        raise ValueError('Positive interval denominator required')
    return _mul(x,(1/y[1],1/y[0]))

def _pow(x,n):
    if x[0]<0:
        raise ValueError('Nonnegative power interval required')
    return x[0]**n,x[1]**n

def _sqrt(x,bits):
    # Integer square comparisons prove lo^2<=x<=hi^2, width<=2^-bits.
    if x<0:
        raise ValueError('Negative radicand')
    scale=1<<bits
    m=isqrt((x.numerator*scale*scale)//x.denominator)
    lo=Q(m,scale)
    return (lo,lo) if lo*lo==x else (lo,Q(m+1,scale))

def _root(x,n,steps):
    lo,hi=Q(0),Q(1)
    for _ in range(steps):
        mid=(lo+hi)/2
        if mid**n==x:
            return mid,mid
        if mid**n<x:
            lo=mid
        else:
            hi=mid
    return lo,hi

def _atan(x):
    total=sum(((-1 if n%2 else 1)*x**(2*n+1)/(2*n+1) for n in range(96)),Q(0))
    return total,total+x**193/193

def _constants(bits):
    x,y=_atan(Q(1,5)),_atan(Q(1,239))
    return _sqrt(Q(3,2),bits),(16*x[0]-4*y[1],16*x[1]-4*y[0])

def _asin(x):
    if x<0:
        a,b=_asin(-x)
        return -b,-a
    if x>=Q(1,16384):
        raise ValueError('Outside proved arcsin argument domain')
    coefficient,power,total=Q(1),x,Q(0)
    for n in range(17):
        total+=coefficient*power
        coefficient*=Q((2*n+1)**2,2*(n+1)*(2*n+3))
        power*=x*x
    return total,total+x**35/(1-x*x)

def _rho(x,constants):
    root,pi=constants
    arg=_mul((A,A),_mul(x,root))
    value=_div((_asin(arg[0])[0],_asin(arg[1])[1]),pi)
    allowance=Q(1,2**120)
    return value[0]-allowance,value[1]+allowance

def _roots(z,cfg):
    bits=cfg['precision_bits']
    if z==0:
        return dict(p=(Q(1),Q(1)),e=(Q(3),Q(3)),n=(Q(1),Q(1)),
                    D=(Q(0),Q(0)),lifetime=(Q(0),Q(0)))
    if z==1:
        p=_sqrt(Q(3),bits)
        return dict(p=p,e=p,n=(Q(0),Q(0)),D=p,lifetime=(Q(9,4),Q(9,4)))
    lo,hi,a=Q(1),Q(7,4),1-z**8
    for _ in range(cfg['max_root_bisections']):
        mid=(lo+hi)/2
        sign=mid**3-3*mid+2*a
        if sign==0:
            lo=hi=mid
            break
        if sign<0:
            lo=mid
        else:
            hi=mid
    p=lo,hi
    e=(_sqrt(12-3*hi**2,bits)[0],_sqrt(12-3*lo**2,bits)[1])
    n=_clip(((e[0]-hi)/2,(e[1]-lo)/2))
    gap=_clip(_div((6*(lo*lo-1),6*(hi*hi-1)),(3*lo+e[0],3*hi+e[1])),Q(0),Q(2))
    eight_l=_clip(_mul(_add(p,e),_pow(gap,3)),Q(0),Q(18))
    return dict(p=p,e=e,n=n,D=gap,lifetime=(eight_l[0]/8,eight_l[1]/8))

def _life_cap(s,tau,steps):
    if s<=tau:
        return Q(1),Q(1)
    return (Q(0),Q(0)) if tau==0 else _root(tau/s,4,steps)

def _metric_cap(z,roots,radius,cfg,constants):
    if radius is None or z==0:
        return Q(1),Q(1)
    if radius==0:
        return Q(0),Q(0)
    def metric(q):
        if q==0:
            return Q(0),Q(0)
        v=_sub(_rho(_mul((q,q),roots['p']),constants),
               _rho(_mul((q,q),roots['n']),constants))
        return max(Q(0),v[0]/A),max(Q(0),v[1]/A)
    if metric(Q(1))[1]<=radius:
        return Q(1),Q(1)
    lo,hi=Q(0),Q(1)
    for _ in range(cfg['max_cap_bisections']):
        mid=(lo+hi)/2
        lower,upper=metric(mid)
        if upper<=radius:
            lo=mid
        elif lower>radius:
            hi=mid
        else:
            break
    return lo,hi

def _configuration(limits):
    defaults=dict(max_endpoint_evaluations=8,max_panels=2,max_refinements=0,
                  initial_partition=[Q(0),Q(1,2),Q(1)],max_root_bisections=128,
                  max_cap_bisections=128,precision_bits=128)
    if type(limits) is not dict or set(limits)-set(defaults):
        raise ValueError('Unknown prediction limit')
    cfg=dict(defaults,**limits)
    for key in defaults:
        if key!='initial_partition' and (type(cfg[key]) is not int or cfg[key]<0):
            raise ValueError('Strict nonnegative integer limit required')
    if cfg['max_refinements']!=0:
        raise ValueError('This source implements frozen partition only')
    if not 128<=cfg['precision_bits']<=512 or cfg['max_root_bisections']>512 or cfg['max_cap_bisections']>512:
        raise ValueError('Precision/bisection outside source contract')
    grid=[_q(x) for x in cfg['initial_partition']]
    if len(grid)<2 or grid[0]!=0 or grid[-1]!=1 or any(a>=b for a,b in zip(grid,grid[1:])):
        raise ValueError('Strict complete [0,1] partition required')
    cfg['initial_partition']=grid
    return cfg

def _specification(spec):
    allowed={'schema_version','ks','events','row_count','development_control','source_preparation_only',
             'selected_utc','claim','definition','observation_quality','numeric_observations','numeric_predictions','hard_utc'}
    if type(spec) is not dict or FORBIDDEN.intersection(spec) or set(spec)-allowed:
        raise ValueError('Observation input forbidden')
    if type(spec.get('schema_version')) is not int or spec['schema_version']!=1:
        raise ValueError('Unsupported schema')
    ks,events=spec.get('ks'),spec.get('events')
    if type(ks) not in (list,tuple) or not ks or any(type(k) is not int or k not in KS for k in ks) or len(set(ks))!=len(ks):
        raise ValueError('Distinct declared laws required')
    if type(events) is not list or not events:
        raise ValueError('Declared events required')
    clean=[]
    for e in events:
        if type(e) is not dict or set(e)!={'id','lifetime_upper','separation_upper','birth_upper'} or type(e['id']) is not str or not e['id']:
            raise ValueError('Invalid event descriptor')
        tau=None if e['lifetime_upper'] is None else _q(e['lifetime_upper'])
        radius=None if e['separation_upper'] is None else _q(e['separation_upper'])
        birth=None if e['birth_upper'] is None else _q(e['birth_upper'])
        if (tau is not None and tau<0) or (radius is not None and radius<0) or birth not in (None,Q(0)):
            raise ValueError('Unsupported threshold')
        clean.append(dict(id=e['id'],tau=tau,radius=radius,birth=birth))
    if len({e['id'] for e in clean})!=len(clean) or type(spec.get('row_count')) is not int or spec['row_count']!=len(ks)*len(clean):
        raise ValueError('Exact unique complete row table required')
    return list(ks),clean

def predict(spec,limits):
    """Pure fixed-partition outward bounds; returned intervals are not admission."""
    ks,events=_specification(spec)
    cfg=_configuration(limits)
    # Invocation-local analytic caches. A cache hit still consumes one endpoint
    # REQUEST; these counters expose only the work avoided by these helpers.
    roots_cache,life_cache,physical_cache={},{},{}
    cache_work=dict(root_evaluations=0,root_cache_hits=0,
                    lifetime_cap_evaluations=0,lifetime_cap_cache_hits=0,
                    lifetime_scalar_cap_calls=0,physical_cap_evaluations=0,
                    physical_cap_cache_hits=0,constants_evaluations=0)
    if cfg['max_endpoint_evaluations']==0:
        rows=[dict(k=k,event_id=e['id'],probability=(Q(0),Q(1)),
                   status='INCONCLUSIVE_BUDGET',endpoint_witnesses={},panels=[],
                   uncertainty=dict(uncomputed_full_mass=Q(1)))
              for e in events for k in ks]
        return dict(schema_version=1,record_class='UNQUALIFIED_ANALYTIC_INTERVAL_OUTPUT',
                    status='INCONCLUSIVE_BUDGET',analytic_spec_sha256=ANALYTIC_SHA256,
                    row_count=len(rows),rows=rows,numeric_prediction_qualified=False,
                    endpoint_evaluations=0,cache_work=cache_work,budget_reason='ZERO_ENDPOINT_BUDGET',
                    qualification_requires='External source/runtime/freeze/terminal and nonauthor artifact review')
    constants=_constants(cfg['precision_bits']) if any(e['radius'] is not None for e in events) and cfg['max_endpoint_evaluations'] else None
    cache_work['constants_evaluations']=int(constants is not None)
    half=_sqrt(Q(1,2),cfg['precision_bits'])
    zlo=_root(1-half[1],8,cfg['max_root_bisections'])[0]
    zhi=_root(1-half[0],8,cfg['max_root_bisections'])[1]
    used,rows=0,[]
    for event in events:
        top=zlo if event['birth'] is not None else Q(1)
        grid=[z for z in cfg['initial_partition'] if z<top]+[top]
        witnesses={}
        over=len(grid)-1>cfg['max_panels']
        for z in grid:
            if used>=cfg['max_endpoint_evaluations'] or over:
                witnesses[z]=dict(integrand=(Q(0),Q(1)),status='UNCOMPUTED_BUDGET')
                continue
            used+=1
            if z in roots_cache:
                root=roots_cache[z]
                cache_work['root_cache_hits']+=1
            else:
                root=_roots(z,cfg)
                roots_cache[z]=root
                cache_work['root_evaluations']+=1
            life_key=z,event['tau']
            if life_key in life_cache:
                life=life_cache[life_key]
                cache_work['lifetime_cap_cache_hits']+=1
            else:
                if event['tau'] is None:
                    life=(Q(1),Q(1))
                else:
                    ll,lu=root['lifetime']
                    life=(_life_cap(lu,event['tau'],cfg['max_cap_bisections'])[0],
                          _life_cap(ll,event['tau'],cfg['max_cap_bisections'])[1])
                    cache_work['lifetime_scalar_cap_calls']+=2
                life_cache[life_key]=life
                cache_work['lifetime_cap_evaluations']+=1
            physical_key=z,event['radius']
            if physical_key in physical_cache:
                physical=physical_cache[physical_key]
                cache_work['physical_cap_cache_hits']+=1
            else:
                physical=_metric_cap(z,root,event['radius'],cfg,constants)
                physical_cache[physical_key]=physical
                cache_work['physical_cap_evaluations']+=1
            cap=min(life[0],physical[0]),min(life[1],physical[1])
            witnesses[z]=dict(root,lifetime_cap=life,physical_cap=physical,combined_cap=cap,
                              integrand=_pow(cap,5),status='OUTWARD_ENCLOSURE')
        lows=[max(witnesses[t]['integrand'][0] for t in grid[j:]) for j in range(len(grid))]
        highs=[min(witnesses[t]['integrand'][1] for t in grid[:j+1]) for j in range(len(grid))]
        if any(a>b for a,b in zip(lows,highs)):
            raise ValueError('Inconsistent monotone endpoint certificate')
        for k in ks:
            panels,lower,upper=[],Q(0),Q(0)
            for j,(left,right) in enumerate(zip(grid,grid[1:])):
                w=right**k-left**k
                lower+=w*lows[j+1]
                upper+=w*highs[j]
                panels.append(dict(Z_interval=(left,right),weight=w,lower_value=lows[j+1],upper_value=highs[j]))
            strip=zhi**k-zlo**k if event['birth'] is not None else Q(0)
            budget=over or any(w['status']=='UNCOMPUTED_BUDGET' for w in witnesses.values())
            row=dict(k=k,event_id=event['id'],probability=_clip((lower,upper+strip)),
                     status='INCONCLUSIVE_BUDGET' if budget else 'OUTWARD_INTERVAL_ONLY',
                     endpoint_witnesses={str(z):w for z,w in witnesses.items()},panels=panels,
                     uncertainty=dict(cutoff_strip_mass=strip,uncomputed_full_mass=Q(1) if budget else Q(0)))
            if event['birth'] is not None:
                row['birth_cutoff']=dict(interval=(zlo,zhi),admitted_region='LOWER_Z',
                                        unknown_strip_mass=strip,conditional_renormalization=False)
            rows.append(row)
    return dict(schema_version=1,record_class='UNQUALIFIED_ANALYTIC_INTERVAL_OUTPUT',
                status='INCONCLUSIVE_BUDGET' if any(r['status']=='INCONCLUSIVE_BUDGET' for r in rows) else 'OUTWARD_INTERVAL_ONLY',
                analytic_spec_sha256=ANALYTIC_SHA256,row_count=len(rows),rows=rows,
                numeric_prediction_qualified=False,endpoint_evaluations=used,cache_work=cache_work,
                qualification_requires='External source/runtime/freeze/terminal and nonauthor artifact review')

def validate_prediction(record,spec):
    """Strict conservative DEVELOPMENT_STRUCTURE_FIXTURE validator only.
    Tested fixture acceptance is never numerical/production artifact validation.
    """
    try:
        ks,events=_specification(spec)
        if type(record) is not dict:
            return False
        artifact,freeze=record['artifact'],record['freeze']
        if type(artifact) is not dict or type(freeze) is not dict:
            return False
        if type(artifact.get('schema_version')) is not int or artifact['schema_version']!=1 or artifact['record_class']!='DEVELOPMENT_STRUCTURE_FIXTURE' or artifact['status']!='INCONCLUSIVE_BUDGET' or artifact['numeric_prediction_qualified'] is not False or artifact['analytic_spec_sha256']!=ANALYTIC_SHA256:
            return False
        if freeze['completed'] is not True or type(freeze['sequence']) is not int or freeze['sequence']<1:
            return False
        raw=(json.dumps(artifact,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')
        if hashlib.sha256(raw).hexdigest()!=freeze['artifact_sha256']:
            return False
        first=record['first_observation_sequence']
        if first is not None and (type(first) is not int or first<=freeze['sequence']):
            return False
        before,after=record['source_before'],record['source_after']
        if type(before) is not dict or not before or before!=after:
            return False
        for v in before.values():
            if type(v) is not dict or set(v)!={'bytes','sha256'} or type(v['bytes']) is not int or v['bytes']<0 or type(v['sha256']) is not str or len(v['sha256'])!=64 or any(c not in '0123456789abcdef' for c in v['sha256']):
                return False
        terminal=record['terminal']
        if terminal.get('observed') is not True or type(terminal.get('returncode')) is not int or terminal['returncode']!=0 or terminal.get('fixture_only') is not True:
            return False
        expected={(k,e['id']) for k in ks for e in events}
        rows=artifact['rows']
        if type(rows) is not list or len(rows)!=len(expected) or type(artifact['row_count']) is not int or artifact['row_count']!=len(expected):
            return False
        seen=set()
        for row in rows:
            key=row['k'],row['event_id']
            if type(row['k']) is not int or key not in expected or key in seen:
                return False
            seen.add(key)
            if tuple(_q(x) for x in row['probability'])!=(Q(0),Q(1)) or row['status']!='INCONCLUSIVE_BUDGET' or _q(row['uncertainty']['uncomputed_full_mass'])!=1:
                return False
        return seen==expected
    except (KeyError,TypeError,ValueError,ZeroDivisionError,AttributeError):
        return False


def _strict_hash(value):
    return (type(value) is str and len(value)==64 and
            all(character in '0123456789abcdef' for character in value))


def _strict_source(value):
    if type(value) is not dict or not value:
        return False
    for name,entry in value.items():
        if (type(name) is not str or not name or type(entry) is not dict or
                set(entry)!={'bytes','sha256'} or type(entry['bytes']) is not int or
                entry['bytes']<0 or not _strict_hash(entry['sha256'])):
            return False
    return True


def _numeric_shape(value,spec,limits):
    """Cheap class/table/type gate before any fresh analytic computation."""
    ks,events=_specification(spec)
    cfg=_configuration(limits)
    if type(value) is not dict:
        return False
    if (type(value.get('schema_version')) is not int or value['schema_version']!=1 or
            value.get('record_class')!='UNQUALIFIED_ANALYTIC_INTERVAL_OUTPUT' or
            value.get('numeric_prediction_qualified') is not False or
            value.get('analytic_spec_sha256')!=ANALYTIC_SHA256 or
            value.get('status') not in ('INCONCLUSIVE_BUDGET','OUTWARD_INTERVAL_ONLY') or
            type(value.get('endpoint_evaluations')) is not int or
            not 0<=value['endpoint_evaluations']<=cfg['max_endpoint_evaluations']):
        return False
    expected={(k,event['id']) for k in ks for event in events}
    rows=value.get('rows')
    if (type(value.get('row_count')) is not int or value['row_count']!=len(expected) or
            type(rows) is not list or len(rows)!=len(expected)):
        return False
    seen=set()
    for row in rows:
        if (type(row) is not dict or type(row.get('k')) is not int or
                type(row.get('event_id')) is not str):
            return False
        key=row['k'],row['event_id']
        if key not in expected or key in seen:
            return False
        seen.add(key)
    return seen==expected


def verify_numeric_artifact(artifact,spec,limits):
    """Exact canonical replay of the pure predictor, not independent numerics.

    The class/flag/table gate rejects fixtures and malformed shape first. Full
    canonical equality then binds every witness, row, uncertainty and counter
    to a fresh predict invocation with exactly the supplied analytic inputs.
    """
    try:
        if not _numeric_shape(artifact,spec,limits):
            return False
        # Reject unsupported input types/domain before doing fresh arithmetic.
        exact_artifact.dumps(spec)
        exact_artifact.dumps(limits)
        submitted=exact_artifact.dumps(artifact)
        return submitted==exact_artifact.dumps(predict(spec,limits))
    except (KeyError,TypeError,ValueError,ZeroDivisionError,AttributeError,
            OverflowError,RecursionError):
        return False


def validate_numeric_prediction(record,prediction_spec,prediction_limits,
                                expected_source,first_observation_sequence=None):
    """Bind real analytic replay to externally supplied source/freeze receipts.

    Source identity comes from expected_source supplied by the frozen worker
    contract. Receipt booleans/hash bindings are checked evidence fields, not
    cryptographic proof of an independent review or actual runtime observation.
    The runner must supply the actual first FIELD_ATTEMPT_START sequence before
    its first certifier call; None permits preflight before that record exists.
    """
    try:
        if type(record) is not dict or not _strict_source(expected_source):
            return False
        before,after=record['source_before'],record['source_after']
        if not _strict_source(before) or not _strict_source(after):
            return False
        expected_raw=exact_artifact.dumps(expected_source)
        if exact_artifact.dumps(before)!=expected_raw or exact_artifact.dumps(after)!=expected_raw:
            return False
        if type(prediction_spec) is not dict or type(prediction_limits) is not dict:
            return False
        development=prediction_spec.get('development_control') is True
        if ('development_control' in prediction_spec and not development) or type(record['development_receipts_only']) is not bool:
            return False
        if record['development_receipts_only'] is not development:
            return False
        # Canonical equality preserves strict int/bool and list/tuple distinctions.
        limits_raw=exact_artifact.dumps(prediction_limits)
        if type(record['limits']) is not dict or exact_artifact.dumps(record['limits'])!=limits_raw:
            return False
        freeze,terminal,review=record['freeze'],record['terminal'],record['review']
        if type(freeze) is not dict or type(terminal) is not dict or type(review) is not dict:
            return False
        sequence=freeze['sequence']
        if (freeze['completed'] is not True or type(sequence) is not int or sequence<1 or
                type(freeze['artifact_bytes']) is not int or freeze['artifact_bytes']<0 or
                not all(_strict_hash(freeze[name]) for name in
                        ('artifact_sha256','spec_sha256','limits_sha256'))):
            return False
        recorded_first=record['first_observation_sequence']
        for first in (recorded_first,first_observation_sequence):
            if first is not None and (type(first) is not int or first<=sequence):
                return False
        if (recorded_first is not None and first_observation_sequence is not None and
                recorded_first!=first_observation_sequence):
            return False
        if (terminal.get('observed') is not True or terminal.get('waited') is not True or
                type(terminal.get('owned_child_pid')) is not int or terminal['owned_child_pid']<1 or
                type(terminal.get('returncode')) is not int or terminal['returncode']!=0):
            return False
        author=record['author_id']
        reviewer=review.get('reviewer_id')
        if (type(author) is not str or not author or type(reviewer) is not str or not reviewer or
                reviewer==author or review.get('author_id')!=author or
                review.get('completed') is not True or review.get('nonauthor') is not True or
                review.get('verdict')!='ACCEPT_NUMERIC_ARTIFACT_SOURCE_EXPOSED' or
                not _strict_hash(review.get('artifact_sha256'))):
            return False
        # Validate all receipt values as canonical exact types/scalar text, too.
        exact_artifact.dumps(record)
        value=record['artifact']
        if not _numeric_shape(value,prediction_spec,prediction_limits):
            return False
        raw=exact_artifact.dumps(value)
        artifact_hash=hashlib.sha256(raw).hexdigest()
        if (freeze['artifact_bytes']!=len(raw) or freeze['artifact_sha256']!=artifact_hash or
                review['artifact_sha256']!=artifact_hash or
                freeze['spec_sha256']!=hashlib.sha256(exact_artifact.dumps(prediction_spec)).hexdigest() or
                freeze['limits_sha256']!=hashlib.sha256(limits_raw).hexdigest()):
            return False
        return verify_numeric_artifact(value,prediction_spec,prediction_limits)
    except (KeyError,TypeError,ValueError,ZeroDivisionError,AttributeError,
            OverflowError,RecursionError):
        return False
