"""Exact field accounting and shared production/control engine after E02U RED.

Source-only implementation; Root separately admits frozen target invocations.
"""
from copy import deepcopy
from fractions import Fraction as Q
import field
import artifact
import hashlib
import json
import prediction
import field_certificate as fc

KS=(8,12,15,16)
_ACCEPT={'PASS','CERTIFIED','RESOLVED','FINITE_WINDOW_CONSISTENT',
         'FINITE_WINDOW_CONSISTENT_ZERO','FINITE_WINDOW_FALSIFIED_CANDIDATE'}

def _need(ok,message):
    if not ok:
        raise ValueError(message)

def _q(value):
    _need(type(value) in (Q,int,str),'Exact rational required')
    try:
        return Q(value)
    except (ValueError,ZeroDivisionError):
        raise ValueError('Invalid exact rational')

def _interval(value,probability=False):
    _need(type(value) in (tuple,list) and len(value)==2,'Interval required')
    lo,hi=(_q(v) for v in value)
    _need(lo<=hi and (not probability or 0<=lo<=hi<=1),'Invalid interval')
    return lo,hi

def _cell(value):
    _need(type(value) is dict,'Cell dictionary required')
    c={}
    for key in ('Qa','Qb','Za','Zb'):
        v=value.get(key)
        _need(type(v) is Q and 0<=v<=1,'Exact cell coordinate required')
        _need(v.denominator & (v.denominator-1)==0,'Dyadic cell required')
        c[key]=v
    _need(c['Qa']<c['Qb'] and c['Za']<c['Zb'],'Positive cell area required')
    sign=value.get('epsilon')
    _need(type(sign) is int and sign in (-1,1),'Strict sign required')
    c['epsilon']=sign
    return c

def _ids(events):
    _need(type(events) in (tuple,list) and events,'Event list required')
    out=[]
    for event in events:
        _need(type(event) is dict and type(event.get('id')) is str and event['id'],'Event ID required')
        _need(event['id'] not in out,'Duplicate event ID')
        out.append(event['id'])
    return out

def _bound(row,name):
    if row.get('error') or row.get('status') not in (None,'RESOLVED'):
        return 0,1
    pair=row.get('event_bounds',{}).get(name,(0,1))
    _need(type(pair) in (tuple,list) and len(pair)==2,'Cell event bound required')
    lo,hi=pair
    _need(type(lo) is int and type(hi) is int and (lo,hi) in ((0,0),(0,1),(1,1)),
          'Certified binary event bound required')
    return lo,hi

def classify_event(certificate,event):
    _need(type(certificate) is dict and type(event) is dict,'Certificate and event required')
    if certificate.get('status')!='RESOLVED':
        return 0,1
    lo,hi=_interval(certificate['lifetime_scaled'])
    t=event.get('lifetime_upper')
    decisions=[]
    if t is None:
        decisions.append((1,1) if lo>0 else (0,1))
    else:
        t=_q(t)
        _need(t>0,'Positive lifetime threshold required')
        decisions.append((0,0) if hi<=0 or lo>t else (1,1) if lo>0 and hi<=t else (0,1))
    for threshold,key in (('separation_upper','separation_over_A'),('birth_upper','centered_birth_scaled')):
        if event.get(threshold) is not None:
            a,b=_interval(certificate[key])
            cut=_q(event[threshold])
            decisions.append((1,1) if b<=cut else (0,0) if a>cut else (0,1))
    if (0,0) in decisions:
        return 0,0
    return (1,1) if all(d==(1,1) for d in decisions) else (0,1)

def validate_partition(cells):
    try:
        _need(type(cells) in (tuple,list) and 0<len(cells)<=4096,'Bounded leaves required')
        parsed=[_cell(c) for c in cells]
        for sign in (-1,1):
            selected=[c for c in parsed if c['epsilon']==sign]
            _need(selected,'Both signs required')
            changes={}
            for i,c in enumerate(selected):
                changes.setdefault(c['Qa'],[[],[]])[1].append((i,c['Za'],c['Zb']))
                changes.setdefault(c['Qb'],[[],[]])[0].append(i)
            points=sorted(set(changes)|{Q(0),Q(1)})
            active={}
            for i,x in enumerate(points):
                remove,add=changes.get(x,([],[]))
                for index in remove:
                    _need(index in active,'Invalid partition removal')
                    del active[index]
                for index,za,zb in add:
                    _need(index not in active,'Duplicate partition leaf')
                    active[index]=(za,zb)
                if i+1<len(points):
                    cursor=Q(0)
                    for za,zb in sorted(active.values()):
                        _need(za==cursor,'Partition gap or overlap')
                        cursor=zb
                    _need(cursor==1,'Incomplete partition slab')
            _need(not active,'Unclosed partition')
        return True
    except (ValueError,TypeError,KeyError):
        return False

def aggregate_cells(rows,events,ks=KS):
    _need(type(rows) in (tuple,list) and rows,'All planned leaves required')
    ids=_ids(events)
    _need(type(ks) in (tuple,list) and ks and len(set(ks))==len(ks),'Distinct laws required')
    _need(all(type(k) is int and k in KS for k in ks),'Declared laws required')
    _need(validate_partition([r['cell'] for r in rows]),'Incomplete or overlapping partition')
    bounds={k:{name:[Q(0),Q(0)] for name in ids} for k in ks}
    unknown={k:{name:Q(0) for name in ids} for k in ks}
    total={k:Q(0) for k in ks}
    for row in rows:
        c=_cell(row['cell'])
        for k in ks:
            mass=field.cell_mass(c,k)
            _need(type(mass) is Q and mass>=0,'Exact cell mass required')
            total[k]+=mass
            for name in ids:
                lo,hi=_bound(row,name)
                bounds[k][name][0]+=mass*lo
                bounds[k][name][1]+=mass*hi
                unknown[k][name]+=mass*(hi-lo)
    _need(all(m==1 for m in total.values()),'Whole-law mass must equal one')
    return {'law_bounds':{k:{name:tuple(v) for name,v in table.items()} for k,table in bounds.items()},
            'total_mass':total,'unresolved_mass':unknown,
            'resolved_mass':{k:{name:1-v for name,v in table.items()} for k,table in unknown.items()},
            'partition_verified':True}

def _count(value,name):
    _need(type(value) is int and value>=0,'Nonnegative strict counter required: '+name)
    return value

def _refinement_axis(row,cell):
    if 'refinement_terms' in row:
        terms=row['refinement_terms']
        _need(type(terms) is dict and 'Q' in terms and 'Z' in terms,
              'Both source-bound refinement terms required')
        qt,zt=_q(terms['Q']),_q(terms['Z'])
        _need(qt>=0 and zt>=0,'Nonnegative source-bound refinement terms required')
    else:
        # Root-unresolved certificates have no coupling terms. Width decreases
        # after a split, so this fallback cannot keep starving the other axis.
        qt,zt=cell['Qb']-cell['Qa'],cell['Zb']-cell['Za']
    return 'Q' if qt>=zt else 'Z'

def _dyadic_depth(value):
    return value.denominator.bit_length()-1

def choose_refinement(rows,events):
    _need(type(rows) is dict,'Refinement state envelope required')
    leaves,costs,limits=rows.get('leaves'),rows.get('costs'),rows.get('limits')
    _need(type(leaves) is list and type(costs) is dict and type(limits) is dict,
          'Leaves, cumulative costs and limits required')
    ids=_ids(events)
    _need(validate_partition([r['cell'] for r in leaves]),'Full partition required')
    attempted=_count(costs.get('attempted_fields',0),'attempted_fields')
    failed=_count(costs.get('failed_fields',0),'failed_fields')
    vertices=_count(costs.get('sampled_vertices',0),'sampled_vertices')
    _need(failed<=attempted,'Failed attempts cannot exceed all attempts')
    caps=(_count(limits.get('field_evaluations',4096),'field_evaluations'),
          _count(limits.get('control_cells',4096),'control_cells'),
          _count(limits.get('total_vertices',16777216),'total_vertices'))
    _need(caps[0]<=4096 and caps[1]<=4096 and caps[2]<=16777216,'Caps may only reduce')
    depth_cap=_count(limits.get('max_dyadic_depth',32),'max_dyadic_depth')
    _need(depth_cap<=32,'Dyadic depth cap may only reduce')
    created=_count(costs.get('created_cells',len(leaves)),'created_cells')
    if attempted>=caps[0] or created+2>caps[1] or vertices+64>caps[2]:
        return {'status':'INCONCLUSIVE_BUDGET','selected':None,'evaluations_scheduled':0,
                'leaves':deepcopy(leaves),'costs':deepcopy(costs)}
    if any(_dyadic_depth(row['cell'][key])>depth_cap
           for row in leaves for key in ('Qa','Qb','Za','Zb')):
        return {'status':'INCONCLUSIVE_BUDGET','selected':None,'evaluations_scheduled':0,
                'leaves':deepcopy(leaves),'costs':deepcopy(costs)}
    candidates=[]
    names=set()
    for row in leaves:
        name=row.get('cell_id')
        _need(type(name) is str and name and name not in names,'Distinct stable cell IDs required')
        names.add(name)
        if row.get('status')=='IMPLEMENTATION_FAILURE':
            return {'status':'IMPLEMENTATION_FAILURE','selected':None,'evaluations_scheduled':0,
                    'leaves':deepcopy(leaves),'costs':deepcopy(costs)}
        if any(_bound(row,name_)==(0,1) for name_ in ids):
            c=_cell(row['cell'])
            candidates.append((max(field.cell_mass(c,k) for k in KS),name,row))
    if not candidates:
        return {'status':'COMPLETE','selected':None,'evaluations_scheduled':0,
                'leaves':deepcopy(leaves),'costs':deepcopy(costs)}
    mass,name,row=min(candidates,key=lambda item:(-item[0],item[1]))
    c=_cell(row['cell'])
    axis=_refinement_axis(row,c)
    lo,hi=axis+'a',axis+'b'
    mid=(c[lo]+c[hi])/2
    if _dyadic_depth(mid)>depth_cap:
        return {'status':'INCONCLUSIVE_BUDGET','selected':None,'evaluations_scheduled':0,
                'leaves':deepcopy(leaves),'costs':deepcopy(costs)}
    left,right=dict(c),dict(c)
    left[hi],right[lo]=mid,mid
    return {'status':'REFINE','cell_id':name,'selected':name,'priority_mass':mass,
            'axis':axis,'children':[left,right],'child_ids':[name+'/0',name+'/1'],
            'evaluations_scheduled':0}

def compare_row(observed,predicted,qualified):
    observed,predicted=_interval(observed,True),_interval(predicted,True)
    if qualified is not True:
        return 'INCONCLUSIVE_COUPLING'
    if observed[1]<predicted[0] or observed[0]>predicted[1]:
        return 'FINITE_WINDOW_FALSIFIED_CANDIDATE'
    if predicted==(Q(0),Q(0)):
        return 'FINITE_WINDOW_CONSISTENT_ZERO' if observed==(Q(0),Q(0)) else 'INCONCLUSIVE_PRECISION'
    if predicted[0]>0 and observed[1]-observed[0]<=min(Q(1,64),predicted[0]/4):
        return 'FINITE_WINDOW_CONSISTENT'
    return 'INCONCLUSIVE_PRECISION'

def withdraw_acceptance(result,reason):
    _need(type(result) is dict and type(reason) is str and reason,'Result and withdrawal reason required')
    out=deepcopy(result)
    def visit(value):
        if type(value) is dict:
            for child in list(value.values()):
                visit(child)
            if value.get('status') in _ACCEPT:
                value.setdefault('computed_status_before_withdrawal',value['status'])
                value['status']=reason
            for key in ('qualified','full_declared_campaign','numeric_prediction_qualified',
                        'production_qualification','all_comparisons_consistent'):
                if key in value:
                    value[key]=False
        elif type(value) in (list,tuple):
            for child in value:
                visit(child)
    visit(out)
    if out.get('status')!=reason:
        out.setdefault('computed_status_before_withdrawal',out.get('status'))
    out['status']=reason
    out['full_declared_campaign']=False
    return out

def finalize_attempt(result,before,after,terminal,budget):
    _need(type(result) is dict and type(before) is dict and type(after) is dict,'Retained observations required')
    out=deepcopy(result)
    out['source_before'],out['source_after']=deepcopy(before),deepcopy(after)
    out['terminal'],out['budget']=deepcopy(terminal),deepcopy(budget)
    names=set(before)|set(after)
    drift=sorted(n for n in names if before.get(n)!=after.get(n)
                 or type(before.get(n)) is not dict or type(after.get(n)) is not dict
                 or 'error' in before.get(n,{}) or 'error' in after.get(n,{}))
    out['source_drift']=drift
    if not names or drift:
        return withdraw_acceptance(out,'INCONCLUSIVE_SOURCE_DRIFT')
    known=(type(terminal) is dict and terminal.get('observed') is True
           and terminal.get('waited') is True and type(terminal.get('owned_child_pid')) is int
           and terminal['owned_child_pid']>0 and type(terminal.get('returncode')) is int)
    if not known:
        return withdraw_acceptance(out,'TERMINAL_UNOBSERVED')
    if terminal['returncode']!=0:
        return withdraw_acceptance(out,'CHILD_FAILED')
    _need(type(budget) is dict,'Budget observation required')
    exhausted=budget.get('exhausted') is True
    if 'elapsed_seconds' in budget and 'wall_limit_seconds' in budget:
        elapsed=_q(str(budget['elapsed_seconds']))
        limit=_q(str(budget['wall_limit_seconds']))
        _need(elapsed>=0 and limit>0,'Valid elapsed and wall-limit observations required')
        exhausted=exhausted or elapsed>=limit
    if exhausted:
        return withdraw_acceptance(out,'INCONCLUSIVE_BUDGET')
    gates=('full_partition_verified','prediction_freeze_verified',
           'complete_source_closure_verified','all_coupling_gates_verified')
    if not all(out.get(g) is True for g in gates):
        return withdraw_acceptance(out,'INCONCLUSIVE_COUPLING')
    return out

def run(spec,predictions,limits,checkpoint=None,should_stop=None):
    _need(type(spec) is dict and type(limits) is dict,'Run specification and limits required')
    mode=spec.get('mode',spec.get('phase','PRODUCTION'))
    if mode=='DEVELOPMENT_OBSERVATION_ONLY':
        contract=spec.get('development_contract')
        _need(type(contract) is dict and contract.get('record_class')=='BOUNDED_DEVELOPMENT_OBSERVATION'
              and contract.get('production_qualification') is False,
              'Explicit bounded development observation contract required')
        _need(predictions is None,'Development observation cannot masquerade as production prediction')
        cap=_count(limits.get('field_evaluations'),'field_evaluations')
        _need(0<cap<=32,'Tiny development cap must be at most32')
        _need(spec.get('refinement') is False,'Development control has a fixed partition')
        cells=spec.get('initial_cells')
        _need(validate_partition(cells),'Complete development partition required')
        cell_cap=_count(limits.get('control_cells'),'control_cells')
        vertex_cap=_count(limits.get('total_vertices'),'total_vertices')
        root_cap=_count(limits.get('root_bisections'),'root_bisections')
        _need(0<cell_cap<=32 and len(cells)<=cell_cap,'At most32 development cells')
        _need(type(limits.get('tensor_side')) is int and limits['tensor_side']==8,
              'Development geometry requires exactly64 nodes')
        _need(64<=vertex_cap<=32*64,'Development node cap may only reduce')
        _need(2<=root_cap<=512,'Development per-root bisection cap may only reduce')
        _need(limits.get('error_mode')=='tight','Frozen tight development coupling required')
        ks=spec.get('ks')
        _need(type(ks) in (list,tuple) and tuple(ks)==KS,'Four declared development laws required')
        events=spec.get('events');ids=_ids(events)
        # Validate predicates before any field attempt, without calculating predictions.
        for event in events:
            t=event.get('lifetime_upper')
            _need(t is None or _q(t)>0,'Positive lifetime threshold required')
            for key in ('separation_upper','birth_upper'):
                if event.get(key) is not None:_q(event[key])
        rows=[{'cell_id':'initial/'+str(i),'cell':_cell(c),'certificate':None,
               'status':'NOT_ATTEMPTED_BUDGET','event_bounds':{name:(0,1) for name in ids}}
              for i,c in enumerate(cells)]
        costs={'attempted_fields':0,'sampled_vertices':0,'root_bisections':0,
               'failed_fields':0,'created_cells':len(rows)}
        status='DEVELOPMENT_OBSERVATIONS_RETAINED'
        exact_costs=True;unknown_cost_fields=[]
        for row in rows:
            # Reserve one actual64-node geometry before calling the real certifier.
            # Certification replays do not create additional sampled field geometries.
            if costs['attempted_fields']>=cap or costs['sampled_vertices']+64>vertex_cap:
                status='INCONCLUSIVE_BUDGET';break
            one_limits={'field_evaluations':1,'control_cells':1,'tensor_side':8,
                        'total_vertices':64,'root_bisections':root_cap,'error_mode':'tight'}
            costs['attempted_fields']+=1
            try:
                certificate=fc.certify_cell(row['cell'],one_limits)
                row['certificate']=certificate
                _need(type(certificate) is dict,'Actual field certificate required')
                actual=certificate.get('costs')
                _need(type(actual) is dict,'Actual certificate costs required')
                _need(_count(actual.get('attempted_fields'),'certificate attempted_fields')==1,
                      'Every certificate must retain its attempted field')
                nv=_count(actual.get('sampled_vertices'),'certificate sampled_vertices')
                nr=_count(actual.get('root_bisections'),'certificate root_bisections')
                _need(nv in (0,64),'Only actual64-node geometry is counted')
                _need(nr<=3*root_cap,'Actual root costs exceed three admitted root caps')
            except Exception as error:
                # A raised/invalid certificate cannot erase this cell or later mass.
                # Partial geometry/root work is unknown, never invented as zero.
                row['status']='IMPLEMENTATION_FAILURE'
                row['error']={'type':type(error).__name__,'message':str(error)}
                costs['failed_fields']+=1;exact_costs=False
                unknown_cost_fields.append(row['cell_id'])
                status='IMPLEMENTATION_FAILURE';break
            costs['sampled_vertices']+=nv;costs['root_bisections']+=nr
            row['status']=certificate.get('status','IMPLEMENTATION_FAILURE')
            if row['status']=='IMPLEMENTATION_FAILURE':
                costs['failed_fields']+=1;status='IMPLEMENTATION_FAILURE';break
            if row['status']=='RESOLVED':
                try:
                    row['event_bounds']={event['id']:classify_event(certificate,event) for event in events}
                except (ValueError,TypeError,KeyError) as error:
                    row['status']='IMPLEMENTATION_FAILURE'
                    row['error']={'type':type(error).__name__,'message':str(error)}
                    costs['failed_fields']+=1;status='IMPLEMENTATION_FAILURE';break
        known_costs=deepcopy(costs)
        if not exact_costs:
            costs['sampled_vertices']=None;costs['root_bisections']=None
        return {'mode':mode,'status':status,'rows':rows,
                'aggregate':aggregate_cells(rows,events,ks),'costs':costs,
                'costs_exact':exact_costs,'known_costs_before_unaccounted_attempt':known_costs,
                'unknown_cost_fields':unknown_cost_fields,
                'limits':deepcopy(limits),'root_cap_scope':'PER_ROOT_NOT_CUMULATIVE',
                'root_call_upper_bound':3*costs['attempted_fields'],
                'numeric_prediction_qualified':False,'production_qualification':False,
                'full_declared_campaign':False}
    # Preserve historical invalid-input rejection precedence. These basic
    # packet guards do not admit any mode or weaken the strict engine below.
    _need(type(predictions) is dict,'Completed prediction freeze required before observation')
    numeric=predictions.get('artifact')
    _need(type(numeric) is dict,'Production prediction artifact required')
    kind=numeric.get('record_class','')
    _need(type(kind) is str and not kind.startswith('DEVELOPMENT'),
          'Development fixture cannot supply a production prediction')
    freeze=predictions.get('freeze')
    _need(type(freeze) is dict and freeze.get('completed') is True,'Completed prediction freeze required')
    sequence=freeze.get('sequence')
    _need(type(sequence) is int and sequence>=1,'Prediction freeze sequence required')
    first=predictions.get('first_observation_sequence')
    _need(first is None or (type(first) is int and sequence<first),
          'Prediction freeze must precede first observation')
    return _run_production(spec,predictions,limits,checkpoint,should_stop)


# The strict production and tiny control modes share this one observation engine.
# Only the supervised writer establishes durable custody; these reference checks
# bind returned metadata to submitted complete bytes, not to an external authority.
_CAMPAIGN_SHA256='be65c90e312a264d48e58ece142c6d2d93b1c73e564e08ada02ad656f9fefad3'


def _bounded_text(value,maximum,name,nonempty=True):
    _need(type(value) is str and (value or not nonempty),name+' must be text')
    try:
        _need(len(value.encode('utf8'))<=maximum,name+' exceeds frozen byte bound')
    except UnicodeEncodeError:
        raise ValueError(name+' must contain Unicode scalar text')
    return value


def _production_events(events):
    ids=_ids(events)
    clean=[]
    for event in events:
        _need(set(event)=={'id','lifetime_upper','separation_upper','birth_upper'},
              'Exact event descriptor required')
        _bounded_text(event['id'],128,'Event ID')
        parsed={'id':event['id']}
        for key in ('lifetime_upper','separation_upper','birth_upper'):
            value=event[key]
            parsed[key]=None if value is None else _q(value)
        _need(parsed['lifetime_upper'] is None or parsed['lifetime_upper']>0,
              'Positive lifetime threshold required')
        _need(parsed['separation_upper'] is None or parsed['separation_upper']>=0,
              'Nonnegative separation threshold required')
        _need(parsed['birth_upper'] in (None,Q(0)),'Frozen birth threshold required')
        clean.append(parsed)
    return ids,clean


def _production_limits(limits,control,initial_count):
    required={'field_evaluations','control_cells','tensor_side','total_vertices',
              'root_bisections','error_mode','max_dyadic_depth','max_cell_record_bytes',
              'raw_journal_bytes','final_state_reserve_bytes'}
    _need(required<=set(limits) and not set(limits)-required-{'control_record_capacity_bytes'},
          'Complete frozen production limits required')
    cfg=deepcopy(limits)
    cfg.setdefault('control_record_capacity_bytes',16777216)
    for key in ((required-{'error_mode'}) | {'control_record_capacity_bytes'}):
        _count(cfg[key],key)
    cell_cap=32 if control else 4096
    _need(cfg['field_evaluations']<=cell_cap and initial_count<=cfg['control_cells']<=cell_cap,
          'Production attempt/cell ceilings may only reduce')
    _need(cfg['tensor_side']==8 and 64<=cfg['total_vertices']<=cell_cap*64,
          'Frozen64-node geometry and bounded total vertices required')
    _need(2<=cfg['root_bisections']<=96 and cfg['error_mode']=='tight',
          'Frozen tight coupling and per-root cap required')
    _need(cfg['max_dyadic_depth']<=32,'Production dyadic depth ceiling may only reduce')
    for key,maximum in (('max_cell_record_bytes',33554432),
                        ('raw_journal_bytes',134217728),
                        ('final_state_reserve_bytes',16777216),
                        ('control_record_capacity_bytes',16777216)):
        _need(0<cfg[key]<=maximum,'Reduced positive output ceiling required: '+key)
    return cfg


def _production_preflight(spec,predictions,limits,checkpoint,should_stop):
    _need(callable(checkpoint) and callable(should_stop),
          'Production requires checkpoint and soft-stop callbacks')
    mode=spec.get('mode',spec.get('phase','PRODUCTION'))
    control=mode=='DEVELOPMENT_PRODUCTION_CONTROL'
    _need(control or mode=='PRODUCTION','Unknown observation mode')
    _need(type(spec.get('schema_version')) is int and spec['schema_version']==1,
          'Production schema1 required')
    _need(type(spec.get('ks')) in (list,tuple) and tuple(spec['ks'])==KS
          and all(type(k) is int for k in spec['ks']),'Exact four laws required')
    ids,events=_production_events(spec.get('events'))
    analytic=spec.get('prediction_spec')
    analytic_limits=spec.get('prediction_limits')
    _need(type(analytic) is dict and type(analytic_limits) is dict,
          'Separate analytic prediction inputs required')
    analytic_ids,analytic_events=_production_events(analytic.get('events'))
    _need(type(analytic.get('schema_version')) is int and analytic['schema_version']==1
          and type(analytic.get('ks')) in (list,tuple) and tuple(analytic['ks'])==KS
          and all(type(k) is int for k in analytic['ks'])
          and type(analytic.get('row_count')) is int
          and analytic['row_count']==4*len(ids) and analytic_ids==ids and analytic_events==events,
          'Observation and analytic event/law tables differ')
    if control:
        contract=spec.get('development_contract')
        _need(type(contract) is dict
              and contract.get('record_class')=='BOUNDED_DEVELOPMENT_PRODUCTION_CONTROL'
              and contract.get('production_qualification') is False,
              'Explicit synthetic-receipt development contract required')
        _need(0<len(ids)<=2 and analytic.get('development_control') is True,
              'Only tiny analytic control tables allowed')
    else:
        # Reject wrong tables and structural fixtures before a possible72-row
        # analytic replay. Never find campaign data through the filesystem.
        captured=globals().get('__captured_data__')
        _need(type(captured) is dict,'Captured original campaign bytes required')
        raw=captured.get('CAMPAIGN_SPEC.json')
        _need(type(raw) is bytes and hashlib.sha256(raw).hexdigest()==_CAMPAIGN_SHA256,
              'Original campaign byte identity differs')
        original=json.loads(raw.decode('utf8'))
        _,declared_events=_production_events(original.get('events'))
        _need(len(ids)==18 and events==declared_events
              and original.get('ks')==list(KS) and original.get('row_count')==72,
              'Strict production requires the exact inherited72-row campaign')
        _need(set(analytic)=={'schema_version','ks','events','row_count'},
              'Production analytic input must be the exact semantic campaign projection')
    _need(type(predictions) is dict,'Completed numeric prediction packet required')
    numeric=predictions.get('artifact')
    _need(type(numeric) is dict
          and numeric.get('record_class')=='UNQUALIFIED_ANALYTIC_INTERVAL_OUTPUT'
          and numeric.get('numeric_prediction_qualified') is False,
          'Genuine unqualified numeric artifact required; fixtures cannot supply predictions')
    _need(predictions.get('development_receipts_only') is control,
          'Synthetic external receipts cannot qualify production')
    cells=spec.get('initial_cells')
    _need(validate_partition(cells),'Complete initial production partition required')
    cfg=_production_limits(limits,control,len(cells))
    _need(all(_dyadic_depth(c[key])<=cfg['max_dyadic_depth']
              for c in cells for key in ('Qa','Qb','Za','Zb')),
          'Initial cell exceeds frozen dyadic depth')
    _need(type(spec.get('refinement')) is bool,'Explicit refinement selection required')
    freeze=predictions.get('freeze')
    _need(type(freeze) is dict and freeze.get('completed') is True
              and type(freeze.get('sequence')) is int and freeze['sequence']>=1,
          'Completed positive freeze sequence required')
    start=spec.get('observation_sequence_start')
    _need(type(start) is int and start>freeze['sequence'],
          'Frozen observation sequence must follow prediction freeze')
    expected_source=spec.get('expected_prediction_source')
    _need(prediction.validate_numeric_prediction(predictions,analytic,analytic_limits,
              expected_source,None) is True,'Numeric prediction packet preflight rejected')
    predicted={}
    _need(type(numeric.get('rows')) is list and len(numeric['rows'])==4*len(ids),
          'Complete prediction rows required')
    for row in numeric['rows']:
        _need(type(row) is dict and type(row.get('k')) is int
              and row['k'] in KS and row.get('event_id') in ids,
              'Declared numeric prediction row required')
        key=row['k'],row['event_id']
        _need(key not in predicted,'Duplicate prediction row')
        predicted[key]=_interval(row.get('probability'),True)
    _need(len(predicted)==4*len(ids),'Complete unique numeric prediction rows required')
    return control,ids,events,cfg,predicted


def _production_text_envelope(value):
    if type(value) is str:
        _bounded_text(value,4096,'Journal text',False)
    elif type(value) is dict:
        for key,child in value.items():
            if type(key) is str:
                _bounded_text(key,4096,'Journal key',False)
            _production_text_envelope(child)
    elif type(value) in (list,tuple):
        for child in value:
            _production_text_envelope(child)


class _ProductionJournal:
    def __init__(self,checkpoint,cfg,start):
        self.checkpoint=checkpoint
        self.cfg=cfg
        self.next_sequence=start
        self.references=[]
        self.bytes=0
        self.submitted_bytes=0
        self.control_bytes=0
        self.failed=False
        self.error=None
        self.final_seen=False

    def capacity_for_attempt(self):
        remaining_controls=self.cfg['control_record_capacity_bytes']-self.control_bytes
        return self.bytes+self.cfg['max_cell_record_bytes']+self.cfg['final_state_reserve_bytes']+remaining_controls<=self.cfg['raw_journal_bytes']

    def write(self,record):
        if self.failed:
            return None
        try:
            _need(not self.final_seen,'No record after FINAL_STATE')
            kind=record['kind']
            _need(kind in ('INITIAL_PARTITION','FIELD_ATTEMPT_START','CELL_ATTEMPT',
                          'SPLIT_COMMIT','FINAL_STATE'),'Unknown journal kind')
            _production_text_envelope(record)
            raw=artifact.dumps(record)
            size=len(raw)
            remaining_controls=self.cfg['control_record_capacity_bytes']-self.control_bytes
            if kind=='CELL_ATTEMPT':
                _need(size<=self.cfg['max_cell_record_bytes'],'Complete cell record exceeds byte envelope')
                reserve=self.cfg['final_state_reserve_bytes']+remaining_controls
            elif kind=='FINAL_STATE':
                _need(size<=self.cfg['final_state_reserve_bytes'],'Complete final state exceeds reserved envelope')
                reserve=0
            else:
                _need(size<=1048576 and size<=remaining_controls,
                      'Complete control record exceeds reserved envelope')
                reserve=self.cfg['final_state_reserve_bytes']+remaining_controls-size
                if kind=='FIELD_ATTEMPT_START':
                    reserve+=self.cfg['max_cell_record_bytes']
            _need(self.bytes+size+reserve<=self.cfg['raw_journal_bytes'],
                  'Complete journal record exceeds reserved total envelope')
            # A callback gets a private copy. A mutation cannot change the bytes
            # whose digest was fixed above or silently mutate live engine state.
            self.submitted_bytes+=size
            ref=self.checkpoint(deepcopy(record))
            _need(type(ref) is dict,'Checkpoint reference required')
            _need(type(ref.get('sequence')) is int and ref['sequence']==self.next_sequence,
                  'Checkpoint sequence differs from frozen monotonic stream')
            _need(type(ref.get('bytes')) is int and ref['bytes']==size,
                  'Checkpoint byte count differs from complete record')
            _need(type(ref.get('sha256')) is str
                  and ref['sha256']==hashlib.sha256(raw).hexdigest(),
                  'Checkpoint digest differs from complete record')
            _need(ref.get('kind')==kind and type(ref.get('kind')) is str,
                  'Checkpoint kind differs from submitted record')
            _production_text_envelope(ref)
            _need(len(artifact.dumps(ref))<=1048576,'Reference exceeds control envelope')
            self.references.append(deepcopy(ref))
            self.next_sequence=ref['sequence']+1
            self.bytes+=size
            if kind not in ('CELL_ATTEMPT','FINAL_STATE'):
                self.control_bytes+=size
            self.final_seen=kind=='FINAL_STATE'
            return deepcopy(ref)
        except Exception as error:
            self.failed=True
            # An oversized original diagnostic is not shortened into qualified
            # evidence. Keep an explicit envelope failure and withdraw custody.
            message=str(error)
            try:
                _bounded_text(message,4096,'Writer diagnostic',False)
                self.error={'type':type(error).__name__,'message':message}
            except ValueError:
                self.error={'type':'DIAGNOSTIC_ENVELOPE_EXCEEDED',
                            'message':'Writer failed with a diagnostic outside the frozen envelope'}
            return None


def _run_production(spec,predictions,limits,checkpoint,should_stop):
    control,ids,events,cfg,predicted=_production_preflight(
        spec,predictions,limits,checkpoint,should_stop)
    # Callbacks cannot mutate a packet/specification they own after admission.
    # No full analytic replay is repeated just to check a returned start sequence.
    spec=deepcopy(spec)
    predictions=deepcopy(predictions)
    mode=spec.get('mode',spec.get('phase','PRODUCTION'))
    rows=[{'cell_id':'initial/'+str(i),'cell':_cell(c),'certificate_reference':None,
           'status':'NOT_ATTEMPTED_BUDGET','event_bounds':{name:(0,1) for name in ids}}
          for i,c in enumerate(spec['initial_cells'])]
    costs={'attempted_fields':0,'sampled_vertices':0,'root_bisections':0,
           'failed_fields':0,'created_cells':len(rows)}
    journal=_ProductionJournal(checkpoint,cfg,spec['observation_sequence_start'])
    state={'status':'OBSERVATIONS_RETAINED','stop_reason':None,'costs_exact':True,
           'unknown_cost_fields':[],'first_observation_sequence':None,
           'orphan_child_ids':[],'callback_error':None,'diagnostic_envelope_failure':False}
    initial={'kind':'INITIAL_PARTITION','mode':mode,'cells':[r['cell'] for r in rows],
             'cell_ids':[r['cell_id'] for r in rows],'limits':cfg,
             'prediction_freeze':deepcopy(predictions['freeze'])}
    journal.write(initial)

    def stop_before_attempt():
        if journal.failed:
            state['status']='INCONCLUSIVE_RAW_OUTPUT'
            return True
        try:
            reason=should_stop()
            _need(reason is None or (type(reason) is str and reason),
                  'Soft-stop callback must return None or a nonempty frozen reason')
            if reason is not None:
                _bounded_text(reason,4096,'Soft-stop reason')
                state['stop_reason']=reason
                state['status']='INCONCLUSIVE_BUDGET'
                return True
        except Exception as error:
            state['status']='IMPLEMENTATION_FAILURE'
            message=str(error)
            try:
                _bounded_text(message,4096,'Callback diagnostic',False)
                state['callback_error']={'type':type(error).__name__,'message':message}
            except ValueError:
                state['diagnostic_envelope_failure']=True
                state['callback_error']={'type':'DIAGNOSTIC_ENVELOPE_EXCEEDED',
                                         'message':'Soft-stop diagnostic exceeds frozen envelope'}
            return True
        if (costs['attempted_fields']>=cfg['field_evaluations']
                or costs['sampled_vertices']+64>cfg['total_vertices']
                or not journal.capacity_for_attempt()):
            state['status']='INCONCLUSIVE_BUDGET'
            state['stop_reason']='FIELD_VERTEX_OR_RAW_RESERVATION'
            return True
        return False

    def attempt(row):
        if stop_before_attempt():
            return False
        _bounded_text(row['cell_id'],256,'Cell ID')
        start_ref=journal.write({'kind':'FIELD_ATTEMPT_START','cell_id':row['cell_id'],
                                 'cell':row['cell'],'attempt_number':costs['attempted_fields']+1})
        if start_ref is None:
            state['status']='INCONCLUSIVE_RAW_OUTPUT'
            return False
        first=state['first_observation_sequence']
        if first is None:
            # This sequence is the writer's ACTUAL returned durable-start ref,
            # checked before the very first real certifier call.
            first=start_ref['sequence']
            packet_first=predictions.get('first_observation_sequence')
            if not (type(first) is int and first>predictions['freeze']['sequence']
                    and (packet_first is None or (type(packet_first) is int and packet_first==first))):
                journal.failed=True
                journal.error={'type':'FREEZE_ORDER_MISMATCH',
                               'message':'Actual first field-start sequence does not follow bound freeze'}
                state['status']='INCONCLUSIVE_RAW_OUTPUT'
                return False
            state['first_observation_sequence']=first
        one_limits={'field_evaluations':1,'control_cells':1,'tensor_side':8,
                    'total_vertices':64,'root_bisections':cfg['root_bisections'],'error_mode':'tight'}
        certificate=None
        error_record=None
        costs['attempted_fields']+=1
        try:
            certificate=fc.certify_cell(deepcopy(row['cell']),one_limits)
            _need(type(certificate) is dict and certificate.get('cell')==row['cell'],
                  'Actual source-bound cell certificate required')
            actual=certificate.get('costs')
            _need(type(actual) is dict,'Actual certificate costs required')
            _need(_count(actual.get('attempted_fields'),'certificate attempted_fields')==1,
                  'Certificate must retain its actual field attempt')
            nv=_count(actual.get('sampled_vertices'),'certificate sampled_vertices')
            nr=_count(actual.get('root_bisections'),'certificate root_bisections')
            _need(nv in (0,64) and nr<=3*cfg['root_bisections'],
                  'Certificate costs exceed admitted geometry or three per-root caps')
        except Exception as error:
            state['costs_exact']=False
            state['unknown_cost_fields'].append(row['cell_id'])
            costs['failed_fields']+=1
            row['status']='IMPLEMENTATION_FAILURE'
            error_record={'type':type(error).__name__,'message':str(error)}
            state['status']='IMPLEMENTATION_FAILURE'
        else:
            costs['sampled_vertices']+=nv
            costs['root_bisections']+=nr
            try:
                status=certificate.get('status')
                _need(status in ('RESOLVED','UNRESOLVED','IMPLEMENTATION_FAILURE'),
                      'Unknown cell certificate status')
                row['status']=status
                if 'refinement_terms' in certificate:
                    row['refinement_terms']=deepcopy(certificate['refinement_terms'])
                    _refinement_axis(row,row['cell'])
                if status=='RESOLVED':
                    coupling=certificate.get('coupling')
                    _need(type(coupling) is dict and coupling.get('verified') is True
                          and fc.verify_coupling_certificate(row['cell'],certificate['roots'],
                              certificate['grid'],certificate['observation'],coupling) is True,
                          'Resolved certificate requires actual verified field/UF/BFS coupling')
                    row['event_bounds']={event['id']:classify_event(certificate,event) for event in events}
                if status=='IMPLEMENTATION_FAILURE':
                    costs['failed_fields']+=1
                    state['status']='IMPLEMENTATION_FAILURE'
            except Exception as error:
                costs['failed_fields']+=1
                row['status']='IMPLEMENTATION_FAILURE'
                row['event_bounds']={name:(0,1) for name in ids}
                error_record={'type':type(error).__name__,'message':str(error)}
                state['status']='IMPLEMENTATION_FAILURE'
        if error_record is not None:
            row['error']=deepcopy(error_record)
            try:
                _bounded_text(error_record['type'],4096,'Error type')
                _bounded_text(error_record['message'],4096,'Error message',False)
            except ValueError:
                state['diagnostic_envelope_failure']=True
                journal.failed=True
                journal.error={'type':'DIAGNOSTIC_ENVELOPE_EXCEEDED',
                               'message':'Attempt diagnostic exceeds frozen envelope; raw qualification withdrawn'}
        record={'kind':'CELL_ATTEMPT','cell_id':row['cell_id'],'cell':row['cell'],
                'field_start_reference':start_ref,'certificate':certificate,
                'status':row['status'],'event_bounds':row['event_bounds'],
                'error':error_record,'costs_exact':state['costs_exact'],
                'cumulative_known_costs':deepcopy(costs)}
        ref=journal.write(record)
        row['certificate_reference']=ref
        if journal.failed:
            state['status']='INCONCLUSIVE_RAW_OUTPUT'
            return False
        return row['status']!='IMPLEMENTATION_FAILURE'

    # Initial tiling is completely retained before any field. Every initial cell
    # gets its turn before adaptive refinement can consume additional capacity.
    for row in rows:
        if not attempt(row):
            break
    else:
        if spec['refinement']:
            while True:
                choice=choose_refinement({'leaves':rows,'costs':costs,'limits':cfg},events)
                if choice['selected'] is None:
                    if choice['status']!='COMPLETE':
                        state['status']=choice['status']
                        state['stop_reason']='REFINEMENT_CAP_OR_DEPTH'
                    break
                parent_id=choice['cell_id']
                children=[{'cell_id':name,'cell':cell,'certificate_reference':None,
                           'status':'NOT_ATTEMPTED_BUDGET','event_bounds':{name_:(0,1) for name_ in ids}}
                          for name,cell in zip(choice['child_ids'],choice['children'])]
                for child in children:
                    _bounded_text(child['cell_id'],256,'Child cell ID')
                # Both allocated children count even if a stop leaves them as
                # unaggregated orphans. Parent remains live until durable commit.
                costs['created_cells']+=2
                state['orphan_child_ids'].extend(choice['child_ids'])
                if not attempt(children[0]) or not attempt(children[1]):
                    break
                commit={'kind':'SPLIT_COMMIT','parent_id':parent_id,
                        'child_ids':choice['child_ids'],'axis':choice['axis'],
                        'children':deepcopy(children),'cumulative_known_costs':deepcopy(costs)}
                if journal.write(commit) is None:
                    state['status']='INCONCLUSIVE_RAW_OUTPUT'
                    break
                rows[:]=[row for row in rows if row['cell_id']!=parent_id]+children
                for name in choice['child_ids']:
                    state['orphan_child_ids'].remove(name)
                # A failed/partial split never enters this live-partition update.
    aggregate=aggregate_cells(rows,events,KS)
    coupling_ok=(state['costs_exact'] and costs['failed_fields']==0
                 and state['callback_error'] is None and not state['diagnostic_envelope_failure'])
    raw_ok=not journal.failed and not state['diagnostic_envelope_failure']
    comparisons=[{'k':k,'event_id':name,'observed':aggregate['law_bounds'][k][name],
                  'predicted':predicted[k,name],
                  'status':compare_row(aggregate['law_bounds'][k][name],predicted[k,name],coupling_ok and raw_ok),
                  'provisional':True,'production_qualification':False}
                 for k in KS for name in ids]
    reported_costs=deepcopy(costs)
    if not state['costs_exact']:
        reported_costs['sampled_vertices']=None
        reported_costs['root_bisections']=None
    result={'mode':mode,'status':state['status'],'rows':rows,'aggregate':aggregate,
            'comparisons':comparisons,'costs':reported_costs,'costs_exact':state['costs_exact'],
            'known_costs_before_unaccounted_attempt':deepcopy(costs),
            'unknown_cost_fields':state['unknown_cost_fields'],'orphan_child_ids':state['orphan_child_ids'],
            'limits':cfg,'root_cap_scope':'PER_ROOT_NOT_CUMULATIVE',
            'root_call_upper_bound':3*costs['attempted_fields'],
            'stop_reason':state['stop_reason'],'callback_error':state['callback_error'],
            'first_observation_sequence':state['first_observation_sequence'],
            'prediction_freeze_reference':{key:deepcopy(predictions['freeze'][key]) for key in (
                'sequence','artifact_bytes','artifact_sha256','spec_sha256','limits_sha256')},
            'journal':deepcopy(journal.references),'raw_output_qualified':raw_ok,
            'raw_journal_bytes':journal.bytes if not journal.failed else None,
            'known_verified_raw_journal_bytes':journal.bytes,
            'submitted_raw_journal_bytes':journal.submitted_bytes,
            'control_record_bytes':journal.control_bytes,
            'raw_output_error':deepcopy(journal.error),'full_partition_verified':True,
            'prediction_freeze_verified':True,'all_coupling_gates_verified':coupling_ok,
            'complete_source_closure_verified':False,'external_runtime_qualified':False,
            'numeric_prediction_qualified':False,'numeric_prediction_packet_admitted':True,
            'synthetic_external_receipts_only':control,'production_qualification':False,
            'scientific_acceptance':False,
            'comparison_coverage_complete':len(comparisons)==4*len(ids),
            'all_comparisons_consistent':all(row['status'] in (
                'FINITE_WINDOW_CONSISTENT','FINITE_WINDOW_CONSISTENT_ZERO') for row in comparisons),
            'full_declared_campaign':not control and len(comparisons)==72}
    if journal.failed or state['diagnostic_envelope_failure']:
        result=withdraw_acceptance(result,'INCONCLUSIVE_RAW_OUTPUT')
    elif not coupling_ok:
        result=withdraw_acceptance(result,'IMPLEMENTATION_FAILURE')
    final_ref=journal.write({'kind':'FINAL_STATE','state':deepcopy(result)})
    result['journal']=deepcopy(journal.references)
    result['final_state_retained']=final_ref is not None
    result['raw_journal_bytes']=journal.bytes if not journal.failed else None
    result['known_verified_raw_journal_bytes']=journal.bytes
    result['submitted_raw_journal_bytes']=journal.submitted_bytes
    result['control_record_bytes']=journal.control_bytes
    if final_ref is None:
        result['raw_output_qualified']=False
        result['raw_output_error']=deepcopy(journal.error)
        result=withdraw_acceptance(result,'INCONCLUSIVE_RAW_OUTPUT')
    return result
