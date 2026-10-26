"""Source-only causal controls. Only Root executes. No actual72 prediction.
Synthetic external receipts below confer no actual runtime/review qualification.
The canonical sink retains complete raw records outside the reference-only result.
"""
import hashlib
import json
import unittest
from copy import deepcopy
from fractions import Fraction as Q
import artifact
import exact_h0
import field_certificate as fc
import prediction
import run_falsifier as runner

KS = (8, 12, 15, 16)
CAMPAIGN_SHA256 = 'be65c90e312a264d48e58ece142c6d2d93b1c73e564e08ada02ad656f9fefad3'
SOURCE = {'synthetic-development.py': {'bytes': 17, 'sha256': 'a'*64}}
EVENT = {'id': 'count', 'lifetime_upper': Q(1,4), 'separation_upper': None, 'birth_upper': None}


class patched:
    """Test-only external boundary replacement with unconditional restoration."""
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.previous = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.module, self.name, self.previous)
        return False


def analytic_spec(rich=False):
    events = [dict(EVENT)]
    if rich:
        events.append({'id': 'physical_birth', 'lifetime_upper': Q(9,64),
                       'separation_upper': Q(1,16), 'birth_upper': Q(0)})
    return {'schema_version': 1, 'ks': list(KS), 'events': events,
            'row_count': 4*len(events), 'development_control': True}


def analytic_limits():
    return {'max_endpoint_evaluations': 6, 'max_panels': 2, 'max_refinements': 0,
            'initial_partition': [Q(0), Q(1,2), Q(1)], 'max_root_bisections': 16,
            'max_cap_bisections': 4, 'precision_bits': 128}


def cells(micro=False):
    cuts = (Q(0), Q(1,2), Q(1,2)+Q(1,2**24), Q(1)) if micro else (Q(0), Q(1))
    return [dict(Qa=a,Qb=b,Za=c,Zb=d,epsilon=s) for s in (-1,1)
            for a,b in zip(cuts,cuts[1:]) for c,d in zip(cuts,cuts[1:])]


def limits(fields=2, **changes):
    out = {'field_evaluations': fields, 'control_cells': 32, 'tensor_side': 8,
           'total_vertices': 64*max(1,fields), 'root_bisections': 96, 'error_mode': 'tight',
           'max_dyadic_depth': 32, 'max_cell_record_bytes': 33554432,
           'raw_journal_bytes': 134217728, 'final_state_reserve_bytes': 16777216}
    out.update(changes)
    return out


class Sink:
    """Controlled external writer; refs bind complete retained canonical bytes."""
    def __init__(self, test, damage=None, fail_kind=None):
        self.test, self.damage, self.fail_kind = test, damage, fail_kind
        self.records, self.references = [], []
        self.sequence = 2

    def __call__(self, record):
        self.test.assertIs(type(record), dict)
        raw = self.test.api(artifact, 'dumps', record)
        retained = self.test.api(artifact, 'loads', raw, len(raw))
        self.test.assertEqual(retained, record)
        self.records.append(retained)
        if record['kind'] == self.fail_kind:
            raise OSError('synthetic-development writer failure')
        ref = dict(sequence=self.sequence, bytes=len(raw), kind=record['kind'],
                   sha256=hashlib.sha256(raw).hexdigest())
        self.sequence += 1
        if self.damage:
            ref = self.damage(deepcopy(ref))
        self.references.append(deepcopy(ref))
        return ref


class ProductionTests(unittest.TestCase):
    def api(self, module, name, *args, **kwargs):
        fn = getattr(module, name, None)
        self.assertTrue(callable(fn), 'Missing production API: '+name)
        try:
            got = fn(*args, **kwargs)
        except (NotImplementedError, AttributeError, TypeError) as error:
            self.fail('Production feature unavailable: %s: %s' % (name,error))
        self.assertIsNotNone(got, 'Inert production API: '+name)
        return got

    def rejected(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except (ValueError, OverflowError):
            return
        except (NotImplementedError, AttributeError, TypeError) as error:
            self.fail('Missing rejection implementation: '+str(error))
        self.fail('Malformed input was admitted')

    def seal(self, packet, spec, cap):
        for key,value in (('artifact',packet['artifact']), ('spec',spec), ('limits',cap)):
            raw = self.api(artifact,'dumps',value)
            packet['freeze'][key+'_sha256'] = hashlib.sha256(raw).hexdigest()
            if key == 'artifact':
                packet['freeze']['artifact_bytes'] = len(raw)
        packet['review']['artifact_sha256'] = packet['freeze']['artifact_sha256']
        return packet

    def packet(self, spec=None, cap=None):
        spec = analytic_spec() if spec is None else spec
        cap = analytic_limits() if cap is None else cap
        packet = {'artifact': self.api(prediction,'predict',spec,cap), 'limits': deepcopy(cap),
                  'freeze': {'completed': True, 'sequence': 1}, 'first_observation_sequence': None,
                  'source_before': deepcopy(SOURCE), 'source_after': deepcopy(SOURCE),
                  'terminal': dict(observed=True,waited=True,owned_child_pid=123,returncode=0),
                  'author_id': 'synthetic-author', 'review': {'completed': True,'nonauthor': True,
                      'author_id': 'synthetic-author','reviewer_id': 'synthetic-reviewer',
                      'verdict': 'ACCEPT_NUMERIC_ARTIFACT_SOURCE_EXPOSED'},
                  'development_receipts_only': True}
        return self.seal(packet,spec,cap)

    def spec(self, micro=False, refinement=False):
        return {'schema_version': 1, 'mode': 'DEVELOPMENT_PRODUCTION_CONTROL',
                'ks': list(KS),'events': [dict(EVENT)],'initial_cells': cells(micro),
                'refinement': refinement,'prediction_spec': analytic_spec(),
                'prediction_limits': analytic_limits(),'expected_prediction_source': deepcopy(SOURCE),
                'observation_sequence_start': 2,'development_contract': {
                    'record_class': 'BOUNDED_DEVELOPMENT_PRODUCTION_CONTROL',
                    'production_qualification': False,
                    'purpose': 'synthetic external receipts; actual tiny field/analytic control'}}

    def control(self, spec=None, packet=None, cap=None, sink=None, stop=None):
        spec = self.spec() if spec is None else spec
        packet = self.packet(spec['prediction_spec'],spec['prediction_limits']) if packet is None else packet
        sink = Sink(self) if sink is None else sink
        got = self.api(runner,'run',spec,packet,limits() if cap is None else cap,
                       checkpoint=sink,should_stop=(lambda: None) if stop is None else stop)
        self.assertIs(type(got),dict)
        return got,sink

    def full_mass(self, got):
        self.assertTrue(got['aggregate']['partition_verified'])
        self.assertTrue(runner.validate_partition([r['cell'] for r in got['rows']]))
        for k in KS:
            self.assertEqual(got['aggregate']['total_mass'][k],Q(1))
        self.assertEqual({(r['k'],r['event_id']) for r in got['comparisons']}, {(k,'count') for k in KS})
        self.assertEqual(len(got['comparisons']),4)
        self.assertFalse(got['production_qualification'])
        self.assertFalse(got['full_declared_campaign'])

    # Break: serialization erases tuple/Fraction identity or collides 8 and '8'.
    def test_artifact_exact_roundtrip_and_deterministic_key_order(self):
        value = {8: (Q(1),1,None,True,False),'8': [Q(-1,2),'é'],'nested': {'q': Q(1,3)}}
        raw = self.api(artifact,'dumps',value)
        self.assertIs(type(raw),bytes)
        self.assertTrue(raw.endswith(b'\n'))
        raw.decode('ascii')
        got = self.api(artifact,'loads',raw,len(raw))
        self.assertEqual(got,value)
        for value,want in ((got[8],tuple),(got['8'],list),(got[8][0],Q),
                           (got[8][1],int),(got[8][3],bool)):
            self.assertIs(type(value),want)
        self.assertEqual(self.api(artifact,'dumps',got),raw)
        reverse = dict(reversed(list(got.items())))
        self.assertEqual(self.api(artifact,'dumps',reverse),raw)

    # Break: parser normalizes noncanonical, malformed or over-budget representations.
    def test_artifact_rejects_bad_wire_types_depth_digits_and_byte_limit(self):
        raw = self.api(artifact,'dumps',Q(1,2))
        for bad in (b' '+raw,raw[:-1],raw+b'\n',b'{"x":1,"x":2}\n',b'NaN\n',
                    b'Infinity\n',b'1.0\n',b'{"unknown":1}\n',
                    raw.replace(b'"2"',b'"0"'),raw.replace(b'"1"',b'"2"')):
            with self.subTest(raw=bad[:80]):
                self.rejected(artifact.loads,bad,4096)
        self.rejected(artifact.loads,raw,len(raw)-1)
        self.rejected(artifact.loads,raw,True)
        self.rejected(artifact.loads,raw.decode('ascii'),4096)
        for bad in (1.0,float('nan'),{True:'bad'},{Q(1):1},{None:1},{1,2},object()):
            with self.subTest(value=type(bad).__name__):
                self.rejected(artifact.dumps,bad)
        deep = None
        for _ in range(66):
            deep = [deep]
        self.rejected(artifact.dumps,deep)
        self.rejected(artifact.dumps,10**10000)

    # Break: checksum-only validation accepts a modified and rehashed witness.
    def test_numeric_replay_rejects_rehashed_witness_weight_row_and_counter_tamper(self):
        spec,cap = analytic_spec(True),analytic_limits()
        good = self.packet(spec,cap)
        self.assertTrue(self.api(prediction,'verify_numeric_artifact',good['artifact'],spec,cap))
        self.assertTrue(self.api(prediction,'validate_numeric_prediction',good,spec,cap,SOURCE))
        changes = [lambda p:p['artifact']['rows'].pop(),
            lambda p:p['artifact']['rows'].append(deepcopy(p['artifact']['rows'][0])),
            lambda p:p['artifact'].__setitem__('endpoint_evaluations',0),
            lambda p:p['artifact']['rows'][0]['panels'][0].__setitem__('weight',Q(1)),
            lambda p:p['artifact']['rows'][0]['endpoint_witnesses']['1/2'].__setitem__('p',(Q(1),Q(1))),
            lambda p:p['artifact']['rows'][0]['endpoint_witnesses']['1'].__setitem__('integrand',(Q(1),Q(1))),
            lambda p:p['artifact']['rows'][-1]['endpoint_witnesses']['0'].__setitem__('physical_cap',(Q(0),Q(0))),
            lambda p:p['artifact']['rows'][-1]['birth_cutoff'].__setitem__('unknown_strip_mass',Q(1)),
            lambda p:p['artifact'].__setitem__('numeric_prediction_qualified',True),
            lambda p:p['artifact'].__setitem__('record_class','DEVELOPMENT_STRUCTURE_FIXTURE')]
        for i,change in enumerate(changes):
            with self.subTest(mutation=i):
                bad = deepcopy(good)
                change(bad)
                self.seal(bad,spec,cap)
                self.assertFalse(self.api(prediction,'verify_numeric_artifact',bad['artifact'],spec,cap))
                self.assertFalse(self.api(prediction,'validate_numeric_prediction',bad,spec,cap,SOURCE))
        changed = dict(cap,max_root_bisections=17)
        self.assertFalse(self.api(prediction,'validate_numeric_prediction',good,spec,changed,SOURCE))

    # Break: equal arbitrary source maps, bool counters or forged order qualify.
    def test_packet_external_receipt_source_freeze_order_and_strict_types(self):
        spec,cap = analytic_spec(),analytic_limits()
        good = self.packet(spec,cap)
        self.assertTrue(self.api(prediction,'validate_numeric_prediction',good,spec,cap,SOURCE,3))
        changes = [(['freeze','completed'],False),(['freeze','sequence'],0),
            (['freeze','sequence'],True),(['freeze','artifact_bytes'],True),
            (['freeze','artifact_sha256'],'0'*64),(['freeze','spec_sha256'],'0'*64),
            (['freeze','limits_sha256'],'0'*64),(['terminal','observed'],False),
            (['terminal','waited'],False),(['terminal','owned_child_pid'],True),
            (['terminal','returncode'],True),(['terminal','returncode'],1),
            (['review','completed'],False),(['review','nonauthor'],False),
            (['review','reviewer_id'],'synthetic-author'),(['review','artifact_sha256'],'0'*64),
            (['review','verdict'],'NOT_REVIEWED'),(['source_after'],{}),(['source_before'],{}),
            (['freeze'],None),(['limits'],dict(cap,precision_bits=True))]
        for path,value in changes:
            with self.subTest(path=path,value=value):
                bad,parent = deepcopy(good),None
                parent = bad
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                self.assertFalse(self.api(prediction,'validate_numeric_prediction',bad,spec,cap,SOURCE))
        for first in (0,1,True,'3'):
            with self.subTest(first=first):
                self.assertFalse(self.api(prediction,'validate_numeric_prediction',good,spec,cap,SOURCE,first))
        bad = deepcopy(good)
        bad['first_observation_sequence'] = 1
        self.assertFalse(self.api(prediction,'validate_numeric_prediction',bad,spec,cap,SOURCE))
        bad = deepcopy(good)
        bad['source_before'] = bad['source_after'] = {'other.py': {'bytes':1,'sha256':'b'*64}}
        self.assertFalse(self.api(prediction,'validate_numeric_prediction',bad,spec,cap,SOURCE))
        no_development = deepcopy(spec)
        no_development.pop('development_control')
        self.assertFalse(self.api(prediction,'validate_numeric_prediction',good,no_development,cap,SOURCE))

    # Break: invalid packet/callback reaches a real field.
    def test_preflight_rejects_bad_packets_and_missing_callbacks_before_field(self):
        spec,good = self.spec(),self.packet()
        bads = [None,{},deepcopy(good),deepcopy(good)]
        bads[2]['freeze']['completed'] = False
        bads[3]['artifact']['record_class'] = 'DEVELOPMENT_STRUCTURE_FIXTURE'
        def forbidden(*args,**kwargs):
            self.fail('Field called before packet/callback admission')
        with patched(fc,'certify_cell',forbidden):
            for bad in bads:
                with self.subTest(packet=type(bad).__name__):
                    self.rejected(runner.run,spec,bad,limits(),checkpoint=Sink(self),should_stop=lambda:None)
            for checkpoint,stop in ((None,lambda:None),(Sink(self),None),(17,lambda:None),(Sink(self),17)):
                self.rejected(runner.run,spec,good,limits(),checkpoint=checkpoint,should_stop=stop)

    # Break: strict production relaxes exact72 or starts a forbidden full target calculation.
    def test_strict_production_requires_captured_exact72_without_calculating_target(self):
        captured = globals().get('__captured_data__')
        self.assertIs(type(captured),dict,'Harness captured bytes required; no filesystem fallback')
        raw = captured.get('CAMPAIGN_SPEC.json')
        self.assertIs(type(raw),bytes)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),CAMPAIGN_SHA256)
        original = json.loads(raw.decode('utf-8'))
        campaign = {key:original[key] for key in ('schema_version','ks','events','row_count')}
        expected = {(k,e['id']) for k in KS for e in campaign['events']}
        self.assertEqual(len(expected),72)
        self.assertEqual(campaign['ks'],list(KS))
        tiny,packet = self.spec(),self.packet()
        tiny['mode'] = 'PRODUCTION'
        structural = deepcopy(packet)
        structural['artifact'] = {'record_class':'DEVELOPMENT_STRUCTURE_FIXTURE',
            'numeric_prediction_qualified':False,'rows':[
                {'k':k,'event_id':e['id'],'probability':(Q(0),Q(1))}
                for k in KS for e in campaign['events']]}
        self.assertEqual({(r['k'],r['event_id']) for r in structural['artifact']['rows']},expected)
        full = deepcopy(tiny)
        full['prediction_spec'],full['events'] = campaign,campaign['events']
        def forbidden(*args,**kwargs):
            self.fail('Rejected strict table must not calculate actual72 or evaluate fields')
        with patched(prediction,'predict',forbidden),patched(fc,'certify_cell',forbidden):
            for spec,record in ((tiny,packet),(full,structural)):
                self.rejected(runner.run,spec,record,limits(),checkpoint=Sink(self),should_stop=lambda:None)

    # Break: caller sequence is trusted instead of actual durable start reference.
    def test_actual_start_sequence_checked_before_certifier(self):
        packet = self.packet()
        sink = Sink(self,damage=lambda r:dict(r,sequence=1) if r['kind']=='FIELD_ATTEMPT_START' else r)
        def forbidden(*args,**kwargs):
            self.fail('Actual premature FIELD_ATTEMPT_START reached certifier')
        with patched(fc,'certify_cell',forbidden):
            try:
                got,_ = self.control(packet=packet,sink=sink)
            except ValueError:
                return
        self.assertFalse(got['raw_output_qualified'])
        self.full_mass(got)

    # Break: shared engine fabricates bars, discards raw attempts or miscounts failed work.
    def test_shared_engine_actual18cell_UF_BFS_raw_retention_and_counted_costs(self):
        got,sink = self.control(spec=self.spec(micro=True),cap=limits(18))
        self.full_mass(got)
        self.assertEqual(got['mode'],'DEVELOPMENT_PRODUCTION_CONTROL')
        self.assertEqual(len(got['rows']),18)
        self.assertEqual(got['costs']['attempted_fields'],18)
        self.assertTrue(got['costs_exact'])
        self.assertTrue(got['raw_output_qualified'])
        self.assertEqual(sink.records[0]['kind'],'INITIAL_PARTITION')
        self.assertEqual(sink.records[-1]['kind'],'FINAL_STATE')
        self.assertTrue(runner.validate_partition(sink.records[0]['cells']))
        self.assertEqual(got['journal'],sink.references)
        self.assertTrue(all('certificate' not in r for r in got['journal']))
        attempts = [r for r in sink.records if r['kind']=='CELL_ATTEMPT']
        self.assertEqual(len(attempts),18)
        self.assertEqual(sum(r['kind']=='FIELD_ATTEMPT_START' for r in sink.records),18)
        first = next(r['sequence'] for r in sink.references if r['kind']=='FIELD_ATTEMPT_START')
        self.assertEqual(got['first_observation_sequence'],first)
        self.assertGreater(first,1)
        certificates = [r['certificate'] for r in attempts]
        resolved = [c for c in certificates if c['status']=='RESOLVED']
        self.assertGreaterEqual(len(resolved),2)
        for cert in resolved:
            values,observed = cert['grid']['values'],cert['observation']
            self.assertEqual(len(values),64)
            self.assertEqual(observed['barcode'],exact_h0.compute(values,8))
            self.assertTrue(exact_h0.verify_by_connectivity(values,8,observed['barcode']))
            self.assertTrue(observed['independent_bfs_verified'])
            self.assertTrue(observed['merge_witnesses'])
            self.assertTrue(cert['coupling']['verified'])
        for key in ('attempted_fields','sampled_vertices','root_bisections'):
            self.assertEqual(got['costs'][key],sum(c['costs'][key] for c in certificates))
        by_id = {r['cell_id']:ref for r,ref in zip(sink.records,sink.references) if r['kind']=='CELL_ATTEMPT'}
        for row in got['rows']:
            self.assertNotIn('certificate',row)
            self.assertEqual(row['certificate_reference'],by_id[row['cell_id']])
        # Mutate actual raw grid, BFS witnesses and physical coupling; no fake bars.
        mutations = [lambda c:c['grid']['values'].__setitem__(0,c['grid']['values'][0]+1),
            lambda c:c['observation']['merge_witnesses'][0].__setitem__('path',[]),
            lambda c:c['observation']['merge_witnesses'][0].__setitem__('cut_vertices',[]),
            lambda c:c['observation']['merge_witnesses'][0].__setitem__('older_vertex',-1),
            lambda c:c['coupling'].__setitem__('saddle_root_index',0)]
        for key in ('h','dQ','dZ','rounding'):
            mutations.append(lambda c,key=key:c['coupling']['error_constants'].__setitem__(key,Q(1)))
        for i,mutate in enumerate(mutations):
            with self.subTest(raw_mutation=i):
                bad = deepcopy(resolved[0])
                mutate(bad)
                self.assertFalse(fc.verify_coupling_certificate(bad['cell'],bad['roots'],bad['grid'],
                                                               bad['observation'],bad['coupling']))

    # Break: missing root coupling terms cause Q-only starvation at third split.
    def test_R01_actual_boundary_failure_without_terms_third_split_is_Z(self):
        leaves = []
        for name,c in zip(('a','b'),cells()):
            cert = self.api(fc,'certify_cell',c,limits(1))
            self.assertNotEqual(cert['status'],'RESOLVED')
            self.assertNotIn('refinement_terms',cert)
            leaves.append({'cell_id':name,'cell':c,'status':cert['status'],'event_bounds':{'count':(0,1)}})
        axes = []
        for _ in range(3):
            state = {'leaves':leaves,'costs':{'attempted_fields':2,'failed_fields':0,
                      'sampled_vertices':0,'created_cells':len(leaves)},'limits':limits(20)}
            selected = self.api(runner,'choose_refinement',state,[EVENT])
            axes.append(selected['axis'])
            leaves = [r for r in leaves if r['cell_id']!=selected['cell_id']] + [
                {'cell_id':name,'cell':c,'status':'UNRESOLVED','event_bounds':{'count':(0,1)}}
                for name,c in zip(selected['child_ids'],selected['children'])]
            self.assertTrue(runner.validate_partition([r['cell'] for r in leaves]))
            aggregate = runner.aggregate_cells(leaves,[EVENT])
            for k in KS:
                self.assertEqual(aggregate['total_mass'][k],Q(1))
                self.assertEqual(aggregate['unresolved_mass'][k]['count'],Q(1))
        self.assertEqual(axes,['Q','Q','Z'])

    # Break: field cap discards unattempted mass or normalizes over successes.
    def test_one_field_cap_retains_all_unknown_mass_and_comparisons(self):
        got,sink = self.control(cap=limits(1))
        self.full_mass(got)
        self.assertEqual(got['costs']['attempted_fields'],1)
        self.assertEqual(len(got['rows']),2)
        self.assertEqual(sum(r['kind']=='CELL_ATTEMPT' for r in sink.records),1)
        for k in KS:
            self.assertEqual(got['aggregate']['law_bounds'][k]['count'],(Q(0),Q(1)))
            self.assertEqual(got['aggregate']['unresolved_mass'][k]['count'],Q(1))

    # Break: raw reservation or soft deadline checked only after observation.
    def test_soft_stop_and_raw_reservation_prevent_field_with_full_mass(self):
        packet = self.packet()
        def forbidden(*args,**kwargs):
            self.fail('Field executed after stop or failed raw reservation')
        with patched(fc,'certify_cell',forbidden):
            for cap,stop in ((limits(),lambda:'SOFT_DEADLINE'),
                             (limits(raw_journal_bytes=16777216+33554432-1),lambda:None)):
                with self.subTest(cap=cap):
                    got,sink = self.control(packet=packet,cap=cap,stop=stop)
                    self.full_mass(got)
                    self.assertEqual(got['costs']['attempted_fields'],0)
                    self.assertFalse(any(r['kind']=='CELL_ATTEMPT' for r in sink.records))

    # Break: bad references silently qualify output, writer failure disappears.
    def test_bad_reference_and_writer_failure_withdraw_raw_qualification(self):
        packet = self.packet()
        damages = [lambda r:dict(r,sha256='0'*64),lambda r:dict(r,bytes=r['bytes']+1),
                   lambda r:dict(r,bytes=True),lambda r:dict(r,sequence=True),
                   lambda r:dict(r,sequence=0),lambda r:dict(r,kind='WRONG')]
        for i,damage in enumerate(damages):
            with self.subTest(damage=i):
                try:
                    got,_ = self.control(packet=packet,sink=Sink(self,damage=damage))
                except ValueError:
                    continue
                self.assertFalse(got['raw_output_qualified'])
                self.full_mass(got)
        got,sink = self.control(packet=packet,sink=Sink(self,fail_kind='CELL_ATTEMPT'))
        self.assertFalse(got['raw_output_qualified'])
        self.full_mass(got)
        self.assertTrue(any(r['kind']=='CELL_ATTEMPT' for r in sink.records))

    # Break: failed partial work becomes zero cost and scheduling continues.
    def test_certifier_exception_retains_unknown_cost_full_mass_and_error(self):
        packet = self.packet()
        def failing(*args,**kwargs):
            raise RuntimeError('synthetic-development partial work failure')
        with patched(fc,'certify_cell',failing):
            got,sink = self.control(packet=packet)
        self.full_mass(got)
        self.assertEqual(got['costs']['attempted_fields'],1)
        self.assertFalse(got['costs_exact'])
        self.assertIsNone(got['costs']['sampled_vertices'])
        self.assertIsNone(got['costs']['root_bisections'])
        self.assertTrue(got['unknown_cost_fields'])
        attempts = [r for r in sink.records if r['kind']=='CELL_ATTEMPT']
        self.assertEqual(len(attempts),1)
        self.assertEqual(attempts[0]['error']['type'],'RuntimeError')

    # Break: partial children replace parent before durable commit, losing mass.
    def test_split_writer_failure_keeps_parent_and_charges_orphan_children(self):
        got,sink = self.control(spec=self.spec(refinement=True),cap=limits(8),
                               sink=Sink(self,fail_kind='SPLIT_COMMIT'))
        self.full_mass(got)
        self.assertFalse(got['raw_output_qualified'])
        commits = [r for r in sink.records if r['kind']=='SPLIT_COMMIT']
        self.assertEqual(len(commits),1)
        live = {r['cell_id'] for r in got['rows']}
        self.assertIn(commits[0]['parent_id'],live)
        self.assertTrue(set(commits[0]['child_ids']).isdisjoint(live))
        attempts = [r for r in sink.records if r['kind']=='CELL_ATTEMPT']
        self.assertTrue(set(commits[0]['child_ids']) <= {r['cell_id'] for r in attempts})
        self.assertEqual(got['costs']['attempted_fields'],len(attempts))

    # Break: successful split double-counts parent or commits before both attempts.
    def test_split_commit_after_children_replaces_parent_without_double_counting(self):
        sink = Sink(self)
        def stop():
            return 'SOFT_DEADLINE' if any(r['kind']=='SPLIT_COMMIT' for r in sink.records) else None
        got,sink = self.control(spec=self.spec(refinement=True),cap=limits(8),sink=sink,stop=stop)
        self.full_mass(got)
        index = next(i for i,r in enumerate(sink.records) if r['kind']=='SPLIT_COMMIT')
        commit = sink.records[index]
        prior = {r['cell_id'] for r in sink.records[:index] if r['kind']=='CELL_ATTEMPT'}
        self.assertTrue(set(commit['child_ids']) <= prior)
        live = {r['cell_id'] for r in got['rows']}
        self.assertNotIn(commit['parent_id'],live)
        self.assertTrue(set(commit['child_ids']) <= live)

    # Break: field/created-cell/vertex/depth caps still schedule refinement.
    def test_refinement_caps_preserve_leaves_without_scheduling(self):
        leaves = [{'cell_id':str(i),'cell':c,'status':'UNRESOLVED','event_bounds':{'count':(0,1)}}
                  for i,c in enumerate(cells())]
        controls = [(limits(2),dict(attempted_fields=2,sampled_vertices=0)),
                    (limits(4,control_cells=3),dict(attempted_fields=0,sampled_vertices=0)),
                    (limits(4,total_vertices=64),dict(attempted_fields=0,sampled_vertices=64)),
                    (limits(4,max_dyadic_depth=0),dict(attempted_fields=0,sampled_vertices=0))]
        for cap,cost in controls:
            with self.subTest(cap=cap):
                state = {'leaves':deepcopy(leaves),'costs':dict(cost,failed_fields=0,created_cells=2),'limits':cap}
                got = self.api(runner,'choose_refinement',state,[EVENT])
                self.assertIsNone(got['selected'])
                self.assertEqual(got['evaluations_scheduled'],0)
                self.assertEqual(got['leaves'],leaves)

    # Break: duplicates, missing failed mass or a missing sign pass aggregation.
    def test_partition_rejects_missing_duplicate_and_one_sign(self):
        rows = [{'cell_id':str(i),'cell':c,'status':'UNRESOLVED','event_bounds':{'count':(0,1)}}
                for i,c in enumerate(cells())]
        for bad in (rows[:1],rows+[deepcopy(rows[0])],[]):
            with self.subTest(rows=len(bad)):
                self.rejected(runner.aggregate_cells,bad,[EVENT])

    # Break: outer failure leaves nested qualified PASS or destroys exact evidence.
    def test_recursive_finalization_withdraws_labels_and_retains_exact_evidence(self):
        original = {'status':'PASS','full_declared_campaign':True,
            'full_partition_verified':True,'prediction_freeze_verified':True,
            'complete_source_closure_verified':True,'all_coupling_gates_verified':True,
            'rows':[{'status':'FINITE_WINDOW_CONSISTENT','qualified':True,
                     'observed':(Q(1,4),Q(1,4)),'error':{'retained':'original'}}],
            'nested':({'status':'RESOLVED','production_qualification':True,'raw':[1,Q(1,2)]},)}
        source = deepcopy(SOURCE)
        terminal = dict(observed=True,waited=True,owned_child_pid=123,returncode=0)
        budget = dict(exhausted=False,elapsed_seconds='1',wall_limit_seconds='2')
        controls = [(source,{},terminal,budget),(source,source,{},budget),
                    (source,source,dict(terminal,returncode=1),budget),
                    (source,source,terminal,dict(budget,elapsed_seconds='2'))]
        for before,after,term,cap in controls:
            with self.subTest(after=after,terminal=term,budget=cap):
                got = self.api(runner,'finalize_attempt',original,before,after,term,cap)
                self.assertNotEqual(got['status'],'PASS')
                self.assertFalse(got['full_declared_campaign'])
                self.assertFalse(got['rows'][0]['qualified'])
                self.assertFalse(got['nested'][0]['production_qualification'])
                self.assertEqual(got['rows'][0]['computed_status_before_withdrawal'],'FINITE_WINDOW_CONSISTENT')
                self.assertEqual(got['rows'][0]['observed'],original['rows'][0]['observed'])
                self.assertEqual(got['rows'][0]['error'],original['rows'][0]['error'])
                self.assertEqual(got['nested'][0]['raw'],original['nested'][0]['raw'])
        self.assertEqual(original['status'],'PASS')


if __name__ == '__main__':
    unittest.main()
