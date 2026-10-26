"""New prediction RED controls; inert preparation, no execution by this author.

The five D01 files remain byte-identical. Tests call prediction entry points
directly. Synthetic packets are explicitly DEVELOPMENT_STRUCTURE_FIXTURE and confer no
runtime, numerical campaign, source-authentication or science credit.
"""
import unittest
import hashlib
import json
from copy import deepcopy
from fractions import Fraction as Q
import prediction

CAMPAIGN_SHA256='be65c90e312a264d48e58ece142c6d2d93b1c73e564e08ada02ad656f9fefad3'
ANALYTIC_SHA256='e1dd749117f141a0f05e9c6c36bcac154efa137164177a0304f741632796a3c5'
KS=(8,12,15,16)
def campaign():
    captured=globals().get('__captured_data__')
    if type(captured) is not dict:
        raise RuntimeError('Captured test data missing: __captured_data__ map required')
    raw=captured.get('CAMPAIGN_SPEC.json')
    if type(raw) is not bytes:
        raise RuntimeError('Captured test data missing: CAMPAIGN_SPEC.json bytes required')
    if hashlib.sha256(raw).hexdigest()!=CAMPAIGN_SHA256:
        raise RuntimeError('Captured campaign SHA256 mismatch; no test/prediction qualification')
    return json.loads(raw.decode('utf-8'))

def synthetic(event_id='unrestricted',tau=None,radius=None,birth=None,ks=KS):
    return {'schema_version':1,'ks':list(ks),
            'events':[{'id':event_id,'lifetime_upper':tau,
                       'separation_upper':radius,'birth_upper':birth}],
            'row_count':len(ks),'development_control':True}

def limits(**changes):
    out={'max_endpoint_evaluations':8,'max_panels':2,'max_refinements':0,
         'initial_partition':[Q(0),Q(1,2),Q(1)],
         'max_root_bisections':128,'max_cap_bisections':128,
         'precision_bits':128}
    out.update(changes)
    return out

def canonical_payload(payload):
    return (json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')

def literal_packet(spec):
    # Complete conservative interval witnesses are valid even without
    # numerical calculation. This fixture is never a production artifact.
    rows=[]
    for k in spec['ks']:
        for event in spec['events']:
            rows.append({'k':k,'event_id':event['id'],'probability':['0','1'],
                         'status':'INCONCLUSIVE_BUDGET',
                         'uncertainty':{'uncomputed_full_mass':'1'}})
    payload={'schema_version':1,'record_class':'DEVELOPMENT_STRUCTURE_FIXTURE',
             'status':'INCONCLUSIVE_BUDGET','analytic_spec_sha256':ANALYTIC_SHA256,
             'row_count':spec['row_count'],'rows':rows,
             'numeric_prediction_qualified':False}
    return {'artifact':payload,
            'freeze':{'artifact_sha256':hashlib.sha256(canonical_payload(payload)).hexdigest(),
                      'sequence':1,'completed':True},
            'first_observation_sequence':None,
            'source_before':{'fixture.py':{'bytes':1,'sha256':'a'*64}},
            'source_after':{'fixture.py':{'bytes':1,'sha256':'a'*64}},
            'terminal':{'observed':True,'returncode':0,'fixture_only':True}}

def reseal(packet):
    packet['freeze']['artifact_sha256']=hashlib.sha256(canonical_payload(packet['artifact'])).hexdigest()

class PredictionTests(unittest.TestCase):
    def call(self,fn,*args,**kwargs):
        try:
            got=fn(*args,**kwargs)
        except NotImplementedError as error:
            self.fail('Prediction feature unavailable: '+fn.__name__+': '+str(error))
        self.assertIsNotNone(got,'Prediction feature unavailable: '+fn.__name__)
        return got

    def by_key(self,artifact):
        return {(row['k'],row['event_id']):row for row in artifact['rows']}

    def test_full72_structure_without_campaign_calculation(self):
        spec=campaign()
        packet=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,packet,spec))
        got=packet['artifact']
        expected={(k,event['id']) for k in (8,12,15,16) for event in spec['events']}
        self.assertEqual(len(expected),72)
        self.assertEqual(set(self.by_key(got)),expected)
        self.assertEqual(len(got['rows']),72)
        self.assertEqual(got['status'],'INCONCLUSIVE_BUDGET')
        for row in got['rows']:
            self.assertEqual(tuple(row['probability']),('0','1'))
            self.assertEqual(row['uncertainty']['uncomputed_full_mass'],'1')
        self.assertFalse(got['numeric_prediction_qualified'])

    def test_unrestricted_normalization_is_one_not_old_mass_or_half(self):
        got=self.call(prediction.predict,synthetic(),limits())
        self.assertEqual(set(self.by_key(got)),{(k,'unrestricted') for k in KS})
        for row in got['rows']:
            self.assertEqual(tuple(row['probability']),(Q(1),Q(1)))

    def test_tiny_zero_budget_retains_full_unknown_mass(self):
        spec=synthetic('tiny_budget',Q(9,64),Q(1,16),Q(0),ks=(8,))
        got=self.call(prediction.predict,spec,limits(max_endpoint_evaluations=0))
        self.assertEqual(got['status'],'INCONCLUSIVE_BUDGET')
        self.assertEqual(tuple(got['rows'][0]['probability']),(Q(0),Q(1)))
        self.assertEqual(got['rows'][0]['uncertainty']['uncomputed_full_mass'],Q(1))
        self.assertFalse(got['numeric_prediction_qualified'])

    def test_exact_fold_endpoint_continuation(self):
        got=self.call(prediction.predict,synthetic('fold',Q(9,64),Q(1,16),ks=(8,)),limits())
        endpoint=got['rows'][0]['endpoint_witnesses']['0']
        for key,want in (('p',Q(1)),('e',Q(3)),('n',Q(1)),('D',Q(0)),('lifetime',Q(0))):
            self.assertEqual(tuple(endpoint[key]),(want,want))
        for key in ('lifetime_cap','physical_cap','combined_cap','integrand'):
            self.assertEqual(tuple(endpoint[key]),(Q(1),Q(1)))

    def test_exact_tie_endpoint_and_fifth_power(self):
        got=self.call(prediction.predict,synthetic('tie',Q(9,64),ks=(8,)),limits())
        endpoint=got['rows'][0]['endpoint_witnesses']['1']
        self.assertEqual(tuple(endpoint['n']),(Q(0),Q(0)))
        self.assertEqual(tuple(endpoint['lifetime']),(Q(9,4),Q(9,4)))
        self.assertEqual(tuple(endpoint['lifetime_cap']),(Q(1,2),Q(1,2)))
        self.assertEqual(tuple(endpoint['combined_cap']),(Q(1,2),Q(1,2)))
        self.assertEqual(tuple(endpoint['integrand']),(Q(1,32),Q(1,32)))
        self.assertGreater(endpoint['p'][0],Q(5,3))
        self.assertLess(endpoint['p'][1],Q(7,4))

    def test_birth_event_is_lower_Z_and_keeps_whole_law_mass(self):
        got=self.call(prediction.predict,synthetic('birth',birth=Q(0),ks=(8,16)),limits())
        keyed=self.by_key(got)
        p8=keyed[(8,'birth')]['probability']
        p16=keyed[(16,'birth')]['probability']
        self.assertGreater(p8[0],Q(1,4))
        self.assertLess(p8[1],Q(1,3))
        self.assertGreater(p16[0],Q(1,16))
        self.assertLess(p16[1],Q(1,8))
        for row in got['rows']:
            cut=row['birth_cutoff']
            self.assertEqual(cut['admitted_region'],'LOWER_Z')
            self.assertGreaterEqual(cut['unknown_strip_mass'],Q(0))
            self.assertEqual(cut['conditional_renormalization'],False)

    def test_endpoint_mass_weights_are_exact_not_equal_panels(self):
        got=self.call(prediction.predict,synthetic('weights',Q(9,64)),limits())
        keyed=self.by_key(got)
        for k,left,right in ((8,Q(1,256),Q(255,256)),
                             (12,Q(1,4096),Q(4095,4096)),
                             (15,Q(1,32768),Q(32767,32768)),
                             (16,Q(1,65536),Q(65535,65536))):
            row=keyed[(k,'weights')]
            self.assertEqual([p['weight'] for p in row['panels']],[left,right])
            self.assertEqual(row['panels'][0]['Z_interval'],(Q(0),Q(1,2)))
            self.assertEqual(row['panels'][1]['Z_interval'],(Q(1,2),Q(1)))

    def test_unresolved_root_cannot_invent_positive_lifetime_floor(self):
        got=self.call(prediction.predict,synthetic('coarse',Q(9,64),ks=(8,)),
                      limits(max_root_bisections=0,max_cap_bisections=0))
        row=got['rows'][0]
        endpoint=row['endpoint_witnesses']['1/2']
        self.assertEqual(endpoint['lifetime_cap'][1],Q(1))
        self.assertEqual(endpoint['combined_cap'][1],Q(1))
        self.assertEqual(row['probability'][1],Q(1))
        self.assertFalse(got['numeric_prediction_qualified'])

    def test_physical_cap_uses_arcsin_metric_not_chart_gap(self):
        got=self.call(prediction.predict,synthetic('metric',radius=Q(1,4),ks=(8,)),limits())
        endpoint=got['rows'][0]['endpoint_witnesses']['1']
        cap=endpoint['physical_cap']
        self.assertGreater(cap[0],Q(1,3))
        self.assertLess(cap[1],Q(2,5))

    def test_observation_input_is_rejected_before_prediction(self):
        for forbidden,value in (('observed_cells',[]),('uf_results',{}),
                                ('uniform_root_certificates',[]),('matching_epsilon','0')):
            with self.subTest(forbidden=forbidden):
                spec=synthetic()
                spec[forbidden]=value
                with self.assertRaises(ValueError):
                    prediction.predict(spec,limits())

    def test_literal_conservative_full72_packet_validates_as_fixture_only(self):
        spec=campaign()
        packet=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,packet,spec))
        self.assertEqual(packet['artifact']['record_class'],'DEVELOPMENT_STRUCTURE_FIXTURE')
        self.assertFalse(packet['artifact']['numeric_prediction_qualified'])

    def test_missing_row_is_rejected_after_valid_baseline(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['artifact']['rows'].pop()
        reseal(bad)
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

    def test_duplicate_row_cannot_hide_missing_row(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['artifact']['rows'][-1]=deepcopy(bad['artifact']['rows'][0])
        reseal(bad)
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

    def test_missing_prediction_freeze_is_rejected(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['freeze']=None
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

    def test_freeze_must_bind_exact_artifact_bytes(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['freeze']['artifact_sha256']='0'*64
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

    def test_prediction_freeze_must_precede_first_observation(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['first_observation_sequence']=1
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

    def test_source_drift_withdraws_prediction_packet(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['source_after']['fixture.py']={'error':'OSError: owned fixture'}
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

    def test_unknown_mass_cannot_be_relabelled_certified(self):
        spec=campaign()
        good=literal_packet(spec)
        self.assertTrue(self.call(prediction.validate_prediction,good,spec))
        bad=deepcopy(good)
        bad['artifact']['status']='CERTIFIED'
        bad['artifact']['numeric_prediction_qualified']=True
        reseal(bad)
        self.assertFalse(self.call(prediction.validate_prediction,bad,spec))

if __name__=='__main__':
    unittest.main()
