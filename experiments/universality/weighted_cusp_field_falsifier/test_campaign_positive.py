"""Direct positive developmental orchestration RED controls; no execution here.

These request tiny real field certificates, never a production prediction or
campaign. No mock supplies bars, pairing, root locations or coupling tolerance.
The run body remains explicitly unavailable until Root records actual RED.
"""
import unittest
from fractions import Fraction as Q
import exact_h0
import run_falsifier as run

KS=(8,12,15,16)
EVENT={'id':'count','lifetime_upper':Q(1,4),
       'separation_upper':None,'birth_upper':None}

def development_spec():
    cuts=(Q(0),Q(1,2),Q(1,2)+Q(1,2**24),Q(1))
    cells=[]
    for sign in (-1,1):
        for qi in range(3):
            for zi in range(3):
                cells.append({'Qa':cuts[qi],'Qb':cuts[qi+1],
                              'Za':cuts[zi],'Zb':cuts[zi+1],'epsilon':sign})
    return {'schema_version':1,'mode':'DEVELOPMENT_OBSERVATION_ONLY',
            'ks':list(KS),'events':[dict(EVENT)],'initial_cells':cells,
            'refinement':False,
            'development_contract':{
                'record_class':'BOUNDED_DEVELOPMENT_OBSERVATION',
                'production_qualification':False,
                'purpose':'tiny full-partition actual-field development control'}}

def bounded_limits(fields):
    return {'field_evaluations':fields,'control_cells':32,'tensor_side':8,
            'total_vertices':64*fields,'root_bisections':512,'error_mode':'tight'}

class PositiveDevelopmentTests(unittest.TestCase):
    def call(self,*args):
        try:
            got=run.run(*args)
        except NotImplementedError as error:
            self.fail('Positive development orchestration unavailable: '+str(error))
        self.assertIsNotNone(got,'Positive development orchestration unavailable')
        return got

    def test_real18cell_full_partition_retains_actual_UF_and_costs(self):
        got=self.call(development_spec(),None,bounded_limits(18))
        self.assertEqual(got['mode'],'DEVELOPMENT_OBSERVATION_ONLY')
        self.assertEqual(got['status'],'DEVELOPMENT_OBSERVATIONS_RETAINED')
        self.assertEqual(len(got['rows']),18)
        self.assertEqual(got['costs']['attempted_fields'],18)
        self.assertFalse(got['numeric_prediction_qualified'])
        self.assertFalse(got['production_qualification'])
        self.assertFalse(got['full_declared_campaign'])
        self.assertTrue(got['aggregate']['partition_verified'])
        resolved=[]
        field_cost=vertex_cost=root_cost=0
        for row in got['rows']:
            certificate=row['certificate']
            self.assertIsNotNone(certificate)
            costs=certificate['costs']
            self.assertEqual(costs['attempted_fields'],1)
            field_cost+=costs['attempted_fields']
            vertex_cost+=costs['sampled_vertices']
            root_cost+=costs['root_bisections']
            if certificate['status']=='RESOLVED':
                resolved.append(certificate)
                values=certificate['grid']['values']
                self.assertEqual(len(values),64)
                barcode=certificate['observation']['barcode']
                self.assertEqual(barcode,exact_h0.compute(values,8))
                self.assertTrue(exact_h0.verify_by_connectivity(values,8,barcode))
                self.assertTrue(certificate['coupling']['verified'])
        # The two strict tiny interior cells, one per sign, require actual
        # independently selected field data rather than universal unknown rows.
        self.assertGreaterEqual(len(resolved),2)
        self.assertEqual(field_cost,got['costs']['attempted_fields'])
        self.assertEqual(vertex_cost,got['costs']['sampled_vertices'])
        self.assertEqual(root_cost,got['costs']['root_bisections'])
        self.assertGreaterEqual(vertex_cost,128)
        self.assertLessEqual(vertex_cost,18*64)
        for k in KS:
            self.assertEqual(got['aggregate']['total_mass'][k],Q(1))

    def test_budget_keeps_all_unattempted_positive_mass(self):
        got=self.call(development_spec(),None,bounded_limits(1))
        self.assertEqual(got['status'],'INCONCLUSIVE_BUDGET')
        self.assertEqual(got['costs']['attempted_fields'],1)
        self.assertEqual(len(got['rows']),18)
        self.assertEqual(sum(row['certificate'] is not None for row in got['rows']),1)
        self.assertTrue(got['aggregate']['partition_verified'])
        # First declared cell touches Q=0 and Z=0. It cannot have a uniform
        # three-simple-root certificate, and all17 unattempted cells stay unknown.
        self.assertNotEqual(got['rows'][0]['certificate']['status'],'RESOLVED')
        for k in KS:
            self.assertEqual(got['aggregate']['total_mass'][k],Q(1))
            self.assertEqual(got['aggregate']['law_bounds'][k]['count'],(Q(0),Q(1)))
            self.assertEqual(got['aggregate']['unresolved_mass'][k]['count'],Q(1))
        self.assertFalse(got['numeric_prediction_qualified'])
        self.assertFalse(got['production_qualification'])
        self.assertFalse(got['full_declared_campaign'])

if __name__=='__main__':
    unittest.main()
