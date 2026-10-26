"""Eight direct-entry RED controls; source-only, never executed by this author.

No real campaign or prediction is run here. The run() calls contain explicitly
invalid production inputs and must be rejected before any observation.
Refinement receives a pure state envelope: leaves,costs,limits. This extends
the planned row-list input contract without changing the existing stub signature.
"""
import unittest
from copy import deepcopy
from fractions import Fraction as Q
import field
import field_certificate as fc
import run_falsifier as run
import exact_h0

A=Q(1,65536)
EVENTS=[{'id':'count','lifetime_upper':Q(1,4),'separation_upper':None,'birth_upper':None}]

def cell(qa=Q(0),qb=Q(1),za=Q(0),zb=Q(1),epsilon=-1):
    return dict(Qa=qa,Qb=qb,Za=za,Zb=zb,epsilon=epsilon)

def leaf(name,c,bounds=(0,1),qterm=Q(1,4),zterm=Q(1,2)):
    return {'cell_id':name,'cell':c,'event_bounds':{'count':bounds},
            'refinement_terms':{'Q':qterm,'Z':zterm}}

def state(leaves,attempted=0,failed=0):
    return {'leaves':leaves,'costs':{'attempted_fields':attempted,
            'failed_fields':failed,'sampled_vertices':64*attempted},
            'limits':{'field_evaluations':4096,'control_cells':4096,
                      'total_vertices':16777216}}

def invalid_production_spec():
    return {'schema_version':1,'ks':[8],'events':deepcopy(EVENTS),'row_count':1,
            'phase':'PRODUCTION_INPUT_REJECTION_CONTROL'}

class CampaignTests(unittest.TestCase):
    def call(self,fn,*args,**kwargs):
        try:
            got=fn(*args,**kwargs)
        except NotImplementedError as error:
            self.fail('Campaign feature unavailable: '+fn.__name__+': '+str(error))
        self.assertIsNotNone(got,'Campaign feature unavailable: '+fn.__name__)
        return got

    def test_realization_descriptor_has_actual_constants_and_full_guard_coverage(self):
        got=self.call(field.realization_descriptor)
        want={'dimension':2,'period':Q(1),'weights':(Q(1,3),Q(1,9)),
              'b':Q(4,9),'R':Q(1,8),'r1':Q(1,32),'A':A,
              'u0':3*A**2,'v0':2*A**3,'floor_scaled':Q(-16)}
        for key,value in want.items():
            self.assertEqual(got['constants'][key],value)
        self.assertEqual(set(got['region_coverage']),
                         {'core','annulus','collar','torus_exterior'})
        guards=got['quantitative_guards']
        self.assertEqual(guards['step_derivative_upper'],Q(8))
        self.assertEqual(guards['B1_upper'],Q(35,96))
        self.assertEqual(guards['D1_upper'],Q(67,3))
        self.assertGreaterEqual(guards['g_lower'],Q(1,2**19))
        self.assertLess(guards['gradient_perturbation_upper'],guards['g_lower']/2)
        self.assertLess(guards['height_perturbation_upper'],Q(1,2**28))
        self.assertTrue(got['whole_field_defined'])

    def test_certify_cell_directly_produces_real_UF_and_independent_BFS(self):
        width=Q(1,2**24)
        c=cell(Q(1,2),Q(1,2)+width,Q(1,2),Q(1,2)+width,1)
        limits={'field_evaluations':1,'tensor_side':8,'total_vertices':64,
                'root_bisections':512,'error_mode':'tight'}
        got=self.call(fc.certify_cell,c,limits)
        self.assertEqual(got['status'],'RESOLVED')
        values=got['grid']['values']
        self.assertEqual(len(values),64)
        observed=got['observation']['barcode']
        self.assertEqual(observed,exact_h0.compute(values,8))
        self.assertTrue(exact_h0.verify_by_connectivity(values,8,observed))
        self.assertTrue(got['observation']['independent_bfs_verified'])
        self.assertEqual(len(observed['intervals']),1)
        self.assertEqual(len(observed['essential']),1)
        self.assertEqual(got['coupling']['saddle_root_index'],1)
        self.assertTrue(got['coupling']['verified'])

    def test_refinement_selects_largest_unknown_across_four_exact_law_masses(self):
        leaves=[]
        for sign,label in ((-1,'minus'),(1,'plus')):
            leaves.append(leaf('low_q_'+label,cell(qb=Q(1,2),epsilon=sign)))
            leaves.append(leaf('low_z_'+label,cell(qa=Q(1,2),zb=Q(1,2),epsilon=sign)))
            leaves.append(leaf('resolved_'+label,cell(qa=Q(1,2),za=Q(1,2),epsilon=sign),(0,0)))
        got=self.call(run.choose_refinement,state(leaves),EVENTS)
        # Low-Q mass is1/64 for every law. Upper-Q/lower-Z mass is at
        # most31/16384 (k8); resolved larger cells must not outrank it.
        self.assertEqual(got['cell_id'],'low_q_minus')
        self.assertEqual(got['priority_mass'],Q(1,64))
        self.assertEqual(got['axis'],'Z')
        self.assertEqual(len(got['children']),2)
        left,right=got['children']
        self.assertEqual((left['Za'],left['Zb']),(Q(0),Q(1,2)))
        self.assertEqual((right['Za'],right['Zb']),(Q(1,2),Q(1)))
        self.assertEqual(left['epsilon'],-1)
        self.assertEqual(right['epsilon'],-1)

    def test_refinement_ties_are_order_independent_and_split_dyadic_Q(self):
        leaves=[leaf('b',cell(epsilon=1),qterm=Q(1),zterm=Q(1)),
                leaf('a',cell(epsilon=-1),qterm=Q(1),zterm=Q(1))]
        got=self.call(run.choose_refinement,state(leaves),EVENTS)
        reverse=self.call(run.choose_refinement,state(list(reversed(leaves))),EVENTS)
        self.assertEqual(got,reverse)
        self.assertEqual(got['cell_id'],'a')
        self.assertEqual(got['axis'],'Q')
        left,right=got['children']
        self.assertEqual((left['Qa'],left['Qb']),(Q(0),Q(1,2)))
        self.assertEqual((right['Qa'],right['Qb']),(Q(1,2),Q(1)))

    def test_refinement_cap_counts_failed_attempts_and_schedules_nothing(self):
        leaves=[leaf('a',cell(epsilon=-1)),leaf('b',cell(epsilon=1))]
        original=state(leaves,attempted=4096,failed=17)
        got=self.call(run.choose_refinement,deepcopy(original),EVENTS)
        self.assertEqual(got['status'],'INCONCLUSIVE_BUDGET')
        self.assertIsNone(got['selected'])
        self.assertEqual(got['evaluations_scheduled'],0)
        self.assertEqual(got['leaves'],original['leaves'])
        self.assertEqual(got['costs']['attempted_fields'],4096)
        self.assertEqual(got['costs']['failed_fields'],17)

    def test_run_rejects_absent_prediction_freeze_before_observation(self):
        with self.assertRaisesRegex(ValueError,'prediction|freeze'):
            run.run(invalid_production_spec(),None,{'field_evaluations':1})

    def test_run_rejects_prediction_freeze_after_first_observation(self):
        packet={'artifact':{'record_class':'PREDICTION_ARTIFACT'},
                'freeze':{'completed':True,'sequence':2},
                'first_observation_sequence':1}
        with self.assertRaisesRegex(ValueError,'premature|precede|freeze.*order'):
            run.run(invalid_production_spec(),packet,{'field_evaluations':1})

    def test_run_rejects_structure_fixture_as_production_prediction(self):
        packet={'artifact':{'record_class':'DEVELOPMENT_STRUCTURE_FIXTURE',
                            'numeric_prediction_qualified':False},
                'freeze':{'completed':True,'sequence':1},
                'first_observation_sequence':None}
        with self.assertRaisesRegex(ValueError,'fixture|production'):
            run.run(invalid_production_spec(),packet,{'field_evaluations':1})

if __name__=='__main__':
    unittest.main()
