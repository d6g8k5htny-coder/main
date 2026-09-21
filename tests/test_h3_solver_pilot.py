"""Author-side regressions; they do not provide nonauthor review credit."""
from fractions import Fraction as F
from pathlib import Path
import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]/'engine/solver_pilots/h3_20260921'
sys.path.insert(0,str(HERE))
import h3_scalar_arithmetic as a
import h3_scalar_certificate as c
import h3_solver as r


class ExactArithmetic(unittest.TestCase):
    def test_float_rejected(self):
        with self.assertRaises(TypeError):a.I(.1)
    def test_bool_rejected(self):
        with self.assertRaises(TypeError):a.I(True)
    def test_empty_interval(self):
        with self.assertRaises(ValueError):a.I(2,1)
    def test_division_by_zero_interval(self):
        with self.assertRaises(ZeroDivisionError):a.I(1)/a.I(-1,1)
    def test_negative_interval_division(self):
        x=a.I(1,2)/a.I(-4,-2)
        self.assertLessEqual(x.lo,-1);self.assertGreaterEqual(x.hi,F(-1,4))
    def test_square_crosses_zero(self):
        x=a.I(-2,3)**2;self.assertEqual(x.lo,0);self.assertEqual(x.hi,9)
    def test_odd_power_signed(self):
        x=a.I(-2,3)**3;self.assertEqual((x.lo,x.hi),(-8,27))
    def test_sqrt_exact(self):
        x=a.sqrt(a.I(4));self.assertEqual((x.lo,x.hi),(2,2))
    def test_sqrt_irrational_bounds(self):
        x=a.sqrt(a.I(2));self.assertLessEqual(x.lo*x.lo,2);self.assertGreaterEqual(x.hi*x.hi,2)
    def test_sqrt_negative_refusal(self):
        with self.assertRaises(ValueError):a.sqrt(a.I(-1,2))
    def test_signed_rounding(self):
        x=a.I.bound(F(-1,3),F(1,3));self.assertLessEqual(x.lo,F(-1,3));self.assertGreaterEqual(x.hi,F(1,3))
    def test_pi_coarse_rational_bounds(self):
        p=a.pi();self.assertGreater(p.lo,F(3141592653589793,10**15));self.assertLess(p.hi,F(3141592653589794,10**15))
    def test_exp_zero(self):
        e=a.exp_negative_point(0);self.assertEqual((e.lo,e.hi),(1,1))
    def test_exp_one_known_rational_bracket(self):
        e=a.exp_negative_point(1);self.assertGreater(e.lo,F('0.3678794411714423'));self.assertLess(e.hi,F('0.3678794411714424'))
    def test_exp_out_of_domain(self):
        with self.assertRaises(ValueError):a.exp_negative_point(51)
    def test_cdf_zero(self):
        v=a.cdf_point(0);self.assertEqual((v.lo,v.hi),(F(1,2),F(1,2)))
    def test_cdf_symmetry(self):
        x=a.cdf_point(F(3,2));y=a.cdf_point(F(-3,2))
        self.assertLessEqual((x+y).lo,1);self.assertGreaterEqual((x+y).hi,1)
    def test_cdf_bracket(self):
        x=a.cdf_point(1);self.assertGreater(x.lo,F('0.84134474606854'));self.assertLess(x.hi,F('0.84134474606855'))
    def test_cdf_short_series_rejected(self):
        with self.assertRaises(ValueError):a.cdf_point(2,terms=5)
    def test_cdf_unbounded_domain_rejected(self):
        with self.assertRaises(ValueError):a.cdf_point(9)
    def test_display_is_outward(self):
        d=a.outward_decimal(a.I(F(-1,3),F(1,3)),6)
        self.assertLessEqual(F(d['lo']),F(-1,3));self.assertGreaterEqual(F(d['hi']),F(1,3))
    def test_hermite_known(self):self.assertEqual(c.hermite(4),[3,0,-6,0,1])
    def test_moment_tails_retained(self):
        m,report=c.moments();self.assertGreater(F(report['raw_derivative_errors']['8']),0)
        self.assertLess(m[8].lo,105);self.assertGreater(m[8].hi,105)
    def test_gram_energy_budget(self):
        m,_=c.moments();Q,report=c.gram_and_energy(m,F(1,20))
        self.assertEqual(Q,F(1266466,160083));self.assertLess(Q,8)
        self.assertTrue(all(F(x['lo'])>0 for x in report['shifted_gram_pivots']))


class ScientificDirection(unittest.TestCase):
    def test_wide_crossing_not_counterexample(self):
        self.assertEqual(r.classify_enclosure('1','4','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'),'INCONCLUSIVE_REFINE_ENCLOSURE')
    def test_upper_below_floor_is_conflict(self):
        self.assertIn('CONTRADICTION',r.classify_enclosure('1','1.6','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'))
    def test_lower_above_ceiling_is_conflict(self):
        self.assertIn('CONTRADICTION',r.classify_enclosure('3.5','4','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'))
    def test_lower_formula_cannot_refute_true_quantity(self):
        self.assertEqual(r.classify_enclosure('0','1','1.7','3.49','SUFFICIENT_LOWER_BOUND'),'WRONG_QUANTITY_FOR_REFUTATION')
    def test_equal_endpoint_not_strict_violation(self):
        self.assertEqual(r.classify_enclosure('1','1.7','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'),'INCONCLUSIVE_REFINE_ENCLOSURE')
    def test_compatible_point_not_universal_proof(self):
        self.assertIn('DOMAIN_ONLY',r.classify_enclosure('2','3','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'))
    def test_no_float_parameter_in_proof(self):
        with self.assertRaises(ValueError):c.prove(epsilon=.206)
    def test_two_axial_tails_required(self):
        with self.assertRaises(ValueError):c.prove(axial_tails=F(1))
    def test_two_difference_tails_required(self):
        with self.assertRaises(ValueError):c.prove(difference_tails=F(1))
    def test_no_understated_beta(self):
        with self.assertRaises(ValueError):c.prove(beta_scale=F(1,8))
    def test_no_understated_variance_loss(self):
        with self.assertRaises(ValueError):c.prove(variance_loss=F(1,8))
    def test_radius_scope_refusal(self):
        with self.assertRaises(ValueError):c.prove(radius=F(1,10))
    def test_zero_radius_not_endpoint_claim(self):
        with self.assertRaises(ValueError):c.prove(radius=F(0))
    def test_invalid_parameter_refusal(self):
        with self.assertRaises(ValueError):c.prove(epsilon=F(1))
    def test_search_is_diagnostic(self):
        x=r.search();self.assertEqual(x['evaluated'],30351)
        self.assertEqual(x['status'],'NONCERTIFYING_PARAMETER_SEARCH')
        self.assertEqual((x['epsilon'],x['gap_half']),('103/500','9/100'))
    def test_malformed_interval_refused(self):
        with self.assertRaises(ValueError):r.classify_enclosure('2','1','0','3','TARGET_QUANTITY_ENCLOSURE')


class Provenance(unittest.TestCase):
    def sample_context(self):return json.loads((HERE/'CONTEXT.json').read_text())
    def test_context_denial_before_cache(self):
        x=self.sample_context();x['source_usable_for_candidate']=False
        with self.assertRaises(ValueError):r.recipe({'archive_sha256':r.ARCHIVE_SHA},x,{})
    def test_context_new_hash_new_key(self):
        x=self.sample_context();y=dict(x,observation='new review')
        self.assertNotEqual(c.digest(r.recipe({'archive_sha256':r.ARCHIVE_SHA},x,{})),c.digest(r.recipe({'archive_sha256':r.ARCHIVE_SHA},y,{})))
    def test_context_source_mismatch(self):
        with self.assertRaises(ValueError):r.recipe({'archive_sha256':'0'*64},self.sample_context(),{})
    def test_independence_cannot_be_imported(self):
        x=self.sample_context();x['organizational_independence_credit']=1
        with self.assertRaises(ValueError):r.recipe({'archive_sha256':r.ARCHIVE_SHA},x,{})
    def test_local_cache_hit_no_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            x,tag,_=r.cached({'input':1},d,lambda:{'answer':2})
            y,tag2,_=r.cached({'input':1},d,lambda:self.fail('must hit'))
            self.assertEqual(x,y);self.assertIn('NOT_NEW_EVIDENCE',tag2)
    def test_cache_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,key=r.cached({'input':1},d,lambda:{'answer':2})
            p=Path(d)/(key+'.json');v=json.loads(p.read_text());v['payload']['answer']=3;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):r.cached({'input':1},d,lambda:{'answer':2})
    def test_cold_cache_disagreement_detected(self):
        with tempfile.TemporaryDirectory() as d:
            r.cached({'input':1},d,lambda:{'answer':2})
            with self.assertRaises(ValueError):r.cached({'input':1},d,lambda:{'answer':3},cold=True)
    def test_changed_inputs_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            r.cached({'input':1},d,lambda:2)
            y,tag,_=r.cached({'input':2},d,lambda:3)
            self.assertEqual((y,tag),(3,'MISS_COMPUTED'))
    def test_dependency_propagation(self):
        self.assertEqual(r.affected_nodes(['midpoint']),['coefficient','determinant','midpoint'])
    def test_unknown_lineage_node_rejected(self):
        with self.assertRaises(ValueError):r.affected_nodes(['fake'])
    def test_archive_mutation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip';p.write_bytes(b'not the source')
            with self.assertRaises(ValueError):r.source_check(p)


class ExactCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.result=c.prove()
    def test_new_floor(self):
        v=self.result['values']['derived_lower_coefficient']
        self.assertGreater(F(v['lo']),F(1747,1000))
        self.assertLess(F(v['hi']),F(7,4))
    def test_scope_is_preserved(self):
        self.assertEqual(self.result['conclusion']['endpoint_at_rmax'],'1747/400000')
        self.assertFalse(self.result['scientific_status_changed'])
        self.assertEqual(self.result['organizational_independence_credit'],0)
    def test_positive_budget_before_product(self):
        for key in ('positive_bracket','p_axial','p_difference'):
            self.assertGreater(F(self.result['values'][key]['lo']),0)
    def test_formula_root_changes_with_upstream_data(self):
        x=copy.deepcopy(self.result);x['parameters']['epsilon']='1/5'
        a=r.formula_lineage(self.result,{'archive_sha256':r.ARCHIVE_SHA});b=r.formula_lineage(x,{'archive_sha256':r.ARCHIVE_SHA})
        self.assertNotEqual(a['root'],b['root']);self.assertEqual(a['spreadsheet_cells_used'],[])
    def test_unattainable_target_rejected(self):
        with self.assertRaises(ValueError):c.prove(target=F(7,4))
    def test_original_parameters_still_clear_original_target(self):
        x=c.prove(epsilon=F(1,5),gap_half=F(3,25),target=F(17,10))
        self.assertGreater(F(x['values']['derived_lower_coefficient']['lo']),F(17,10))


if __name__=='__main__':unittest.main()
