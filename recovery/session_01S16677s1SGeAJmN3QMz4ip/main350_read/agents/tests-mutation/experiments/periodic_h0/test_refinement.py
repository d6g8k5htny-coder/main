"""Topology, coupling and boundary controls for the resolution comparison."""
import importlib.util
import copy
import json
from pathlib import Path
import unittest
from fractions import Fraction
import numpy as np
import experiment as ex

spec=importlib.util.find_spec('refinement')
if spec:import refinement as ref
else:ref=None

class RefinementTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(ref,'Refinement implementation missing')

    def test_adapted_minimum_diagonal_preserves_cubical_h0(self):
        rng=np.random.default_rng(321)
        for shape in ((3,4),(5,6),(7,7)):
            for tied in (False,True):
                f=rng.normal(size=shape)
                if tied:f=np.round(f)
                self.assertEqual(ref.triangulated_h0(f,'minimum'),ex.gudhi_h0(f))

    def test_fixed_diagonal_can_change_a_saddle_pair(self):
        f=np.zeros((5,5));f[1,1]=5;f[2,2]=4
        cub=ex.gudhi_h0(f)
        self.assertEqual(cub['intervals'],[[4.,0.]])
        self.assertEqual(ref.triangulated_h0(f,'plus')['intervals'],[])
        self.assertEqual(ref.triangulated_h0(f,'minus')['intervals'],[[4.,0.]])
        self.assertEqual(ref.oracle_triangulated_h0(f,'plus'),ref.triangulated_h0(f,'plus'))
        self.assertEqual(ref.oracle_triangulated_h0(f,'minus'),ref.triangulated_h0(f,'minus'))

    def test_fixed_diagonals_match_independent_union_find(self):
        rng=np.random.default_rng(914)
        for sign in ('plus','minus'):
            for n in (4,5,8):
                for tied in (False,True):
                    f=rng.normal(size=(n,n))
                    if tied:f=np.round(f)
                    self.assertEqual(ref.triangulated_h0(f,sign),ref.oracle_triangulated_h0(f,sign))
        with self.assertRaises(ValueError):ref.triangulated_h0(np.zeros((3,3)),'unknown')

    def test_exact_half_open_sandwich_and_diagonal_obstruction(self):
        q=[Fraction(1,10),Fraction(1,5),Fraction(3,10),Fraction(2,5)]
        b=ref.bin_sandwich(q,Fraction(1,5),Fraction(2,5),Fraction(1,40))
        self.assertEqual(b,{'lower':1,'upper':3,'diagonal_obstruction':False})
        b=ref.bin_sandwich(q,Fraction(1,10),Fraction(1,5),Fraction(1,20))
        self.assertIsNone(b['upper']);self.assertTrue(b['diagonal_obstruction'])
        # A true length 2 epsilon may disappear against the diagonal exactly.
        self.assertEqual(ref.bin_sandwich([],Fraction(1,10),Fraction(1,5),Fraction(1,20))['lower'],0)
        with self.assertRaises(ValueError):ref.bin_sandwich(q,Fraction(0),Fraction(1),Fraction(0))

    def test_runner_rejects_invalid_design_and_tampered_observations(self):
        import run_refinement as runner
        c=json.loads(Path(__file__).with_name('refinement_config.json').read_text())
        c.update(seeds=[1,2],grids=[64,128])
        data=runner.execute(c);self.assertTrue(runner.verify(data))
        rounded=copy.deepcopy(data)
        for row in rounded['summary']:
            for b in row['bins']:b['mean_mass']=float(np.nextafter(b['mean_mass'],np.inf))
        self.assertTrue(runner.verify(rounded))
        self.assertFalse(runner.equivalent({'count':1},{'count':1.0}))
        self.assertFalse(runner.equivalent({'value':1.0},{'value':float('nan')}))
        for field in ('counts','essential','duplicate','summary','comparisons','diagnostics'):
            d=copy.deepcopy(data)
            if field=='counts':d['records'][0]['counts'][0]+=1
            elif field=='essential':d['records'][0]['essential']=[]
            elif field=='duplicate':d['records'][1]=copy.deepcopy(d['records'][0])
            elif field=='summary':d['summary'][0]['bins'][0]['mean_mass']+=1
            elif field=='comparisons':d['comparisons'][0]['per_field'][0]['finite_bottleneck']+=1
            else:d['diagnostics'][0]['interpolation_epsilon_float'][0]=0
            with self.subTest(field=field),self.assertRaises(ValueError):runner.verify(d)
        for key,value in [('seeds',[1,1]),('grids',[64,96]),('grids',[48,96]),('methods',['cubical']),('cutoff',12)]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runner.validate(bad)

    def test_single_mode_hessian_majorant_and_quadratic_scaling(self):
        bank={(0,0):(0.,0.),(1,0):(1.,0.)}
        h=ref.hessian_diagnostic(bank,1)
        amplitude=ex.field_grid(bank,64,1)[0,0]
        self.assertAlmostEqual(h,amplitude*(2*np.pi/24)**2,places=14)
        self.assertAlmostEqual(ref.interpolation_diagnostic(h,128)*4,ref.interpolation_diagnostic(h,64),places=14)

if __name__=='__main__':unittest.main()
