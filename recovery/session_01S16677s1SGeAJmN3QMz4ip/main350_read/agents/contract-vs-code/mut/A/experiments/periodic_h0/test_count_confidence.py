"""Fixed-sample inference controls; no empirical field sample is generated."""
import importlib
import importlib.util
import itertools
import unittest
from fractions import Fraction as Q


class CountConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('count_confidence') is None:
            raise AssertionError('Fixed-sample confidence implementation is missing')
        cls.m=importlib.import_module('count_confidence')

    def test_family_risk_pays_for_both_sides_and_all_bins(self):
        risk=self.m.risk_upper(1,40000,8,Q(4,25))
        self.assertGreater(risk,Q(5367402046440189,10**18))
        self.assertLess(risk,Q(5367402046440190,10**18))
        self.assertEqual(risk,16*self.m.risk_upper(1,40000,1,Q(4,25))/2)
        self.assertEqual(self.m.risk_upper(24,40000,8,Q(2304,25)),risk)

    def test_missing_rows_keep_original_denominator_and_universal_upper(self):
        rows=[(4,5),None,(2,3),(6,8)]
        self.assertEqual(self.m.mean_interval(1,Q(1,1000),rows,Q(1,2),4),
                         (Q(621,250),Q(2129,250)))
        with self.assertRaises(ValueError):
            self.m.mean_interval(1,Q(1,1000),rows[:3],Q(1,2),4)
        self.assertEqual(self.m.mean_interval(1,Q(0),[None]*4,Q(1,2),4),(Q(0),Q(16)))

    def test_count_weighted_clipping_and_clamps_are_not_omitted(self):
        self.assertEqual(self.m.mean_interval(1,Q(1,10),[(4,5)],Q(1),1),(Q(7,5),Q(38,5)))
        self.assertEqual(self.m.mean_interval(1,Q(0),[(0,16)],Q(1),1),(Q(0),Q(16)))

    def test_invalid_inputs_do_not_become_probability_claims(self):
        for k,n,m,r in [(True,5,1,Q(1)),(1,True,1,Q(1)),(1,0,1,Q(1)),
                         (1,5,0,Q(1)),(1,5,True,Q(1)),(1,5,1,0.1),
                         (1,5,1,Q(0)),(1,5,1,Q(-1))]:
            with self.assertRaises(ValueError):self.m.risk_upper(k,n,m,r)
        for rows in [[],[(False,2)],[(3,2)],[(0,17)],[(0.0,2)],[(0,2,3)]]:
            with self.assertRaises(ValueError):self.m.mean_interval(1,Q(0),rows,Q(1),1)
        for p in [Q(-1),Q(2),0.01]:
            with self.assertRaises(ValueError):self.m.mean_interval(1,p,[(0,1)],Q(1),1)

    def test_large_exponents_stay_outward_and_vacuous_risk_is_clipped(self):
        self.assertEqual(self.m.risk_upper(1,1,8,Q(1,100)),Q(1))
        self.assertGreater(self.m.risk_upper(1,10**20,8,Q(1)),Q(0))
        self.assertLess(self.m.risk_upper(1,10**20,8,Q(1)),Q(1,10**30))

    def test_exact_iid_finite_law_with_adaptive_grid_failures(self):
        # Latent fixed count is 16*Bernoulli(1/2), mean8. Processing may
        # depend on the whole sequence. Missing rows cannot damage coverage.
        n=8;r=Q(6);covered=0;bad=0
        for bits in itertools.product((0,1),repeat=n):
            rows=[None if (sum(bits)+i)%3==0 else (16*x,16*x) for i,x in enumerate(bits)]
            lo,hi=self.m.mean_interval(1,Q(0),rows,r,n)
            covered+=lo<=8<=hi
            bad+=not (lo<=8<=hi)
        self.assertEqual(covered+bad,256)
        self.assertLessEqual(Q(bad,256),self.m.risk_upper(1,n,1,r))

    def test_dependence_and_success_only_selection_break_the_premises(self):
        # Repeating one random bit eight times has miss probability1,
        # larger than the IID budget. The API cannot detect that dependence.
        n=8;r=Q(6)
        for bit in (0,1):
            lo,hi=self.m.mean_interval(1,Q(0),[(16*bit,16*bit)]*n,r,n)
            self.assertFalse(lo<=8<=hi)
        self.assertLess(self.m.risk_upper(1,n,1,r),Q(1))
        # Keeping only zero-count successes can report [0,6] although mu=8.
        lo,hi=self.m.mean_interval(1,Q(0),[(0,0)]*n,r,n)
        self.assertLess(hi,Q(8))


if __name__=='__main__':unittest.main()
