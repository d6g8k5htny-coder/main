"""Adversarial exact controls for event loss and grid-count observables."""
import unittest
from fractions import Fraction as Q
try:
    import finite_count_loss as cl
except ModuleNotFoundError:
    cl = None


class FiniteCountLossTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cl, 'Finite-polynomial count-loss bridge is missing')

    def test_event_loss_requires_count_cap_not_probability_alone(self):
        # Two-outcome model, bad mass1/8: good counts7; bad sampler16,target0.
        # Means65/8 and49/8 differ by2, not by1/8. Reversing gives the upper case.
        lo, hi = cl.expectation_interval(1,Q(1,8),Q(65,8),Q(65,8))
        self.assertEqual(lo,Q(49,8))
        self.assertGreater(lo,Q(0))
        self.assertGreater(Q(65,8)-Q(1,8),Q(49,8))
        lo, hi = cl.expectation_interval(1,Q(1,8),Q(49,8),Q(49,8))
        self.assertEqual(hi,Q(65,8))
        self.assertEqual(cl.expectation_interval(1,Q(1),Q(7),Q(9)),(Q(0),Q(16)))
        self.assertEqual(cl.expectation_interval(1,Q(0),Q(7),Q(9)),(Q(7),Q(9)))

    def test_strict_diagonal_and_half_open_endpoints(self):
        # Total diagram radius1; lifetime radius2; [3,10) contracts to[5,8).
        lengths=[Q(1),Q(5),Q(8),Q(12)]
        self.assertEqual(cl.grid_observables(1,Q(3),Q(10),Q(1),Q(0),lengths),(1,3))
        # At a=2*radius a target interval can be matched to the diagonal.
        self.assertEqual(cl.grid_observables(1,Q(2),Q(10),Q(1),Q(0),[]),(0,16))
        # Coupling radius must be added, not ignored.
        self.assertEqual(cl.grid_observables(1,Q(3),Q(10),Q(1),Q(1),[]),(0,16))

    def test_empty_contraction_and_unresolved_grid_keep_full_cap(self):
        self.assertEqual(cl.grid_observables(1,Q(3),Q(4),Q(1),Q(0),[Q(3)]),(0,1))
        self.assertEqual(cl.unresolved_grid(1),(0,16))
        # Expanded grid count may exceed the polynomial cap; clipping is justified.
        self.assertEqual(cl.grid_observables(1,Q(3),Q(4),Q(1),Q(0),[Q(3)]*20),(0,16))
        # Too many contracted bars contradict the finite-polynomial certificate.
        with self.assertRaises(ValueError):
            cl.grid_observables(1,Q(3),Q(10),Q(1),Q(0),[Q(6)]*17)

    def test_rejects_invalid_domains_and_unjustified_inputs(self):
        for k in (True,0,-1,1.0):
            with self.assertRaises(ValueError): cl.bar_cap(k)
        for args in [(1,Q(-1),Q(0),Q(1)),(1,Q(2),Q(0),Q(1)),
                     (1,0.1,Q(0),Q(1)),(1,Q(0),Q(-1),Q(1)),
                     (1,Q(0),Q(2),Q(1)),(1,Q(0),Q(0),Q(17))]:
            with self.assertRaises(ValueError): cl.expectation_interval(*args)
        for a,b,e,r,ls in [(Q(0),Q(1),Q(0),Q(0),[]),
                           (Q(2),Q(1),Q(0),Q(0),[]),
                           (Q(1),Q(2),Q(-1),Q(0),[]),
                           (Q(1),Q(2),Q(0),Q(-1),[]),
                           (Q(1),Q(2),Q(0),Q(0),[Q(0)]),
                           (Q(1),Q(2),Q(0),Q(0),[1.0])]:
            with self.assertRaises(ValueError): cl.grid_observables(1,a,b,e,r,ls)

    def test_finite_coupled_diagrams_and_bad_events_exactly(self):
        # Good match endpoint error1/4, hence lifetime displacement<=1/2.
        # A diagonal bar at1/2 is allowed, while bins above the threshold resolve.
        outcomes=[(Q(3,4),[Q(3),Q(5)],[Q(7,2),Q(9,2)],False),
                  (Q(1,4),[Q(6)]*16,[],True)]
        for a,b in [(Q(2),Q(4)),(Q(4),Q(7)),(Q(1,2),Q(7))]:
            ml=mu=actual=Q(0)
            for mass,source,target,bad in outcomes:
                l,u=cl.grid_observables(1,a,b,Q(0),Q(1,4),source)
                ml+=mass*l;mu+=mass*u
                actual+=mass*sum(a<=x<b for x in target)
            lo,hi=cl.expectation_interval(1,Q(1,4),ml,mu)
            self.assertLessEqual(lo,actual);self.assertLessEqual(actual,hi)


if __name__=='__main__': unittest.main()
