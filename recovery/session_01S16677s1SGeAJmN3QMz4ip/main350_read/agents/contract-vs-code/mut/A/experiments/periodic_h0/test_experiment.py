"""Controls derived from literal landscapes and independent Fourier identities."""
import importlib.util
import math
from fractions import Fraction
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.find_spec('experiment')
if spec:
    import experiment as ex
else:
    ex=None

class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ex, 'Experiment implementation is not yet present')

    def test_unequal_peaks_obey_elder_and_keep_longest_finite(self):
        f=np.tile([5.,1.,4.,2.,3.,0.],(3,1))
        for method in (ex.oracle_h0,ex.gudhi_h0):
            result=method(f)
            self.assertEqual(sorted(result['intervals']),[[3.,2.],[4.,1.]])
            self.assertEqual(result['essential'],[5.])
            self.assertGreater(result['zero_count'],0)

    def test_periodic_gluing_changes_barcode(self):
        f=np.tile([5.,0.,1.,4.],(3,1))
        self.assertEqual(ex.gudhi_h0(f)['intervals'],[])
        self.assertEqual(ex.oracle_h0(f)['intervals'],[])
        self.assertEqual(ex.oracle_h0(f,periodic=False)['intervals'],[[4.,0.]])

    def test_oracle_matches_library_on_ties_and_random_small_fields(self):
        rng=np.random.default_rng(909)
        for shape in ((3,4),(5,6),(7,8)):
            for tied in (False,True):
                f=rng.normal(size=shape)
                if tied:f=np.round(f)
                a,b=ex.oracle_h0(f),ex.gudhi_h0(f)
                np.testing.assert_allclose(sorted(a['intervals']),sorted(b['intervals']),atol=1e-12)
                self.assertEqual(a['essential'],b['essential'])

    def test_shift_and_amplitude_transform_actual_bars(self):
        f=np.tile([5.,1.,4.,2.,3.,0.],(3,1))
        lengths=lambda x:sorted(b-d for b,d in ex.gudhi_h0(x)['intervals'])
        self.assertEqual(lengths(f+17),[1.,3.])
        self.assertEqual(lengths(2*f),[2.,6.])

    def test_invalid_grid_and_aliasing_rejected(self):
        for f in ([],[[1,2],[3]],[[1,float('nan')],[2,3]]):
            with self.assertRaises(ValueError):ex.oracle_h0(f)
        with self.assertRaises(ValueError):ex.field_grid(ex.mode_bank(1,12),24,12)

    def test_generator_matches_one_real_mode_and_nested_grids(self):
        bank={(0,0):(0.,0.),(1,0):(1.,0.)}
        f=ex.field_grid(bank,16,1)
        den=sum(math.exp(-2*math.pi**2*k*k/24**2) for k in range(-64,65))**2
        amplitude=math.sqrt(2*math.exp(-2*math.pi**2/24**2)/den)
        expected=amplitude*np.cos(2*np.pi*np.arange(16)/16)
        np.testing.assert_allclose(f[:,0],expected,atol=1e-14)
        bank=ex.mode_bank(51,24)
        np.testing.assert_allclose(ex.field_grid(bank,64,24),ex.field_grid(bank,128,24)[::2,::2],atol=2e-14)
        self.assertEqual(bank,ex.mode_bank(51,24))

    def test_covariance_against_independent_spatial_kernel(self):
        for lag in ((0,0),(1,0),(1,1),(3,0)):
            expected=math.exp(-sum(t*t for t in lag)/2)
            self.assertAlmostEqual(ex.reference_covariance(24,lag),expected,places=8)

    def test_half_open_bins_replicates_and_volume(self):
        counts=ex.bin_counts([.1,.2,.3,.4],[.1,.2,.4])
        self.assertEqual(counts,[1,2])
        r=ex.summarize_counts([[1,2],[0,0]],[.1,.2,.4],side=2)
        self.assertEqual(r[0]['mean_mass'],.125)
        self.assertAlmostEqual(r[0]['intensity'],1.25)
        self.assertAlmostEqual(r[0]['se_mass'],.125)
        self.assertNotEqual(r[0]['mean_mass'],.5) # omitted volume
        for bad in ([0,.2,.4],[.2,.1,.4],[.1,float('nan')]):
            with self.assertRaises(ValueError):ex.bin_counts([],bad)

    def test_exact_bin_shape_and_amplitude_coefficient(self):
        self.assertEqual(ex.rational_shape(Fraction(1,2),Fraction(3,2)),Fraction(3))
        base=ex.predicted_bin_mass(.001,.008,coefficient=1)
        scaled=ex.predicted_bin_mass(.008,.064,coefficient=.25)
        self.assertAlmostEqual(base,scaled,places=14) # A=8 -> c_A=c/4
        self.assertNotAlmostEqual(base,ex.predicted_bin_mass(.008,.064,coefficient=1))

if __name__=='__main__':unittest.main()
