"""Independent deterministic controls; intentional mutations stay in this harness."""
import inspect
import json
import math
from fractions import Fraction
import numpy as np
import experiment as ex

LAGS=((0,0),(2,0),(2,2),(8,0))

def basis_covariance(*,decay=1.,variance_factor=1.):
    """Sum responses to independent unit-normal basis vectors of the actual FFT."""
    covariance=np.zeros(len(LAGS))
    for mode in ex.mode_bank(0,12):
        for component in range(1 if mode==(0,0) else 2):
            bank={(0,0):(0.,0.)}
            bank[mode]=(1.,0.) if component==0 else (0.,1.)
            f=ex.field_grid(bank,64,12,decay=decay,variance_factor=variance_factor)
            covariance+=f[0,0]*np.array([f[x,y] for x,y in LAGS])
    return covariance

def run_controls():
    reference=np.array([ex.reference_covariance(12,(x*24/64,y*24/64)) for x,y in LAGS])
    actual=basis_covariance()
    wrong_spectrum=basis_covariance(decay=4.,variance_factor=4.)
    wrong_variance=basis_covariance(variance_factor=.5)
    boundary=np.tile([5.,0.,1.,4.],(3,1))
    peaks=np.tile([5.,1.,4.,2.,3.,0.],(3,1))
    # Execute real branch mutations in an isolated namespace, without source edits.
    elder_source=inspect.getsource(ex.oracle_h0)
    assert '(births[u],-u)<' in elder_source
    ns=dict(ex.__dict__)
    exec(elder_source.replace('(births[u],-u)<','(births[u],-u)>'),ns)
    wrong_elder=ns['oracle_h0'](peaks)
    volume_source=inspect.getsource(ex.summarize_counts)
    assert 'mass=c/side**2' in volume_source
    ns2=dict(ex.__dict__)
    exec(volume_source.replace('mass=c/side**2','mass=c'),ns2)
    wrong_volume=ns2['summarize_counts']([[1,2],[0,0]],[.1,.2,.4],side=2)
    rejected={
        'wrong_spectrum':bool(np.max(abs(wrong_spectrum-reference))>.1),
        'missing_conjugate_variance':bool(np.max(abs(wrong_variance-reference))>.1),
        'missing_periodic_gluing':ex.oracle_h0(boundary,False)['intervals']!=ex.gudhi_h0(boundary)['intervals'],
        'wrong_elder':wrong_elder['intervals']!=ex.gudhi_h0(peaks)['intervals'],
        'missing_volume':wrong_volume[0]['mean_mass']!=.125}
    error=float(np.max(abs(actual-reference)))
    rational=ex.rational_shape(Fraction(1,2),Fraction(3,2))==3
    return {'all_controls_pass':error<1e-12 and all(rejected.values()) and rational,
            'covariance_basis_max_error':error,'finite_covariance_reference':reference.tolist(),
            'finite_covariance_fft_basis':actual.tolist(),
            'wrong_spectrum_covariance':wrong_spectrum.tolist(),
            'missing_variance_covariance':wrong_variance.tolist(),
            'mutants_rejected':rejected,'exact_rational_bin_control':rational,
            'scope':'Finite numerical generator/topology/reporting controls; no continuum or theorem certificate'}

if __name__=='__main__':
    result=run_controls();print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['all_controls_pass'] else 1)
