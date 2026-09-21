"""Exact algebra challenges and arithmetic/source/CLI negative controls."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import os
import subprocess
import sys

import pytest

from research.rn import density_majorant as dm
from research import interval as interval_package
from research.interval import Interval as I, exp, pi, sqrt

ROOT = Path(interval_package.__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def bound(D=1, Q=0, **kwargs):
    return dm.density_moment_majorant(D, Q, repo_root=ROOT, **kwargs)


@pytest.fixture
def arithmetic_only(monkeypatch):
    # Cheap unit cases intentionally isolate arithmetic. Real source checks
    # below and every CLI run use the full fresh source contract.
    monkeypatch.setattr(dm, 'source_binding', lambda root: {'test_only': True})


def gaussian_sixth(mean, variance):
    # Independent binomial expansion against centered Gaussian even moments.
    return sum(F(comb(6, 2*j))*mean**(6-2*j)*variance**j*moment
               for j, moment in enumerate((1, 1, 3, 15)))


def test_sixth_moment_bernstein_identity_and_domination():
    for T in map(F, (8, 9, 100)):
        for a in (F(0), F(1, 4), F(1, 2), F(1)):
            direct = a**3*T**3+15*a*a*T*T*(1-a)+45*a*T*(1-a)**2+15*(1-a)**3
            controls = (F(15), 15*T, 5*T*T, T**3)
            bernstein = sum(comb(3,j)*controls[j]*a**j*(1-a)**(3-j) for j in range(4))
            assert direct == bernstein <= T**3
    witness = dm.algebra_certificate()
    for q in (F(0), F(1, 3), F(99)):
        T = q+8
        gaps = (T**3-15, T**3-15*T, T**3-5*T*T, F(0))
        for coeffs, gap in zip(witness['T_cubed_minus_control_coefficients'], gaps):
            assert sum(c*q**j for j,c in enumerate(coeffs)) == gap >= 0
        assert sum(c*q**j for j,c in enumerate(witness['negative_twice_derivative_after_exp_factor'])) == (q+8)**2*(q+2) > 0


def test_coupled_variance_required_and_degenerate_cases_allowed():
    # One observation Z=z: X=2Z has raw variance 4, explained variance 4,
    # zero conditional variance, and saturates the scalar sixth-moment bound.
    assert gaussian_sixth(F(6), F(0)) == 64*9**3
    assert gaussian_sixth(F(0), F(4)) == 15*64 <= 64*8**3
    # Combining the maximum mean with the maximum residual variance is an
    # invalid regression coupling, and demonstrably violates the sharp cap.
    assert gaussian_sixth(F(6), F(4)) > 64*9**3
    # Sixth-moment claim would fail if applied below its energy threshold.
    assert gaussian_sixth(F(0), F(1)) > 1**3


def test_determinant_absolute_sign_and_entry_order():
    for a,b,c in ((-2,3,4),(5,5,0),(-5,-5,0),(0,0,3)):
        determinant = a*b-c*c
        assert abs(determinant) <= F(a*a+b*b,2)+c*c
    assert 0*0-3*3 < 0  # Dropping abs or swapping xy/yy is not licensed.
    assert abs(2*(-3)-4*4) != abs(2*4-(-3)**2)


def test_full_source_custody_and_explicit_scope():
    result = bound()
    binding = result['source_binding']
    assert binding['checked'] and binding['current_eligibility_checked']
    assert not binding['source_programs_executed']
    assert binding['authenticated_identities']
    assert result['typed_integrand_range'].lo == 0 < result['majorant_interval'].lo
    for key in ('field_certified','wedge_certified','spatial_cover','scientific_status_changed',
                'external_spatial_premises_checked','original_prize_closed'):
        assert result[key] is False
    assert result['authority'] == 'NONE' and result['organizational_independence_credit'] == 0
    assert result['window_length'] == F(1,48000)
    assert result['normalization_floor'] == F('0.0077592917375327855')


def test_direct_formula_inclusion_and_rational_oracle(arithmetic_only):
    result = bound(F(1,16), F(3), jacobian_upper=2)
    independent = (F(2*512,48000)*11**3/dm.Z_LO * exp(I.exact(F(-3,2)),100)
                   / ((2*pi(100))**1 * sqrt(2*pi(100)*I.exact(F(1,16)),100)))
    assert result['majorant_interval'].lo <= independent.lo <= independent.hi <= result['majorant_interval'].hi
    # Entirely rational loose oracle: (2*pi)^(3/2)>14 since pi>3;
    # exp(-3/2)<1/2 since exp(1)>2; sqrt(D)=1/4.
    rational_upper = F(2*512,48000)*11**3/dm.Z_LO * F(1,2)*F(4,14)
    assert result['integrand_upper'] < rational_upper


def test_monotonicity_scaling_and_safe_large_energy_plateau(arithmetic_only):
    values = [bound(1,q)['integrand_upper'] for q in (0,1,10,100,2048)]
    assert all(a>b for a,b in zip(values,values[1:]))
    huge = bound(1,10**100)
    assert huge['mahalanobis_used'] == 2048
    assert huge['integrand_upper'] == values[-1] > 0
    assert huge['integrand_upper'] < F(1,10**400)
    base = bound()
    # Density Jacobian and determinant have distinct, exact scaling laws.
    twice = bound(jacobian_upper=2)['majorant_interval']
    quarter_D = bound(F(1,4))['majorant_interval']
    assert 2*base['majorant_interval'].lo in twice
    assert 2*base['majorant_interval'].hi in quarter_D


@pytest.mark.parametrize('key,value', [
    ('determinant_lower',0),('determinant_lower',-1),('mahalanobis_lower',-1),
    ('jacobian_upper',0),('jacobian_upper',-1),('determinant_lower',True),
    ('mahalanobis_lower',False),('mahalanobis_lower',1.0),
    ('mahalanobis_lower','1'),('determinant_lower',F(1,2**4097)),
    ('mahalanobis_lower',2**4097),('bits',True),('bits',192.0),
    ('bits',127),('bits',1025)])
def test_invalid_or_excessive_inputs_fail_before_source_read(monkeypatch,key,value):
    monkeypatch.setattr(dm,'source_binding',lambda root: pytest.fail('read source before invalid input refusal'))
    args={'determinant_lower':1,'mahalanobis_lower':0,'repo_root':ROOT}
    args[key]=value
    with pytest.raises((ValueError,TypeError)):
        dm.density_moment_majorant(**args)


@pytest.mark.parametrize('key,value', [('HESSIAN_VARIANCE_CAP',3),('MOMENT_FACTOR',511),
                                      ('ENERGY_CEILING',9),('WINDOW_LENGTH',F(1,96000)),
                                      ('RADIUS',F(1,10)),('BIRTH',F(1))])
def test_weakened_proof_constants_refused(arithmetic_only,monkeypatch,key,value):
    monkeypatch.setattr(dm,key,value)
    with pytest.raises(dm.MajorantError):
        bound()


def test_unresolved_sqrt_positivity_is_inconclusive_not_zero(arithmetic_only,monkeypatch):
    monkeypatch.setattr(dm,'sqrt',lambda value,precision: I(0,1))
    with pytest.raises(dm.MajorantError,match='positivity inconclusive'):
        bound(F(1,2**4000),bits=128)


def test_extreme_allowed_determinant_retains_finite_positive_bound(arithmetic_only):
    result=bound(F(1,2**4000),bits=128)
    assert result['majorant_interval'].lo>0 and result['integrand_upper']<2**2010


def test_source_authentication_is_fresh(monkeypatch):
    bound()
    def denied(root):
        raise ValueError('current source eligibility denied')
    monkeypatch.setattr(dm,'authenticated_inputs',denied)
    with pytest.raises(ValueError,match='eligibility denied'):
        bound()


def test_historical_inner_member_exclusion_precedes_body_read(monkeypatch):
    original = dm.read_bounded
    def held(path,limit):
        if str(path).endswith('quarantine/EXCLUSIONS.json'):
            return b'{"exclusions":[{"kind":"archive_member","member_path":"closure_round2/rn_field.py"}]}'
        return original(path,limit)
    monkeypatch.setattr(dm,'read_bounded',held)
    monkeypatch.setattr(dm,'ZipFile',lambda *a,**k: pytest.fail('opened historical archive after member hold'))
    with pytest.raises(ValueError,match='excluded member'):
        dm.source_binding(ROOT)


def test_altered_source_member_pin_is_rejected(monkeypatch):
    contracts=list(dm.SOURCE_CONTRACTS)
    relative,size,digest,count,members=contracts[0]
    changed=list(members);name,nbytes,_=changed[2];changed[2]=(name,nbytes,'0'*64)
    contracts[0]=(relative,size,digest,count,tuple(changed))
    monkeypatch.setattr(dm,'SOURCE_CONTRACTS',tuple(contracts))
    with pytest.raises(dm.MajorantError,match='member hash mismatch'):
        dm.source_binding(ROOT)


@pytest.mark.parametrize('optimized',[False,True])
def test_cli_success_and_insufficient_ceiling_rejection(optimized):
    command=[sys.executable]+(['-O'] if optimized else [])+[str(ROOT/'tools/rn_density_majorant_check.py'),
        '--repo-root',str(ROOT),'--determinant-lower','1','--mahalanobis-lower','0']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)+os.pathsep+str(HERE))
    good=subprocess.run(command+['--ceiling','100'],env=env,text=True,capture_output=True,timeout=20)
    bad=subprocess.run(command+['--ceiling','0'],env=env,text=True,capture_output=True,timeout=20)
    assert good.returncode==0 and 'spatial_premises=UNCHECKED' in good.stdout
    assert bad.returncode!=0 and 'below the proved sufficient upper bound' in bad.stdout
