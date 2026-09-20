"""Exact falsification controls; no sampling is used to certify a uniform bound."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys

import pytest
from research.parallel.lpw import lpw_modulus as m


HERE=Path(m.__file__).resolve().parent


@pytest.fixture(scope='module')
def report():
    return m.make_report()


def cli(*args, optimized=False):
    return subprocess.run([sys.executable]+(['-O'] if optimized else [])+
                          [str(HERE/'lpw_modulus.py'),*map(str,args)],
                          capture_output=True,text=True,timeout=30)


def test_source_and_claims(report):
    assert len(report['sources'])==6
    assert F(report['linear_replay']['frobenius_coefficient']['hi']) < F('19.072')
    assert F(report['quadratic']['frobenius_coefficient']['hi']) < F('1.804')
    assert F(report['quadratic']['simple_conditional_uniform_eigenfloor'])==F('0.1239999998196')
    assert F(report['quadratic']['computed_loss_budget_ratio'])>1000000


def test_normal_and_optimized_cli_byte_identical(tmp_path):
    outputs=[tmp_path/'normal.json',tmp_path/'optimized.json']
    normal=cli('--output',outputs[0])
    optimized=cli('--output',outputs[1],optimized=True)
    assert normal.returncode==optimized.returncode==0,(normal.stderr,optimized.stderr)
    assert normal.stdout==optimized.stdout
    assert outputs[0].read_bytes()==outputs[1].read_bytes()
    assert cli('--certificate',outputs[0]).returncode==0


@pytest.mark.parametrize('optimized',[False,True])
@pytest.mark.parametrize('flag,value',[('--claim-linear','19'),('--claim-quadratic','1.803')])
def test_actual_cli_rejects_understated_ceiling(flag,value,optimized):
    result=cli(flag,value,optimized=optimized)
    assert result.returncode==1 and 'modulus claim rejected' in result.stderr


def mutate(report,kind):
    x=deepcopy(report)
    if kind=='missing_tail': x['lattice']['one_dimensional_tails'][12]['tail_upper']='0'
    elif kind=='skip_first': x['lattice']['first_omitted']=72
    elif kind=='normalization': x['lattice']['Z1_full']['hi']=x['lattice']['Z1_truncated']['lo']
    elif kind=='phase_drop': x['quadratic']['entry_majorants'][2][2]='0'
    elif kind=='wrong_factor':
        x['quadratic']['entry_majorants']=[[str(F(v)/2) for v in row] for row in x['quadratic']['entry_majorants']]
    elif kind=='wrong_g5': x['quadratic']['derivative_slope_bounds'][5]='1/31'
    elif kind=='independence': x['authority']['organizational_independence_credit']=1
    elif kind=='bool_is_not_zero': x['authority']['organizational_independence_credit']=False
    elif kind=='source': x['sources'][0]['sha256']='0'*64
    else: raise ValueError(kind)
    return x


@pytest.mark.parametrize('kind',['missing_tail','skip_first','normalization','phase_drop','wrong_factor','wrong_g5','independence','bool_is_not_zero','source'])
@pytest.mark.parametrize('optimized',[False,True])
def test_mutated_explicit_certificate_rejected(report,tmp_path,kind,optimized):
    path=tmp_path/(kind+'.json')
    path.write_text(json.dumps(mutate(report,kind)))
    result=cli('--certificate',path,optimized=optimized)
    assert result.returncode==1 and 'reconstruction mismatch' in result.stderr


def test_changed_raw_source_rejected(tmp_path):
    for relative,_,_,_ in m.SOURCES:
        source=m.DEFAULT_REPO/m.PREFIX/relative
        target=tmp_path/m.PREFIX/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    first=tmp_path/m.PREFIX/m.SOURCES[0][0]
    first.write_bytes(first.read_bytes()+b'\n')
    result=cli('--repo',tmp_path)
    assert result.returncode==1 and 'source identity mismatch' in result.stderr


def test_tail_ratio_and_entire_tail_controls():
    h=m.pi(m.PREC)/12;a=h*h/2
    for p in range(13):
        tail,ratio,first=m.moment_tail(h,a,p)
        assert ratio.hi<F(1,2)
        assert tail.lo>first.hi
        # Dropping the two-sided multiplicity cannot even cover first omitted pair.
        assert tail.hi/2<first.lo
        # Exact monotonicity: rho(n+1)/rho(n) has a strictly subunit factor.
        n=71
        assert F(n*(n+2),(n+1)**2)<=1
        ratio_drop=m.exp(-2*a,m.PREC)*m.I(F(n*(n+2),(n+1)**2))**p
        assert ratio_drop.hi<1
    with pytest.raises(ValueError,match='ratio'):
        m.moment_tail(h,a,12,1)


@pytest.mark.parametrize('power',[3,4])
def test_shell_ratio_and_no_tail_omission(power):
    h=m.pi(m.PREC)/12;a=h*h/2
    tail,ratio,first=m.shell_tail(h,a,power)
    assert ratio.hi<1 and tail.lo>first.hi>0
    # The omitted s=71 shell contains at least these four axial sites.
    axis=4*m.exp(-a*F(71*71,2),m.PREC)*(1+h*71)**power
    assert axis.lo>0 and tail.hi>axis.hi


def test_denominator_direction_counterexample():
    result=m.normalize_moment(m.I(5),m.I(0,1),m.I(2),m.I(2,3))
    assert F(5,3) in result and F(3) in result
    # Using upper normalization in the upper quotient fails an admissible point.
    wrong_upper=F(6,3)
    assert F(5,2)>wrong_upper


def test_g5_derivative_slope_falsifier():
    theta=m.I(F(1,10))
    # Exact analytic derivative, independently evaluated away from its removable pole.
    derivative=((3-theta**2)*m.sin(theta,30)-3*theta*m.cos(theta,30))/(2*theta**4)
    ratio=derivative/theta
    assert ratio.lo>F(1,31)  # Refutes a tempting stronger uniform bound.
    assert ratio.hi<F(1,30)


def test_sinc_derivative_slope_falsifier():
    theta=m.I(F(1,10))
    derivative=(theta*m.cos(theta,30)-m.sin(theta,30))/theta**2
    ratio=abs(derivative/theta)
    assert ratio.lo>F(1,4) and ratio.hi<F(1,3)


def test_quadratic_integration_factor_falsifier():
    # For g(theta)=cos(theta), k=1, r=1/5, covariance is cos(r/2)^2.
    # d=1, PB=1 gives correct 2/8=1/4 coefficient of r^2.
    # Halving that coefficient is false, not merely different output bytes.
    r=F(1,5)
    actual=1-m.cos(m.I(r/2),30)**2
    assert actual.lo>r*r/8 and actual.hi<=r*r/4


def test_phase_and_coordinate_parity(report):
    # Fourier derivative cross factor i^degree_i (-i)^degree_j is imaginary
    # for opposite degree parity; symmetric +/- lattice then integrates to zero.
    for i,(xi,yi) in enumerate(m.POWERS):
        for j,(xj,yj) in enumerate(m.POWERS):
            real_phase=(1,0,-1,0)[(xi+yi-xj-yj)%4]
            excluded=(real_phase==0 or (yi+yj)%2==1)
            expected=(xi+yi-xj-yj)%2 or (yi+yj)%2
            assert bool(excluded)==bool(expected)
            if excluded:
                assert report['quadratic']['entry_majorants'][i][j]=='0'
    # g2/g2 is not symmetry-forbidden and its second-order variation is nonzero.
    assert F(report['quadratic']['entry_majorants'][2][2])>0


def test_strict_json_negative_controls(tmp_path):
    for raw in ('{"x":1,"x":2}','{"x":NaN}','{"x":1.0}'):
        path=tmp_path/'bad.json';path.write_text(raw)
        with pytest.raises(ValueError): m.strict_json(path)
    oversized = tmp_path/'oversized.json'
    oversized.write_bytes(b' ' * 2_000_001)
    with pytest.raises(ValueError, match='certificate too large'):
        m.strict_json(oversized)


def test_portable_candidate_omits_host_paths_and_matches_exact_report(report):
    assert 'source_root' not in report and 'interval_implementation_root' not in report
    assert json.dumps(m.strict_json(HERE/'candidate.json'), sort_keys=True) == json.dumps(report, sort_keys=True)


def test_default_candidate_from_other_working_directory_and_no_overwrite(tmp_path):
    output = tmp_path/'retained.json'
    output.write_bytes(b'original retained bytes')
    args = [sys.executable, '-B', str(HERE/'lpw_modulus.py')]
    good = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert good.returncode == 0, good.stderr
    refused = subprocess.run([*args, '--output', str(output)], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert refused.returncode != 0
    assert output.read_bytes() == b'original retained bytes'
