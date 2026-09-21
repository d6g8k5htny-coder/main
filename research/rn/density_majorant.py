"""Conditional SIDE24 density times three-determinant envelope.

This checks arithmetic and custody of imported source premises. Supplied D,Q
are NOT checked spatial premises and cannot establish field applicability,
coverage, any status change, or organizational independence. See PROOF.md.
Only the repository's certified interval package is imported as math code.
"""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import json
import re
import stat
from zipfile import ZipFile

from research.interval import Interval as I, exp, pi, sqrt
from research.rn.n6_inputs import authenticated_inputs
from tools.twelve_project_check import MAX_JSON, check_exclusions, read_bounded, strict_json

RADIUS = F(1, 20)
BIRTH = F(6, 5)
WINDOW_LENGTH = F(1, 48000)
Z_LO = F(15518583475065571, 2000000000000000000)
PIN_ENERGY_UPPER = F(1266466, 160083)
ENERGY_CEILING = 8
HESSIAN_VARIANCE_CAP = 4
MOMENT_FACTOR = 512
Q_LIMIT = F(2048)
INPUT_BITS = 4096

# All members below are read as data; no archived program is imported/executed.
SOURCE_CONTRACTS = (
    ('research/campaigns/h3_rn_n6_20260920_v1.zip', 571735,
     '73b9e63800f77c677c504f3097fe0bb2459ff6c94cdee5772b56062320aefc21', 71, (
        ('MANIFEST.json', 12008, '81770e26ec654d3b462f06cebaa20d0dd282cfd4df10ac593c7887ea0d292b40'),
        ('pin_energy/PROOF.md', 2199, '695daddd5b96e191865c10c02aa611c828a8a13f0d7c99f3e0c6030f476ea113'),
        ('pin_energy/PROOF_v2.md', 1663, '16ab7b5fa54c8f4601d470945508d1686dacc9a79d8cdf0122f4cea17338b50e'),
        ('pin_energy/result-v2.json', 2310, 'bcaff3f0afa46473d1ca61b5b1c6985cf3db86988fafed5f89c84b8bf5b2db69'),
        ('pin_energy/check_energy.py', 3750, '8b83fc319f830db8b3caeb28fce7ac821193775b7184ff6e84bdb190f463b509'),
        ('rn_n6/PROOF.md', 10662, '850291813ff1e0c83cf85987b8132a756dfe3bf83fda2731b5c2bdc0ebe7e3df'),
        ('rn_n6/side24_taylor.py', 23681, '32c4933c2ab543f738bd54f98066b351a334e377277e8ebcab5879438f418598'),
    )),
    ('drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_REPAIR_AND_ERRATUM_BUNDLE.zip',
     140170, '28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e', 27, (
        ('closure_round2/rn_field.py', 16701, 'd9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8'),
        ('intake/rn_source/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md', 7003,
         '6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa'),
    )),
)


class MajorantError(ValueError):
    """Rejected arithmetic/premise contract; never a positive field verdict."""


def _exact(value, name):
    if type(value) not in (int, F):
        raise TypeError(f'{name}: exact int/Fraction required; float/bool refused')
    value = F(value)
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > INPUT_BITS:
        raise MajorantError(f'{name}: rational resource limit exceeded')
    return value


def _precision(bits):
    if type(bits) is not int or not 128 <= bits <= 1024:
        raise MajorantError('bits must be an integer in 128..1024')
    return (bits+2)//3+24


def algebra_certificate():
    """Exact finite identities supporting the uniform analytic proof.

    Gaps are coefficients in ascending powers of q>=0, for T=8+q.
    This is an arithmetic witness, not an automated proof of Gaussian theory.
    """
    if not PIN_ENERGY_UPPER < ENERGY_CEILING == 8:
        raise MajorantError('source pin energy no longer fits the proved ceiling')
    if HESSIAN_VARIANCE_CAP != 4 or MOMENT_FACTOR != (2*HESSIAN_VARIANCE_CAP)**3:
        raise MajorantError('Hessian variance or Holder constant changed')
    if RADIUS != F(1, 20) or BIRTH != F(6, 5) or WINDOW_LENGTH != RADIUS**3/6:
        raise MajorantError('fixed-radius pin/window contract changed')
    return {
        'gaussian_sixth_moment_coefficients': (1, 15, 45, 15),
        'bernstein_controls': ('15', '15*T', '5*T^2', 'T^3'),
        'T_cubed_minus_control_coefficients': (
            (497, 192, 24, 1), (392, 177, 24, 1), (192, 112, 19, 1), (0,)),
        'negative_twice_derivative_after_exp_factor': (128, 96, 18, 1),
        'sixth_moment_variance_factor': 64,
        'three_determinant_moment_factor': MOMENT_FACTOR,
        'strict_derivative_identity': "-2*exp(q/2)*f'(q)=(q+8)^2*(q+2)",
    }


def source_binding(repo_root=None):
    """Fresh bounded checks of exact imported sources; no proof rerun/cache.

    Byte identity establishes custody, not a new proof or operator acceptance.
    The historical fixed-r floor is deliberately retained as an imported premise.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    _, authenticated_identities = authenticated_inputs(root)
    # The successor gate sees the historic outer ZIP as a dependency. Apply
    # current holds to our selected inner members too, before reading bodies.
    exclusions = strict_json(read_bounded(root/'quarantine/EXCLUSIONS.json', MAX_JSON))['exclusions']
    for relative, size, digest, _, members in SOURCE_CONTRACTS:
        check_exclusions({relative: {'bytes': size, 'sha256': digest}}, exclusions)
        check_exclusions({name: {'bytes': size, 'sha256': digest}
                          for name, size, digest in members}, exclusions)
    records, content = [], {}
    for relative, size, digest, count, members in SOURCE_CONTRACTS:
        path = root/relative
        st = path.lstat()
        if path.absolute() != path.resolve() or not stat.S_ISREG(st.st_mode) or st.st_size != size:
            raise MajorantError(f'source size/type mismatch: {relative}')
        # Hash and inspect the same bounded byte buffer, avoiding a second read.
        with path.open('rb') as handle:
            raw = handle.read(size+1)
        if len(raw) != size or sha256(raw).hexdigest() != digest:
            raise MajorantError(f'source archive hash mismatch: {relative}')
        from io import BytesIO
        with ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) != count or len({x.filename for x in entries}) != count:
                raise MajorantError('source archive member count/duplicates')
            member_records = []
            for name, nbytes, member_hash in members:
                entry = archive.getinfo(name)
                if entry.file_size != nbytes:
                    raise MajorantError(f'source member size mismatch: {name}')
                data = archive.read(entry)
                if len(data) != nbytes or sha256(data).hexdigest() != member_hash:
                    raise MajorantError(f'source member hash mismatch: {name}')
                content[name] = data
                member_records.append({'path': name, 'bytes': nbytes, 'sha256': member_hash,
                                       'executed': False})
        records.append({'path': relative, 'bytes': size, 'sha256': digest, 'members': member_records})
    energy = json.loads(content['pin_energy/result-v2.json'])
    if (energy['schema'] != 'side24-uniform-pin-energy-v1'
            or F(energy['uniform_energy_upper']) != PIN_ENERGY_UPPER
            or energy['birth'] != '6/5' or energy['fixed_axis'] != 'x'):
        raise MajorantError('source pin-energy semantic contract changed')
    floor = content[SOURCE_CONTRACTS[1][4][1][0]].decode('utf-8')
    match = re.findall(r'Z_\{0\.05\} ∈ \[\s*([0-9.eE+\-]+)', floor)
    if len(match) != 1 or F(match[0]) != Z_LO:
        raise MajorantError('source fixed-r floor semantic contract changed')
    return {'archives': records, 'checked': True, 'source_programs_executed': False,
            'current_eligibility_checked': True, 'authenticated_identities': authenticated_identities,
            'pin_energy': 'IMPORTED_EXISTING_PROOF_NOT_RERUN',
            'hessian_variance': 'IMPORTED_NORMALIZED_TORUS_CAP_NOT_PLANAR_EQUALITY',
            'h3_floor': 'IMPORTED_FIXED_R_SOURCE_PREMISE_NOT_REPROVED'}


def density_moment_majorant(determinant_lower, mahalanobis_lower, *,
                            jacobian_upper=1, bits=192, repo_root=None):
    """Conditional upper envelope, assuming the supplied D,Q hold everywhere.

    D>0 lower-bounds det(Sigma) for the six-pin conditional three-jet law.
    Q>=0 lower-bounds its Mahalanobis energy at (v,0,0), for ALL window marks.
    An invertible three-jet transform is allowed only with the matching D and
    a bound on its absolute determinant, supplied as jacobian_upper. This is
    a density-coordinate Jacobian, not the spatial area Jacobian.

    majorant_interval encloses the explicit envelope expression. Its lower
    endpoint is NOT a lower bound on the typed integrand. External numeric
    premises never self-promote to applicability, field, or wedge certificates.
    """
    prec = _precision(bits)
    D = _exact(determinant_lower, 'determinant_lower')
    Q = _exact(mahalanobis_lower, 'mahalanobis_lower')
    J = _exact(jacobian_upper, 'jacobian_upper')
    if D <= 0 or Q < 0 or J <= 0:
        raise MajorantError('require determinant_lower>0, mahalanobis_lower>=0, jacobian_upper>0')
    algebra = algebra_certificate()
    binding = source_binding(repo_root)
    # f is decreasing. Lowering Q remains safe, bounds exp resources, and does
    # not silently replace a tiny positive upper bound by zero/underflow.
    q_used = min(Q, Q_LIMIT)
    decay = exp(I.exact(-q_used/2), prec).round_out(bits)
    denominator = sqrt((2*pi(prec))**3*I.exact(D), prec).round_out(bits)
    if denominator.lo <= 0:
        raise MajorantError('density denominator positivity inconclusive')
    majorant = (I.exact(J*WINDOW_LENGTH*MOMENT_FACTOR*(8+q_used)**3/Z_LO)
                *decay/denominator).round_out(bits)
    if majorant.lo < 0 or majorant.hi <= 0:
        raise MajorantError('positive finite majorant not established')
    return {
        'schema': 'side24-density-moment-majorant-v1',
        'determinant_lower': D, 'mahalanobis_lower': Q,
        'mahalanobis_used': q_used, 'jacobian_upper': J,
        'window_length': WINDOW_LENGTH, 'normalization_floor': Z_LO,
        'pin_energy_upper': PIN_ENERGY_UPPER, 'hessian_variance_cap': HESSIAN_VARIANCE_CAP,
        'round_bits': bits, 'majorant_interval': majorant,
        'integrand_upper': majorant.hi, 'typed_integrand_range': I(0, majorant.hi),
        'arithmetic_checked': True, 'external_spatial_premises_checked': False,
        'applicability': 'CONDITIONAL_ON_COMMON_LAW_DETERMINANT_AND_FULL_MARK_ENERGY_BOUNDS',
        'radius': RADIUS, 'birth': BIRTH, 'fixed_pin_axis': 'x',
        'hessian_order': ('xx', 'yy', 'xy'), 'three_jet_order': ('f', 'fx', 'fy'),
        'source_binding': binding, 'algebra': algebra,
        'authority': 'NONE', 'field_certified': False, 'spatial_cover': False,
        'wedge_certified': False, 'scientific_status_changed': False,
        'organizational_independence_credit': 0, 'original_prize_closed': False,
        'not_established': (
            'validity of supplied D,Q,Jacobian premises for any spatial region',
            'spatial area or partition or integral upper bound',
            'RN uniform closure, all-r or all-angle result, q0 or event-0 closure',
            'reproof or promotion of imported H3 floor and pin-energy sources'),
    }
