"""Finite Fourier law candidates and an explicitly uncertified float sampler.

Ideal circular coefficient vectors preserve stationarity under spatial shifts.
This implementation uses Python's deterministic PRNG and libm; it does not
certify continuous IID inputs, Gaussianity, rotation invariance or field error.
"""
import hashlib
import json
import math
from pathlib import Path
import random


CATALOG_PATH = Path(__file__).with_name('models.json')
SPECTRUM_FORMULAS = {
    'gaussian': 'exp(-s)',
    'super_gaussian': 'exp(-s*s)',
    'polynomial4': '(1+s)^(-4)',
    'rational_quartic': '(1+s*s)^(-2)',
    'two_scale': '(3/4)*exp(-s)+(1/4)*exp(-s/16)',
    'annulus': 'exp(-(sqrt(s)-2)^2)',
    'notch': 'exp(-s)*(1-(9/10)*exp(-(s-4)^2))',
    'angular': 'exp(-s)*(1+(kx^4+ky^4)/(1+s*s))',
    'separable': 'exp(-abs(kx)-abs(ky))',
    'axial': 'exp(-s)*(1+3/(1+kx*kx)+3/(1+ky*ky))',
}
LAW_CONSTRUCTIONS = {
    'gaussian': '(Z1,Z2), independent standard normals',
    'fixed_radius': 'sqrt(2)*(cos(T),sin(T)), T uniform on [0,2*pi)',
    'uniform_disk': '2*sqrt(U)*(cos(T),sin(T)), U uniform (0,1), T uniform [0,2*pi), independent',
    'exponential_scale': 'sqrt(V)*(Z1,Z2), V exponential mean 1, independent standard normals',
    'student5': 'sqrt(3/S)*(Z1,Z2), S chi-square(5), independent standard normals',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: '+key)
        result[key] = value
    return result


def load_catalog(path=CATALOG_PATH):
    catalog = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog):
    require(type(catalog) is dict and type(catalog.get('schema_version')) is int
            and catalog.get('schema_version') == 1,
            'Unknown catalog schema')
    require(catalog.get('scientific_status_authority') is False,
            'Catalog cannot award scientific status')
    domain = catalog.get('domain')
    require(type(domain) is dict and type(domain.get('dimension')) is int
            and domain['dimension'] == 2, 'Only the declared planar catalog is implemented')
    spectra, laws, items = (catalog.get(k) for k in ('spectra', 'coefficient_laws', 'models'))
    require(type(spectra) is dict and set(spectra) == set(SPECTRUM_FORMULAS),
            'Expected the ten implemented spectral mechanisms')
    require(type(laws) is dict and set(laws) == set(LAW_CONSTRUCTIONS),
            'Expected the five implemented circular laws')
    for key, formula in SPECTRUM_FORMULAS.items():
        require(spectra[key].get('formula') == formula, 'Unimplemented spectral definition: '+key)
    for key, construction in LAW_CONSTRUCTIONS.items():
        require(laws[key].get('construction') == construction
                and laws[key].get('ideal_vector_symmetry') == 'circular'
                and laws[key].get('ideal_component_variance') == '1',
                'Unimplemented coefficient law: '+key)
    require(type(items) is list and len(items) == 50, 'Exactly fifty distinct laws required')
    pairs, identities = set(), set()
    for model in items:
        require(type(model) is dict and set(model) == {'id', 'spectrum_id', 'coefficient_law_id', 'class'},
                'A model is a law, not a seed or status record')
        spec, law = model['spectrum_id'], model['coefficient_law_id']
        require(spec in spectra and law in laws, 'Unknown model definition')
        require(model['id'] == spec+'__'+law, 'Model identity must identify its law')
        require((spec, law) not in pairs and model['id'] not in identities, 'Duplicate law')
        pairs.add((spec, law)); identities.add(model['id'])
        kind = 'gaussian_candidate' if law == 'gaussian' else 'non_gaussian_stress_candidate'
        require(model['class'] == kind, 'Gaussian and non-Gaussian scope mismatch')
    require(pairs == {(spec, law) for spec in spectra for law in laws}, 'Missing spectrum-law pair')
    return True


def _weight(spectrum_id, x, y):
    s = x*x+y*y
    if spectrum_id == 'gaussian': return math.exp(-s)
    if spectrum_id == 'super_gaussian': return math.exp(-s*s)
    if spectrum_id == 'polynomial4': return (1+s)**-4
    if spectrum_id == 'rational_quartic': return (1+s*s)**-2
    if spectrum_id == 'two_scale': return .75*math.exp(-s)+.25*math.exp(-s/16)
    if spectrum_id == 'annulus': return math.exp(-(math.sqrt(s)-2)**2)
    if spectrum_id == 'notch': return math.exp(-s)*(1-.9*math.exp(-(s-4)**2))
    if spectrum_id == 'angular': return math.exp(-s)*(1+(x**4+y**4)/(1+s*s))
    if spectrum_id == 'separable': return math.exp(-abs(x)-abs(y))
    if spectrum_id == 'axial': return math.exp(-s)*(1+3/(1+x*x)+3/(1+y*y))
    raise ValueError('Unknown spectral mechanism')


def normalized_spectrum(spectrum_id, cutoff):
    require(spectrum_id in SPECTRUM_FORMULAS, 'Unknown spectral mechanism')
    require(type(cutoff) is int and 1 <= cutoff <= 3,
            'This float pilot supports cutoffs 1 through 3; larger profiles need underflow handling')
    weights = {(x, y): _weight(spectrum_id, x, y)
               for x in range(-cutoff, cutoff+1) for y in range(-cutoff, cutoff+1)}
    require(all(math.isfinite(v) and v > 0 for v in weights.values()),
            'A positive finite spectral profile is required')
    total = math.fsum(weights.values())
    return {k: v/total for k, v in weights.items()}


def covariance(weights, angular_lag):
    """Ideal finite-law covariance; angular coordinates have period 2*pi."""
    return math.fsum(q*math.cos(x*angular_lag[0]+y*angular_lag[1])
                     for (x, y), q in weights.items())


def definition(model, *, cutoff=3, side=24, catalog=None):
    catalog = load_catalog() if catalog is None else catalog
    require(model in catalog['models'], 'Model is not in the validated catalog')
    require(type(side) in (int, float) and math.isfinite(side) and side > 0,
            'Positive finite torus side required')
    weights = normalized_spectrum(model['spectrum_id'], cutoff)
    profile = [{'mode': list(k), 'q_float_hex': weights[k].hex()} for k in sorted(weights)]
    return {'model': model, 'dimension': 2, 'torus_side': side, 'square_cutoff': cutoff,
            'spectrum': catalog['spectra'][model['spectrum_id']],
            'coefficient_law': catalog['coefficient_laws'][model['coefficient_law_id']],
            'normalized_float_spectrum': profile,
            'covariance_fingerprint_sha256': digest(profile),
            'field_formula': 'sqrt(q0)*C0 + sum_canonical_pairs sqrt(2*qk)*(Ak*cos(2*pi*k.x/L)+Bk*sin(2*pi*k.x/L))',
            'constant_mode': 'First coordinate of a separately sampled coefficient vector',
            'continuum_admissibility': 'UNASSESSED; finite-cutoff candidate only'}


def _uniform(rng):
    # Open dyadic midpoints avoid log(0). This discrete law is not ideal U(0,1).
    return (rng.getrandbits(52)+.5)/2**52


def _normal_pair(rng):
    radius = math.sqrt(-2*math.log(_uniform(rng)))
    angle = 2*math.pi*_uniform(rng)
    return radius*math.cos(angle), radius*math.sin(angle)


def _coefficient_pair(law_id, rng):
    if law_id == 'gaussian': return _normal_pair(rng)
    if law_id in ('fixed_radius', 'uniform_disk'):
        radius = math.sqrt(2) if law_id == 'fixed_radius' else 2*math.sqrt(_uniform(rng))
        angle = 2*math.pi*_uniform(rng)
        return radius*math.cos(angle), radius*math.sin(angle)
    if law_id == 'exponential_scale':
        z = _normal_pair(rng)
        factor = math.sqrt(-math.log(_uniform(rng)))
        return factor*z[0], factor*z[1]
    if law_id == 'student5':
        z = _normal_pair(rng)
        normals = [v for _ in range(3) for v in _normal_pair(rng)][:5]
        factor = math.sqrt(3/math.fsum(v*v for v in normals))
        return factor*z[0], factor*z[1]
    raise ValueError('Unknown coefficient law')


def sample_grid(model, seed, *, n=16, cutoff=3, side=24):
    """Return flat row-major float samples; no nodal/continuum certificate."""
    require(type(n) is int and n >= 3 and type(cutoff) is int and n > 2*cutoff,
            'Grid must satisfy n > 2K without aliasing')
    require(type(seed) is int and seed >= 0, 'Nonnegative strict integer seed required')
    definition(model, cutoff=cutoff, side=side)
    weights = normalized_spectrum(model['spectrum_id'], cutoff)
    rng = random.Random(seed)
    law = model['coefficient_law_id']
    constant = math.sqrt(weights[0, 0])*_coefficient_pair(law, rng)[0]
    modes = []
    for (x, y), weight in weights.items():
        if x > 0 or (x == 0 and y > 0):
            a, b = _coefficient_pair(law, rng)
            factor = math.sqrt(2*weight)
            modes.append((x, y, factor*a, factor*b))
    values = [constant+math.fsum(a*math.cos(2*math.pi*(kx*x+ky*y)/n)
                               + b*math.sin(2*math.pi*(kx*x+ky*y)/n)
                               for kx, ky, a, b in modes)
              for x in range(n) for y in range(n)]
    require(all(math.isfinite(v) for v in values), 'Nonfinite floating samples')
    return values
