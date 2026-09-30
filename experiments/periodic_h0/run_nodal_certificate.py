"""Record fresh NumPy samples and certify them against stored finite coefficients."""
import argparse
import hashlib
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc
import nodal_certificate as nc


def exact_float(value):
    q = fc.rational(value)
    try: result = float(q)
    except OverflowError as exc: raise ValueError('Coefficient not representable as float64') from exc
    fc.require(Q.from_float(result) == q, 'Coefficient not exactly representable as float64')
    return result


def sample(data, seed, grid):
    import numpy as np
    record = nc.check_selection(data, seed, grid)
    coefficients = np.zeros((grid, grid), dtype=np.complex128)
    coefficients[0, 0] = exact_float(record['dc'][0])
    for x, y, real, imag in record['modes']:
        z = complex(exact_float(real), exact_float(imag))
        coefficients[x % grid, y % grid] = z
        coefficients[-x % grid, -y % grid] = z.conjugate()
    values = np.fft.ifft2(coefficients, norm='forward').real
    result = {'schema_version': 1, 'coefficient_object_sha256': fc.digest(data),
              'seed': seed, 'grid': grid, 'layout': 'row-major-x-then-y',
              'values': [float(v).hex() for v in values.flat]}
    nc.parse_samples(data, result)
    return result


def produce(output, *, seed=34000, grid=128):
    output = Path(output)
    fc.require(not output.exists(), 'Output must be a new directory')
    data = fc.load_json(nc.BASE/nc.COEFFICIENT_PATH)
    samples = sample(data, seed, grid)
    certificate = nc.certify(data, samples)
    import numpy as np
    output.mkdir(parents=True)
    for filename, obj in [('SAMPLES.json', samples), ('CERTIFICATE.json', certificate)]:
        (output/filename).write_bytes(fc.canonical_bytes(obj))
    receipt = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
               'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
               'sources': {p: nc.file_identity(nc.BASE/p) for p in nc.SOURCE_PATHS},
               'outputs': {p: nc.file_identity(output/p) for p in ('SAMPLES.json', 'CERTIFICATE.json')},
               **nc.SCOPE}
    raw = fc.canonical_bytes(receipt)
    (output/'RUN.json').write_bytes(raw)
    (output/'RUN.sha256').write_bytes((hashlib.sha256(raw).hexdigest()+'\n').encode('ascii'))
    (output/'RESULTS.md').write_bytes(nc.render_report(certificate).encode('utf-8'))
    nc.verify_directory(output)
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=34000)
    parser.add_argument('--grid', type=int, default=128)
    args = parser.parse_args()
    cert = produce(args.output, seed=args.seed, grid=args.grid)
    print('Nodal error upper endpoint: '+cert['nodal_error_decimal_up'])
    print('Abstract filtration bound upper endpoint: '+cert['abstract_diagram_bound_decimal_up'])


if __name__ == '__main__': main()
