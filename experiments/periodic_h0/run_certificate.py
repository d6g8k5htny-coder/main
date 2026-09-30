"""Capture rounded generator coefficients, then run an exact finite-field replay."""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch
import finite_certificate as fc


def extract_coefficients(config):
    # Import only for extraction. finite_certificate.py --verify is stdlib-only.
    import experiment as ex
    import numpy as np
    fc.require(config['side'] == 24 and config['dimension'] == 2 and config['cutoff'] == config['bank_max_cutoff'] == 24, 'Only the existing planar cutoff24 configuration is supported')
    records = []
    n = 64
    expected = {(x, y) for x in range(25) for y in range(-24, 25) if x > 0 or y > 0}
    for seed in config['seeds']:
        captured = []

        def capture(array, *, norm):
            fc.require(norm == 'forward' and array.shape == (n, n) and np.isfinite(array).all(), 'Unexpected FFT input')
            captured.append(array.copy())
            # No FFT or field samples are calculated in this extraction call.
            return np.zeros_like(array)

        with patch.object(ex.np.fft, 'ifft2', side_effect=capture):
            ex.field_grid(ex.mode_bank(seed, 24), n, 24)
        fc.require(len(captured) == 1, 'Expected exactly one generator coefficient array')
        coeff = captured[0]
        fc.require(coeff[0, 0].imag == 0, 'Nonreal DC coefficient')
        occupied = {(0, 0)}
        modes = []
        for x, y in sorted(expected):
            z = coeff[x % n, y % n]
            fc.require(coeff[-x % n, -y % n] == z.conjugate(), 'Conjugate symmetry failed')
            occupied.update(((x % n, y % n), (-x % n, -y % n)))
            modes.append([x, y, str(Q.from_float(float(z.real))), str(Q.from_float(float(z.imag)))])
        fc.require(all(coeff[x, y] == 0 for x in range(n) for y in range(n) if (x, y) not in occupied), 'Unexpected occupied coefficient')
        records.append({'seed': seed, 'dc': [str(Q.from_float(float(coeff[0, 0].real))), '0'], 'modes': modes})
    result = {'schema_version': 1, 'model': fc.MODEL, 'side': 24, 'cutoff': 24,
              'grids': config['grids'], 'records': records}
    fc.validate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory')
    base = Path(__file__).resolve().parent
    config = fc.load_json(base/'refinement_config.json')
    data = extract_coefficients(config)
    certificate = fc.certify(data)
    fc.verify(data, certificate)
    sources = {}
    for role, filename in fc.SOURCE_FILES.items():
        raw = (base/filename).read_bytes()
        sources[role] = {'filename': filename, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    args.output.mkdir(parents=True)
    outputs = {}
    for filename, obj in [('COEFFICIENTS.json', data), ('CERTIFICATE.json', certificate)]:
        raw = fc.canonical_bytes(obj)
        (args.output/filename).write_bytes(raw)
        outputs[filename] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    import numpy as np
    receipt = {'schema_version': 1, 'utc': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
               'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
               'sources': sources, 'outputs': outputs, **fc.RECEIPT_TEXT}
    receipt_bytes = fc.canonical_bytes(receipt)
    (args.output/'RUN.json').write_bytes(receipt_bytes)
    (args.output/'RUN.sha256').write_bytes((hashlib.sha256(receipt_bytes).hexdigest()+'\n').encode('ascii'))
    (args.output/'RESULTS.md').write_bytes(fc.render_report(certificate).encode('utf-8'))
    fc.verify_directory(args.output)
    print(json.dumps({'records': len(data['records']), 'mode_pairs': sum(len(r['modes']) for r in data['records']), 'outputs': outputs}))


if __name__ == '__main__':
    main()
