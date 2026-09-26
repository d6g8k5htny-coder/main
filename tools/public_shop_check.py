"""Check the static shop against bounded, immutable public input bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def main():
    config = json.loads((ROOT / 'docs/site/config.json').read_bytes())
    with tempfile.TemporaryDirectory(prefix='public-shop-') as directory:
        fixtures = Path(directory)
        for field in ('coefficient', 'proof', 'imports', 'query'):
            pin = config[field]
            if pin['repository'] not in ('d6g8k5htny-coder/Math-', 'd6g8k5htny-coder/query-'):
                raise ValueError('Unexpected public repository')
            from urllib.parse import quote
            expected = 'https://raw.githubusercontent.com/' + pin['repository'] + '/' + pin['commit'] + '/' + quote(pin['path'], safe='/')
            if pin['url'] != expected or len(pin['commit']) != 40:
                raise ValueError('Pin URL/identity mismatch')
            with urlopen(expected, timeout=30) as response:
                raw = response.read(pin['bytes'] + 1)
            if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
                raise ValueError('Public source identity mismatch: ' + field)
            destination = fixtures / ('query' if field == 'query' else 'math') / pin['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        subprocess.run(['python3', '-B', 'tools/public_shop_data.py', '--math-root', str(fixtures/'math'), '--check'], cwd=ROOT, check=True)
        env = dict(os.environ, SHOP_MATH_FIXTURE=str(fixtures/'math'), SHOP_QUERY_FIXTURE=str(fixtures/'query'))
        subprocess.run(['node', 'tests/test_public_shop_app.mjs'], cwd=ROOT, env=env, check=True)
    notebook = json.loads((ROOT/'docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb').read_bytes())
    if notebook['nbformat'] != 4 or not notebook['cells']:
        raise ValueError('Invalid notebook format')
    print('Pinned public sources, static interactions and notebook structure checked. Scientific effect NONE.')

if __name__ == '__main__':
    main()
