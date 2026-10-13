"""census_auxiliary whitelist: manifest-only mutation (the fixture's check.py reads the fixture's SOURCE_FILES.json)."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

TREE = Path(sys.argv[1])
SIDE = TREE / 'companions/torus_filtration_20261011'
spec = importlib.util.spec_from_file_location('tft', SIDE / 'test_torus_filtration.py')
tft = importlib.util.module_from_spec(spec)
sys.argv = [str(SIDE / 'test_torus_filtration.py')]
spec.loader.exec_module(tft)
c = tft.checker()
CONGR = 'TorusFiltration.torus_periodBasis.congr_simp'


def fail_line(run):
    return [l for l in run.stderr.splitlines() if 'FAIL' in l][:1]


with tempfile.TemporaryDirectory() as tmp:
    f = tft.Fixture(c, tmp)
    for aux in ([CONGR, 'TorusFiltration.superlevel_mono.hidden'], [CONGR, 'TorusFiltration..x'],
                [CONGR, 'TorusFiltration._private.0.x'], [CONGR, 'TorusFiltration.a.b.c.d'],
                [CONGR, 'TorusFiltration.hidden'], ['Other.a.b'], [CONGR, CONGR]):
        f.manifest(lambda m, aux=aux: m.__setitem__('census_auxiliary', aux))
        meta_in_fixture = json.loads((f.side / 'SOURCE_FILES.json').read_text())['census_auxiliary']
        run = f.run('source')
        print('aux=%-45s fixture-manifest=%s source rc=%d %s' % (aux[-1], meta_in_fixture == aux, run.returncode, fail_line(run)))
    f.restore()
    # finish with a user-named where-clause auxiliary whitelisted: craft census.log = targets + both auxiliaries
    f.manifest(lambda m: m.__setitem__('census_auxiliary', [CONGR, 'TorusFiltration.superlevel_mono.hidden']))
    out = f.evidence()
    meta = json.loads((f.side / 'SOURCE_FILES.json').read_text())
    assert meta['census_auxiliary'] == [CONGR, 'TorusFiltration.superlevel_mono.hidden']
    listed = sorted(c.targets(meta) + meta['census_auxiliary'])
    (out / 'census.log').write_text(''.join(n + '\n' for n in listed) + 'CENSUS_END %d\n' % len(listed))
    fin = f.run('finish', str(out))
    print('finish with whitelisted superlevel_mono.hidden: rc=%d' % fin.returncode,
          json.loads(fin.stdout)['census'] if fin.returncode == 0 else fail_line(fin))
    # control: same census.log without the whitelist entry in the manifest must be refused as extra
    (out / 'receipt.json').unlink()
    f.manifest(lambda m: m.__setitem__('census_auxiliary', [CONGR]))
    src = f.run('source')
    (out / 'source.json').write_text(src.stdout)
    fin2 = f.run('finish', str(out))
    print('control, whitelist removed: rc=%d %s' % (fin2.returncode, fail_line(fin2)))
