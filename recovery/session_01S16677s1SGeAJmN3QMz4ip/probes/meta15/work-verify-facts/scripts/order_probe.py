import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe
probe.SRC = Path(sys.argv[1]); probe.PY = sys.argv[2]; probe.SCRIPT = probe.SRC/'tools/architecture_conformance.py'
S = probe.SSA
for n in (600, 1000):
    deep = '{"a":'*n + '1' + '}'*n
    probe.summarize(f'n={n} deep then b true', {'x.json': '{"a":' + deep + ',"b":{' + S + ':true}}'})
    probe.summarize(f'n={n} b true then deep', {'x.json': '{"b":{' + S + ':true},"a":' + deep + '}'})
