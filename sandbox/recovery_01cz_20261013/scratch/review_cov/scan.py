import importlib.util, sys, json
from pathlib import Path
pkt = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('chk', pkt / 'check.py')
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
registered, structures, rsha = chk.load_register((pkt / chk.STATEMENT_DIR).resolve())
text = Path(sys.argv[2]).read_text(encoding='utf-8')
res = chk.scan_module(text, 'Side24Covariance', registered)
print(json.dumps({k: v for k, v in res.items()}, indent=1, default=str)[:4000])
# docstring and Proves/Method/DNE full text
code, docs = chk.lex(text)
print(chk.refused_tokens(code))
