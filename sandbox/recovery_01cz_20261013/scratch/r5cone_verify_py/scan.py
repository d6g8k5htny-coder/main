import importlib.util, sys, json
from pathlib import Path
P = Path(sys.argv[1]); F = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('pp_check', P / 'check.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
registered, structures, rsha = m.load_register((P / m.STATEMENT_DIR).resolve())
print('registered has ConeMomentD2Identity:', 'UniversalLaw.Spec.Side24.ConeMomentD2Identity' in registered,
      registered.get('UniversalLaw.Spec.Side24.ConeMomentD2Identity'))
text = F.read_text(encoding='utf-8')
code, docs = m.lex(text)
print('refused_tokens:', m.refused_tokens(code))
scan = m.scan_module(text, 'Side24ConeMoment', registered)
print(json.dumps({k: v for k, v in scan.items()}, indent=1, default=str)[:3000])
