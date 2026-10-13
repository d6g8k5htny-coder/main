import importlib.util, json, sys
from pathlib import Path
P = Path(sys.argv[1]); M = Path(sys.argv[2]); SR = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('pcheck', P / 'check.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
registered, structures, regsha = mod.load_register(SR)
print("ReferenceCovarianceLowerBound registered:", registered.get("UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound"))
info = mod.scan_module(M.read_text(encoding='utf-8'), 'Side24Covariance', registered)
print(json.dumps(info, indent=1, default=str)[:3000])
