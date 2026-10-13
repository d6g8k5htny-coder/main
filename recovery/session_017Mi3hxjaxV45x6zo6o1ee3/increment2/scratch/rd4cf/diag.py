import sys, time, json
sys.argv = ["x"]
sys.path.insert(0, "<SCRATCH>/wtd5/coefficients/side24_d5_20260930")
import importlib.util
spec = importlib.util.spec_from_file_location("d5", "<SCRATCH>/wtd5/coefficients/side24_d5_20260930/cone_moment_d5.py")
d5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d5)
from fractions import Fraction as Fr
br = d5.m4_setup()
out = {}
for name, sel in (("E", [br[0]]), ("R", [br[1]])):
    for K, cells, tag in ((40, None, "K40"), (30, None, "K30"), (40, [(Fr(i, 4), Fr(i + 1, 4)) for i in range(0, 120)] + [(Fr(i), Fr(i + 1)) for i in range(30, 60)], "K40cells1/4")):
        t0 = time.time()
        iv = d5.enclose_m4(sel, K=K, cells=cells)
        out[name + " " + tag] = iv.pair()
        print(name, tag, iv.pair(), round(time.time() - t0, 1), "s", flush=True)
json.dump(out, open("diag.json", "w"), indent=1)
