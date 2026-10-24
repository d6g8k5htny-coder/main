# Reconstruction (21 Oct 2026) of the 18 Oct filtering step behind main#419/6245774604.
# Keeps a scan2.py hit only if the quoted number equals one of the published H3
# values, rounded half-up or truncated at the quote's own number of decimals.
import json, sys
from decimal import Decimal as D, ROUND_HALF_UP, ROUND_DOWN, getcontext
getcontext().prec = 100
P = ["3.66282864761194", "3.66435028046302", "3.67360595202419", "3.72414748332433",
     "3.74767948915996", "2.58288222715895", "2.58174158877406", "2.41792545239159",
     "2.30659559567154"]
hits = json.load(open(sys.argv[1]))
out = []
for h in hits:
    q = h["quoted"]
    n = len(q.split(".")[1])
    e = D(1).scaleb(-n)
    if any(D(v).quantize(e, rounding=r) == D(q) for v in P for r in (ROUND_HALF_UP, ROUND_DOWN)):
        out.append(dict(h, decimals=n))
json.dump(out, sys.stdout, indent=0)
