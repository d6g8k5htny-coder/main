import sys, json
from mc_event import run
r = eval(sys.argv[1]); b = float(sys.argv[2]); k = float(sys.argv[3]); n = int(sys.argv[4]); seed = int(sys.argv[5])
est, Z = run(r, b, k, n, seed)
print(json.dumps({'r': r, 'b': b, 'k': k, 'n': n, 'seed': seed, 'ratio': est, 'Z_r2': Z}))
