# Diagnostic: run two indep_ht.py modes in ONE process (warm mpmath quadrature caches).
import runpy, sys
script, first, second = sys.argv[1], sys.argv[2], sys.argv[3]
for mode in (first, second):
    print(f"=== mode {mode}", flush=True)
    sys.argv = [script] + mode.split(":")
    runpy.run_path(script, run_name="__main__")
