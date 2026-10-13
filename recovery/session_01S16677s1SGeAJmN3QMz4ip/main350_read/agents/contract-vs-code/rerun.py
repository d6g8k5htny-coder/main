import sys, json, time, pathlib
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import run_global_h0_selector as drv
base = pathlib.Path('.').resolve()
out = base/'experiments/universality/results/global_selector_1'
t0=time.monotonic()
res = drv.run_attempt(base=base, output_dir=out)
print('rerun status', res['certificate']['status'], 'elapsed', round(time.monotonic()-t0,2))
