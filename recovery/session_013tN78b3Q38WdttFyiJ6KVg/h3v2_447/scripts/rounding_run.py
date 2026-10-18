"""Run h3_band_v2.main() with mpmath's round-to-nearest mp context switched to a directed mode.

Usage: python3 -I rounding_run.py PROGRAM MODE(n|f|c|d|u) ENVELOPE TRANSCRIPT_OUT
The iv context is untouched; only mp (round-to-nearest) arithmetic changes direction.
"""
import importlib.util, sys
prog, rmode, env, out = sys.argv[1:5]
spec = importlib.util.spec_from_file_location('h3_band_v2_probe', prog)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.mp._prec_rounding[1] = rmode
rc = m.main([env, '--transcript', out])
print('ROUNDING', rmode, env, 'rc', rc, 'final mp rounding', m.mp._prec_rounding)
sys.exit(rc)
