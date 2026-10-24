import sys
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root + '/experiments/universality/two_parameter_cusp_population')
import run_population_certificate as r
res = r.run(output_dir=out,
            performer='Anthropic Claude, session session_01S16677s1SGeAJmN3QMz4ip (Claude S), Claude Code cloud container, Linux x86_64',
            exposure='Source-exposed Anthropic reader/test author of main#343 (A1 finding, v2/v3 tests); not the runner or production author; organizational independence credit 0')
print(res['status'], res['actual_route_evaluations'], res['runtime_seconds'], res['source_drift'])
