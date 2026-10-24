import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'root/experiments/universality/two_parameter_cusp_population'))
import run_population_certificate as runner
t = time.monotonic()
rec = runner.run(output_dir=HERE / 'attempt')
print(rec['status'], rec['exit_code'], rec['actual_route_evaluations'], rec['runtime_seconds'], rec['full_declared_campaign'])
