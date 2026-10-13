import sys
root=sys.argv[1]; out=sys.argv[2]
sys.path.insert(0, root+'/experiments/universality/two_parameter_cusp_population')
import run_population_certificate as r
res=r.run(output_dir=out)
print(res['status'],res['actual_route_evaluations'],res['runtime_seconds'],res['source_drift'])
