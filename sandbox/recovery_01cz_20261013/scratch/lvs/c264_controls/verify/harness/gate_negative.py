"""Apply check.py's negative() (loaded unmodified, no bytecode written) to the verifier's control logs."""
import importlib.util, json, sys
sys.dont_write_bytecode = True
check, reported, vdir = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('torus_check_v', check)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
r = json.load(open(reported, encoding='utf-8'))
c.FALSE_CONTROLS = dict(c.FALSE_CONTROLS, **{k: (v['proposition'], v['tactic']) for k, v in r.items()})
c.CONTROLS = tuple(c.FALSE_CONTROLS) + tuple(c.INJECTIONS)
for name, v in r.items():
    log = open(f'{vdir}/controls/{name}.log', encoding='utf-8').read()
    status = int(open(f'{vdir}/controls/{name}.status').read().strip())
    print(name, 'status', status, 'errors', log.count(': error'), 'warnings', log.count(': warning'),
          'gate:', c.negative(log, name, status),
          'output_matches_reported_modulo_dir:', log.replace('/verify/controls/', '/controls/') == v['lean_output'])
