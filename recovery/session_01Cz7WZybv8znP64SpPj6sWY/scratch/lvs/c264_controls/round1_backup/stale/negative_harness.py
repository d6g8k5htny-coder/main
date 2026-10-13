"""Apply the gate's own negative() to control logs; check.py is loaded unmodified, the three controls added in memory."""
import importlib.util, json, sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('torus_check', sys.argv[1])
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
controls = json.load(open(sys.argv[2], encoding='utf-8'))
c.FALSE_CONTROLS = dict(c.FALSE_CONTROLS, **{k: tuple(v) for k, v in controls.items()})
c.CONTROLS = tuple(c.FALSE_CONTROLS) + tuple(c.INJECTIONS)
for spec_ in sys.argv[3:]:
    control, log, status = spec_.split(':')
    try:
        print(control, log.rsplit('/', 1)[-1], c.negative(open(log, encoding='utf-8').read(), control, int(open(status).read().strip())))
    except ValueError as e:
        print(control, log.rsplit('/', 1)[-1], 'REFUSED:', e)
