# Run indep_ht.collide with mp.prec set to an explicit bit count (not via dps).
# usage: python run_prec.py <src_dir> <prec_bits> <c910|c3435>
import sys, time, importlib.util
src, bits, what = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location('indep_ht', src + '/indep_ht.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
mp = m.mp
mp.mp.prec = bits
print('#prec', mp.mp.prec, 'dps', mp.mp.dps, flush=True)
t0 = time.time()
if what == 'c910':
    m.collide(mp.mpf('-1.878'), mp.mpf('99.33'))
elif what == 'c3435':
    m.collide(mp.mpf('-0.3846'), mp.mpf('223.276'))
print('time', time.time() - t0)
