import json, sys, subprocess, multiprocessing, time
P='/home/user/wt_d3/frontiers/c8_theorem_a_constants_d3_20261001/theorem_a3.py'
def run(args):
    r, b, k, n, seed = args
    t=time.time()
    out=subprocess.run([sys.executable, '-B', '-S', P, '--mc', r, b, k, str(n), str(seed)], capture_output=True, text=True)
    return args, out.stdout.strip(), time.time()-t
if __name__=='__main__':
    jobs=[('1/1024','0','2',40000,s) for s in range(211,219)] + [('1/512','0','2',20000,s) for s in range(111,119)]
    with multiprocessing.Pool(4) as p:
        for args, out, dt in p.imap_unordered(run, jobs):
            print(json.dumps({'args': args, 'out': out, 'sec': round(dt)}), flush=True)
