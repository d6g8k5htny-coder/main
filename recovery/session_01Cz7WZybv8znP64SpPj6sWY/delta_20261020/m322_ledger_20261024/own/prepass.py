#!/usr/bin/env python3
"""Heuristic pre-pass over workflow YAML text (read as data, never executed): counts of
returncode uses, stdout/stderr bindings, mutant markers, and lines using returncode."""
import json, os, re, sys, hashlib
d = sys.argv[1]; out = {}
for fn in sorted(os.listdir(d)):
    t = open(os.path.join(d, fn), encoding='utf-8', errors='replace').read(); raw = t.encode()
    lines = t.splitlines()
    rc = [(i + 1, l.strip()[:160]) for i, l in enumerate(lines) if 'returncode' in l]
    out[fn] = {
        'bytes': len(raw), 'blob': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest(),
        'n_returncode': len(rc), 'returncode_lines': rc[:40],
        'mutant_markers': len(re.findall(r'(?i)mutant|--mutant|MUTATION|\bM[1-9]\b', t)),
        'stdout_refs': len(re.findall(r'\.stdout', t)), 'stderr_refs': len(re.findall(r'\.stderr', t)),
        'stderr_compared': bool(re.search(r"\.stderr\s*(==|!=)|stderr\s*==\s*b?['\"]{2}|not\s+\w*\.stderr", t)),
        'stdout_compared': bool(re.search(r"\.stdout\s*(==|!=|\.strip\(\)\s*(==|!=))|json\.loads\(\w*\.stdout", t)),
        'rc_eq_1': len(re.findall(r'returncode\s*(==|!=)\s*1\b', t)), 'rc_eq_2': len(re.findall(r'returncode\s*(==|!=)\s*2\b', t)),
        'rc_eq_0': len(re.findall(r'returncode\s*(==|!=)\s*0\b', t)),
    }
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for fn, r in out.items():
    print('%-48s rc=%2d mut=%3d so=%2d se=%2d secmp=%d socmp=%d rc1=%d rc2=%d rc0=%d' % (fn[:48], r['n_returncode'], r['mutant_markers'], r['stdout_refs'], r['stderr_refs'], r['stderr_compared'], r['stdout_compared'], r['rc_eq_1'], r['rc_eq_2'], r['rc_eq_0']))
