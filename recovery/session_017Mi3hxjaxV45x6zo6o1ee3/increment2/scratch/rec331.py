# Reconcile PR 331 capture evidence against the replay transcript, the 62 frozen contracts, the 8 baselines and the negative control.
import ast, hashlib, json, pathlib, re, sys
W=pathlib.Path(sys.argv[1]); E=pathlib.Path(sys.argv[2]); OUT=pathlib.Path(sys.argv[3])
pk=W/'frontiers/qs_d3_soft_layer_chain_20261005'
src=(pk/'replay.py').read_text()
RC=ast.literal_eval(re.search(r"REJECTION_CONTRACTS = (\{.*?\n\})\n", src, re.S).group(1))
EMPTY=hashlib.sha256(b'').hexdigest(); NEG=(1,EMPTY,hashlib.sha256(b'FAILED: X9_hardgap\n').hexdigest())
man=json.loads((pk/'SOURCES.json').read_text())
baselines={}
for o in man['objects']:
    for r in o['replays']:
        baselines[(r['script'],tuple(r['args']))]=hashlib.sha256((pk/r['stdout']).read_bytes()).hexdigest()
recs=[]
for d in sorted(E.glob('invocation-*')):
    rec=json.loads((d/'process.json').read_bytes())
    for name in ('stdout','stderr'):
        data=(d/(name+'.bin')).read_bytes(); st=rec['streams'][name]
        assert st['available'] and st['saved'] and st['bytes']==len(data) and st['sha256']==hashlib.sha256(data).hexdigest(), (d,name)
    assert rec['schema']==1 and rec['status']=='completed' and rec['exception'] is None and rec['timeout_seconds']==1800 and rec['evidence_errors']==[] and rec['scientific_effect']=='NONE', d
    cmd=rec['command']; script=pathlib.Path(cmd[cmd.index(next(c for c in cmd if c.endswith('.py') and 'author_controls' in c or c.endswith('addendum_control_k5o.py')))])
    assert rec['cwd']==str(script.parent)
    flags=[c for c in cmd[1:] if c.startswith('-')][:2] if cmd[1].startswith('-') else []
    recs.append((script,tuple(cmd[cmd.index(str(script))+1:]),rec['returncode'],rec['streams']['stdout']['sha256'],rec['streams']['stderr']['sha256'],cmd))
print('records',len(recs),'streams',sum(1 for d in E.glob('invocation-*/*.bin')))
# classify
rej=bas=neg=0; errors=[]
for script,args,code,so,se,cmd in recs:
    rel=None
    try: rel=str(script.resolve().relative_to(pk.resolve()))
    except ValueError: rel=None
    if rel is None:
        # negative control runs on a temp copy: path ends with A3_6/author_controls.py
        assert str(script).endswith('A3_6/author_controls.py') and args==(), (script,args)
        ok=(code,so,se)==NEG; neg+=1
        if not ok: errors.append(('negative',script,code,so[:12],se[:12]))
        continue
    key=(rel,args)
    if key in RC:
        exp=RC[key]; rej+=1
        if (code,so,se)!=exp[:3]: errors.append(('rejection',key,code,so[:12],se[:12],exp[:3]))
    elif key in baselines:
        bas+=1
        if not (code==0 and se==EMPTY and so==baselines[key]): errors.append(('baseline',key,code,so[:12],se[:12]))
    else:
        errors.append(('unexpected invocation',key))
print('baselines',bas,'rejections',rej,'negative',neg,'| contract set covered:', rej==len(RC) and len({(str(s.resolve().relative_to(pk.resolve())) if s.resolve().is_relative_to(pk.resolve()) else None,a) for s,a,*_ in recs}&set(RC))==len(RC))
# transcript reconciliation: report_control JSON lines in replay stdout
lines=[json.loads(l) for l in OUT.read_text().splitlines() if l.startswith('{')]
tr={(l['script'],tuple(l['args'])):(l['returncode'],l['stdout_sha256'],l['stderr_sha256']) for l in lines}
print('transcript control lines',len(lines),'(expect 71)')
mism=0
for script,args,code,so,se,cmd in recs:
    s=script.resolve()
    rel=str(s.relative_to(pk.resolve())) if s.is_relative_to(pk.resolve()) else 'A3_6/author_controls.py'
    t=tr.get((rel,args))
    if t is None or t!=(code,so,se): mism+=1; errors.append(('transcript',rel,args,t,(code,so[:12],se[:12])))
print('transcript vs records mismatches',mism)
# interpreter flags per mode
modes={tuple(c for c in cmd[1:3]) for *_,cmd in recs}
print('child flag sets observed:',modes)
print('ERRORS',len(errors)); [print('  ',e) for e in errors[:10]]
print('final line:',OUT.read_text().strip().splitlines()[-1])
