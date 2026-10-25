import json,sys,re
from pathlib import PurePosixPath
S='<SCRATCH>'
src=open(S+'/dbx/classify_hash.py').read(); line=[l for l in src.splitlines() if l.startswith('PERSONAL=')][0]
ns={'re':re}; exec(line, ns); PERSONAL=ns['PERSONAL']
b=json.load(open(S+'/dbx/phase2_batches.json')); done=json.load(open(S+'/dbx/content_hashes.json'))
todo=[]
for x in b:
    keep=[p for p in x if p.split('//',1)[1] not in done and not PERSONAL.search(PurePosixPath(p).name)]
    if keep: todo.append(keep)
skipped=sum(1 for x in b for p in x if PERSONAL.search(PurePosixPath(p).name))
print('batches left',len(todo),'personal-name paths skipped',skipped, file=sys.stderr)
for x in todo[:int(sys.argv[1])]: print(json.dumps(x, ensure_ascii=False))
