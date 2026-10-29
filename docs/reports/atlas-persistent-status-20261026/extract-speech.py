from pathlib import Path
import json,re,html,datetime,sys,hashlib
run=Path(sys.argv[1]);actions=[json.loads(x) for x in (run/'actions.jsonl').read_text().splitlines()];phases=[(a['time'],a['name']) for a in actions if a['action']=='phase'];date=datetime.datetime.strptime(run.name[:8],'%Y%m%d').date();records=[]
for n,line in enumerate((run/'orca-debug.log').read_text(errors='replace').splitlines(),1):
 if 'SPEECH DISPATCHER: Speaking ' not in line and 'SPEECH OUTPUT: ' not in line:continue
 stamp=datetime.datetime.combine(date,datetime.time.fromisoformat(line[:15].strip()),datetime.timezone.utc).timestamp();phase='startup'
 for t,name in phases:
  if stamp>=t:phase=name
 if 'SPEECH DISPATCHER: Speaking ' in line:
  raw=line.split('Speaking ',1)[1];text=raw[1:raw.rfind("'")];kind='dispatcher-ssml' if '<speak' in raw else 'dispatcher-string';text=html.unescape(re.sub(r'<[^>]*>','',text))
 else: text=line.split('SPEECH OUTPUT: ',1)[1].rsplit("' {",1)[0].lstrip("'");kind='speech-output'
 records.append({'time':stamp,'phase':phase,'kind':kind,'text':text,'raw_log_line':n})
(run/'speech-extracted.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n');out={}
for r in records:
 if r['phase'] not in ['startup','finished'] and r['kind'].startswith('dispatcher'):out.setdefault(r['phase'],[]).append(r['text'])
for k,v in out.items():print(k, v[:9])
