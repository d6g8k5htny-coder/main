#!/usr/bin/env python3
"""Extract built-in Orca speech records, retaining phase/time/raw line provenance."""
import argparse, datetime, html, json, pathlib, re
ap=argparse.ArgumentParser(); ap.add_argument('run',type=pathlib.Path); args=ap.parse_args()
run=args.run
actions=[json.loads(x) for x in (run/'actions.jsonl').read_text().splitlines()]
phases=[(x['time'],x['name']) for x in actions if x['action']=='phase']
date=datetime.date.fromisoformat(run.name[:4]+'-'+run.name[4:6]+'-'+run.name[6:8])
records=[]
for n,line in enumerate((run/'orca-debug.log').read_text(errors='replace').splitlines(),1):
    if 'SPEECH OUTPUT:' not in line and 'SPEECH DISPATCHER: Speaking \'<speak>' not in line: continue
    match=re.match(r'(\d\d:\d\d:\d\d\.\d+) - (.*)',line)
    if not match: continue
    stamp=datetime.datetime.combine(date,datetime.time.fromisoformat(match[1]),datetime.timezone.utc).timestamp()
    phase='startup'
    for start,name in phases:
        if stamp>=start: phase=name
    kind='speech-output' if 'SPEECH OUTPUT:' in line else 'dispatcher-ssml'
    text=match[2]
    if kind=='dispatcher-ssml':
        text=text.split("Speaking '",1)[1].rsplit("'",1)[0]
        text=html.unescape(re.sub(r'<[^>]*>','',text))
    records.append(dict(time=stamp,phase=phase,kind=kind,text=text,line=n))
(run/'speech-transcript.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
report=[]
for phase in dict.fromkeys(r['phase'] for r in records):
    report.append('\n'+phase+'\n')
    for r in records:
        if r['phase']==phase and r['kind']=='dispatcher-ssml':
            report.append(f"{r['line']}: {r['text']}")
(run/'speech-transcript.txt').write_text('\n'.join(report)+'\n')
try:
    import array, math
    pcm=array.array('h'); pcm.frombytes((run/'audio-s16le-22050-mono.raw').read_bytes())
    audio=dict(format='signed 16-bit little-endian mono 22050 Hz',samples=len(pcm),
               duration_seconds=len(pcm)/22050,peak=max(map(abs,pcm),default=0),
               nonzero_samples=sum(x!=0 for x in pcm),human_audition=False)
    (run/'audio-statistics.json').write_text(json.dumps(audio,indent=2))
    print(json.dumps(audio))
except FileNotFoundError: pass
print('\n'.join(report))
