#!/usr/bin/env python3
"""Extract what Orca sent to speech (Orca debug log) and when speech-dispatcher
started playing it (speechd level-5 log), grouped by driver step.
usage: python3 -I extract.py EVIDENCE_DIR   -> writes speech.txt and steps.json there
"""
import html, json, re, sys
from collections import deque
from pathlib import Path

ev = Path(sys.argv[1])
TS = re.compile(r'^(\d\d:\d\d:\d\d\.\d{6}) - (.*)$')

def norm(s):
    s = re.sub(r'<mark name="[^"]*"/>', '', s)
    s = re.sub(r'</?speak>', '', s)
    s = html.unescape(s)
    return ' '.join(s.split())

def loose(s):
    # speech-dispatcher's symbol preprocessing rewrites punctuation (e.g. an ellipsis
    # becomes '...'), so queue/playback records are matched on letters and digits only.
    return re.sub(r'[^0-9A-Za-z]+', '', s).lower()

# ---- Orca debug log -------------------------------------------------------
raw = ev.joinpath('orca-debug.txt').read_text(errors='replace').split('\n')
lines = []  # (time or None, text)
for ln in raw:
    m = TS.match(ln)
    if m:
        lines.append([m.group(1)[:12], m.group(2)])
    elif ln.startswith(' ' * 18) and lines and lines[-1][0]:
        lines[-1][1] += '\n' + ln[18:]
    else:
        lines.append([None, ln])

sent = []  # dicts
in_live = False
pending_regular = 0
last_event = ''
last_key = None
for t, text in lines:
    if text.startswith('vvvvv PRESENT LIVE REGION MESSAGE'):
        in_live = True; continue
    if text.startswith('^^^^^ PRESENT LIVE REGION MESSAGE'):
        in_live = False; continue
    if t and text.startswith('OBJECT EVENT: '):
        last_event = text[len('OBJECT EVENT: '):].split(' for ')[0]
        continue
    if t and text.startswith('KEYBOARD EVENT') or (t and 'KeyboardEvent' in text and 'pressed' in text):
        last_key = t
    if t and text.startswith('SPEECH OUTPUT: '):
        m = re.match(r"SPEECH OUTPUT: '(.*)'(?: \{.*\})?$", text, re.S)
        s = m.group(1) if m else text
        sent.append({'sent': t, 'text': s, 'live': in_live, 'event': last_event, 'kind': None})
        continue
    if t and text.startswith('SPEECH DISPATCHER: Speaking ') and (text.endswith(' as string') or text.endswith(' as char')):
        pending_regular += 1; continue
    if t and text.startswith("SPEECH DISPATCHER: Speaking '<speak>"):
        if pending_regular:
            pending_regular -= 1
            sent_ssml = text
            for d in reversed(sent):
                if 'ssml' not in d and d['kind'] is None:
                    d['ssml'] = norm(text[len("SPEECH DISPATCHER: Speaking '"):].rsplit("'", 1)[0]); break
        else:
            s = norm(text[len("SPEECH DISPATCHER: Speaking '"):].rsplit("'", 1)[0])
            sent.append({'sent': t, 'text': s, 'live': False, 'event': '', 'kind': 'Say All chunk', 'ssml': s})

# ---- widget texts (from the driver's DOM log) ----------------------------
drv = ev.joinpath('driver.log').read_text().split('\n')
widget_texts = set()
for ln in drv:
    m = re.match(r'^\S+ dom\((\S+)\): \[WIDGET status\] (".*")', ln)
    if m:
        widget_texts.add(json.loads(m.group(2)))

PAGELOAD = re.compile(r'^(Loading\.\s+Please wait\.|Finished loading .*|Page has .*|Loading .*)$')
for d in sent:
    if d['kind']:
        continue
    s = d['text']
    if d['live']:
        d['kind'] = 'WIDGET live region' if s in widget_texts else 'outer museum status (live region)'
    elif s in ('Screen reader on.', 'Screen reader off.'):
        d['kind'] = 'Orca startup/shutdown'
    elif PAGELOAD.match(s):
        d['kind'] = 'Orca page-load message'
    elif s.lower() in ('return', 'enter', 'kp enter'):
        d['kind'] = 'Orca key echo'
    else:
        d['kind'] = f'other (last AT-SPI event: {d["event"] or "none"})'

# ---- speech-dispatcher log: played times ---------------------------------
SDTS = re.compile(r'^\[\w{3} \w{3} +\d+ (\d\d:\d\d:\d\d) \d{4} : +(\d+)\] speechd: +(.*)$')
sdl = ev.joinpath('sdlog', 'speech-dispatcher.log').read_text(errors='replace').split('\n')
sd_events = []
i = 0
while i < len(sdl):
    m = SDTS.match(sdl[i])
    if not m:
        i += 1; continue
    t = f'{m.group(1)}.{int(m.group(2)):06d}'[:12]
    body = m.group(3)
    if body.startswith('Queueing message |') or body.startswith('Incoming text: |'):
        buf = body
        endpat = '| with priority' if body.startswith('Queueing') else None
        while True:
            if endpat and endpat in buf: break
            if not endpat and buf.rstrip().endswith('|') and buf.count('|') >= 2: break
            i += 1
            if i >= len(sdl): break
            buf += '\n' + sdl[i]
        if body.startswith('Queueing'):
            txt = buf[len('Queueing message |'):buf.rindex('| with priority')]
            sd_events.append(('Q', t, norm(txt)))
        else:
            txt = buf[len('Incoming text: |'):buf.rstrip().rindex('|')]
            sd_events.append(('P', t, norm(txt)))
    elif body.startswith('Command caught: "cancel"') or body.startswith('Command caught: "stop"'):
        sd_events.append(('C', t, ''))
    elif body.startswith('speak_queue playback thread got END'):
        sd_events.append(('E', t, ''))
    i += 1

queued = []  # list of dict(text, q, played, fate)
fifo = deque(); current = None
for kind, t, txt in sd_events:
    if kind == 'Q':
        e = {'text': txt, 'q': t, 'played': None, 'fate': 'queued at end of log'}
        queued.append(e); fifo.append(e)
    elif kind == 'P':
        while fifo and loose(fifo[0]['text']) != loose(txt):
            x = fifo.popleft(); x['fate'] = 'dropped (no playback record)'
        if fifo:
            current = fifo.popleft(); current['played'] = t; current['fate'] = 'played to end'
    elif kind == 'C':
        if current and current['fate'] == 'played to end' and current.get('ended') is None:
            current['fate'] = f'interrupted by cancel {t}'
        for x in fifo:
            x['fate'] = f'cancelled before playing {t}'
        fifo.clear(); current = None
    elif kind == 'E':
        if current is not None:
            current['ended'] = t
        current = None

# align Orca sends with speechd queue entries by order and text
j = 0
for d in sent:
    key = d.get('ssml', norm(d['text']))
    for k in range(j, min(j + 8, len(queued))):
        if loose(queued[k]['text']) == loose(key):
            d['played'] = queued[k]['played']; d['fate'] = queued[k]['fate']; j = k + 1; break
    else:
        d['played'] = None; d['fate'] = 'no speechd queue record matched'
    if d.get('fate') != 'no speechd queue record matched':
        queued[j - 1]['matched'] = True
# speechd messages with no Orca debug-log record (debug log damaged or truncated)
for q in queued:
    if not q.get('matched'):
        sent.append({'sent': q['q'], 'text': q['text'], 'live': False, 'event': '',
                     'kind': 'SPEECHD-ONLY (no Orca debug-log record)', 'played': q['played'], 'fate': q['fate']})
sent.sort(key=lambda d: d['sent'])

# ---- steps ---------------------------------------------------------------
steps = []
for ln in drv:
    m = re.match(r'^(\S+) STEP (\d+): (.*)$', ln)
    if m:
        steps.append({'step': int(m.group(2)), 'start': m.group(1), 'action': m.group(3), 'dom': [], 'spoken': []})
    m2 = re.match(r'^\S+ dom\((\S+)\): (.*)$', ln)
    if m2 and steps:
        steps[-1]['dom'].append(f'{m2.group(1)} {m2.group(2)}')
for d in sent:
    st = None
    for s in steps:
        if d['sent'] >= s['start']:
            st = s
    target = st if st else (steps[0] if steps else None)
    if target is not None:
        target['spoken'].append(d)

out = []
for s in steps:
    sa = [d for d in s['spoken'] if d['kind'] == 'Say All chunk']
    other = [d for d in s['spoken'] if d['kind'] != 'Say All chunk']
    out.append({'step': s['step'], 'start': s['start'], 'action': s['action'], 'dom': s['dom'],
                'spoken': [{k: d.get(k) for k in ('sent', 'played', 'kind', 'text', 'fate')} for d in other],
                'say_all_chunks': len(sa),
                'say_all_played': sum(1 for d in sa if d.get('played'))})
ev.joinpath('steps.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
with ev.joinpath('speech.txt').open('w') as f:
    for s in out:
        f.write(f"== STEP {s['step']} ({s['start']}): {s['action']}\n")
        for dline in s['dom']:
            f.write(f"   dom {dline}\n")
    f.write('\n== ALL SPEECH SENT BY ORCA (sent / played / kind / text / fate)\n')
    for d in sent:
        f.write(f"{d['sent']} | {d.get('played') or '-':12} | {d['kind']} | {d['text']} | {d.get('fate')}\n")
for s in out:
    print(f"STEP {s['step']} {s['start']} {s['action']}")
    for d in s['spoken']:
        print(f"   sent {d['sent']} played {d['played']} [{d['kind']}] {d['text'][:150]}  <{d['fate']}>")
    print(f"   [Say All] {s['say_all_chunks']} chunk(s) sent ({s['say_all_played']} reached playback)")
