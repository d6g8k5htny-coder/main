#!/usr/bin/python3 -I
"""C220-02 extractor: what Orca sent to speech, what speech-dispatcher played, aligned to driver steps.

usage: extract.py EVIDENCE_SCENARIO_DIR
writes speech.json and speech.txt into that directory.
Sources:
  orca-debug.txt      'SPEECH DISPATCHER: Speaking '<speak>...'' lines = exact strings Orca sent to speech-dispatcher
                      (SPEECH OUTPUT lines give the plain string; WEB SAY ALL lines mark Say All chunks)
  sdlog/speech-dispatcher.log  queue / start ('Module speak!') / end ('reporting end') / CANCEL, real-time paced sink
  driver.json         driver steps, DOM status-change log (MutationObserver), CDP events
"""
import html, json, os, re, sys, calendar, time

d = sys.argv[1]
run = open(os.path.join(d, 'run.txt')).read()
day = re.search(r'start (\d{4}-\d{2}-\d{2})T', run).group(1)
base = calendar.timegm(time.strptime(day, '%Y-%m-%d'))

def orca_t(hms):
    h, m, s = hms.split(':')
    return round((base + int(h) * 3600 + int(m) * 60 + float(s)) * 1000, 3)

def norm(x):
    return re.sub(r'\s+', ' ', x.replace('\u2026', '...')).strip()

def plain(ssml):
    s = re.sub(r'<mark name="[^"]*"/>', '', ssml)
    s = s.replace('<speak>', '').replace('</speak>', '')
    return html.unescape(s)

# ---------- Orca ----------
ts_re = re.compile(r'^(\d\d:\d\d:\d\d\.\d+) - (.*)$')
sent, interrupts, blocks = [], [], []
last_output = None; last_sayall = None; last_obj_event = None
with open(os.path.join(d, 'orca-debug.txt'), encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.rstrip('\n')
        mb = re.match(r'^vvvvv (.*) vvvvv$', line)
        if mb:
            blocks.append(mb.group(1)); continue
        me = re.match(r'^\^\^\^\^\^ (.*) \^\^\^\^\^$', line)
        if me:
            if blocks: blocks.pop()
            last_obj_event = None if not blocks else last_obj_event
            continue
        m = ts_re.match(line)
        if not m:
            continue
        t, body = orca_t(m.group(1)), m.group(2)
        if body.startswith('OBJECT EVENT: '):
            last_obj_event = body[len('OBJECT EVENT: '):][:200]
        elif body.startswith('SPEECH OUTPUT: '):
            last_output = (t, body)
        elif body.startswith('WEB SAY ALL: '):
            last_sayall = t
        elif body.startswith('DEFAULT: Interrupting presentation'):
            interrupts.append({'t': t, 'block': blocks[-1] if blocks else None})
        elif body.startswith("SPEECH DISPATCHER: Speaking '<speak>"):
            ssml = body[len("SPEECH DISPATCHER: Speaking '"):].rstrip()
            if ssml.endswith("'"): ssml = ssml[:-1]
            kind = 'message'
            if last_sayall and (not last_output or last_sayall > last_output[0]):
                kind = 'say-all'
            block = blocks[-1] if blocks else None
            sent.append({'t': t, 'text': plain(ssml), 'kind': kind, 'block': block,
                         'event': last_obj_event if block and block.startswith('PROCESS OBJECT EVENT') else None})
            last_output = None if kind == 'message' else last_output
            if kind == 'say-all': last_sayall = None

# ---------- speech-dispatcher ----------
sd_re = re.compile(r'^\[\w{3} (\w{3} +\d+ \d\d:\d\d:\d\d \d{4}) : (\d+)\] speechd: +(.*)$')
queued, played, cancels = [], [], []
cur = None; await_text = False
sdpath = os.path.join(d, 'sdlog', 'speech-dispatcher.log')
with open(sdpath, encoding='utf-8', errors='replace') as f:
    for line in f:
        m = sd_re.match(line.rstrip('\n'))
        if not m:
            continue
        t = (calendar.timegm(time.strptime(m.group(1), '%b %d %H:%M:%S %Y')) + int(m.group(2)) / 1e6) * 1000
        body = m.group(3)
        if body.startswith('Queueing message |'):
            mm = re.match(r'Queueing message \|(.*)\| with priority (\d+)', body)
            if mm: queued.append({'t': round(t, 3), 'text': plain(mm.group(1)), 'priority': int(mm.group(2))})
        elif body.endswith('DATA:|CANCEL self') or body.endswith('DATA:|STOP self'):
            cancels.append({'t': round(t, 3), 'cmd': body.split('|')[-1], 'playing': cur['text'] if cur else None})
            if cur: cur['cancel_t'] = round(t, 3)
        elif body == 'Module speak!':
            await_text = True
        elif await_text and body.startswith('Command sent to output module: |<speak>'):
            mm = re.search(r'\|(<speak>.*</speak>)\|', body)
            cur = {'start_t': round(t, 3), 'text': plain(mm.group(1) if mm else body.split('|', 1)[1])}
            played.append(cur); await_text = False
        elif body.startswith('speak_queue playback thread reporting end'):
            if cur: cur['end_t'] = round(t, 3); cur = None
        elif body.startswith('speak_queue') and 'stop' in body.lower() and 'thread' not in body.lower():
            if cur and 'cancel_t' in cur: cur.setdefault('stop_t', round(t, 3))

# Orca's debug log is buffered and can lose its last lines when Orca is stopped; any message that
# speech-dispatcher queued (SSIP SPEAK from Orca) without a matching Orca log line is added from the
# speech-dispatcher log, marked log_source='speechd-only'.
for u in sent:
    u['log_source'] = 'orca-debug'
qi = 0
for q in queued:
    hit = next((u for u in sent if not u.get('_q') and norm(u['text']) == norm(q['text']) and abs(u['t'] - q['t']) < 2000), None)
    if hit:
        hit['_q'] = True; hit['speechd_queued_t'] = q['t']
    else:
        sent.append({'t': q['t'], 'text': q['text'], 'kind': 'unknown (speechd-only)', 'block': None, 'event': None, 'log_source': 'speechd-only', 'speechd_queued_t': q['t']})
sent.sort(key=lambda u: u['t'])
for u in sent:
    u.pop('_q', None)

# ---------- driver ----------
drv = json.load(open(os.path.join(d, 'driver.json')))
steps = drv['steps']
windows = []
for i, s in enumerate(steps):
    start = s.get('action_t') or s.get('navigate_t') or (0 if i == 0 else s['t'])
    windows.append({'step': s['step'], 'action': s['action'], 'start': start, 'recorded_t': s['t'],
                    'widget_status': s['snapshot']['widget_status'], 'active': s['snapshot']['active']})
for i, w in enumerate(windows):
    w['end'] = windows[i + 1]['start'] if i + 1 < len(windows) else float('inf')
if windows: windows[0]['start'] = 0

dom = drv['dom_status_changes']
def origin(text):
    for c in reversed(dom):
        if c['text'] and norm(c['text']) == norm(text):
            return c['el']
    return None
def win(t):
    for w in windows:
        if w['start'] is not None and w['start'] <= t < w['end']:
            return w['step']
    return None

for u in sent:
    u['iso'] = time.strftime('%H:%M:%S', time.gmtime(u['t'] / 1000)) + f".{int(u['t'] % 1000):03d}"
    u['step'] = win(u['t']); u['dom_origin'] = origin(u['text'])
for p in played:
    p['step_started'] = win(p['start_t']); p['dom_origin'] = origin(p['text'])
    if 'end_t' in p: p['dur_s'] = round((p['end_t'] - p['start_t']) / 1000, 3)
    p['start_iso'] = time.strftime('%H:%M:%S', time.gmtime(p['start_t'] / 1000)) + f".{int(p['start_t'] % 1000):03d}"
# match each sent utterance to its playback (FIFO by identical text)
used = set()
for u in sent:
    for j, p in enumerate(played):
        if j in used or norm(p['text']) != norm(u['text']) or p['start_t'] + 1 < u['t']:
            continue
        used.add(j); u['played_start_t'] = p['start_t']; u['played_end_t'] = p.get('end_t')
        u['cancelled_during_playback'] = 'cancel_t' in p
        u['queue_delay_s'] = round((p['start_t'] - u['t']) / 1000, 3)
        break
    else:
        u['played_start_t'] = None

out = {'windows': windows, 'orca_sent': sent, 'orca_interrupts': interrupts, 'speechd_queued_count': len(queued),
       'speechd_played': played, 'speechd_cancels': cancels, 'dom_status_changes': dom}
json.dump(out, open(os.path.join(d, 'speech.json'), 'w'), indent=1, default=str)

def fmt(ms):
    return time.strftime('%H:%M:%S', time.gmtime(ms / 1000)) + f".{int(ms % 1000):03d}" if ms else '-'
lines = []
for w in windows:
    lines.append(f"== step {w['step']}: {w['action']}  [from {fmt(w['start']) if w['start'] else 'start'}]  DOM status after: {w['widget_status']!r}  active={w['active']}")
    for c in dom:
        if w['start'] is not None and w['start'] <= c['t'] < w['end']:
            lines.append(f"   DOM {fmt(c['t'])} {c['el']}: {c['text']!r}")
    for u in sent:
        if u['step'] == w['step']:
            pl = f"played {fmt(u['played_start_t'])}-{fmt(u.get('played_end_t'))} (+{u.get('queue_delay_s')}s)" if u.get('played_start_t') else 'NOT PLAYED'
            if u.get('cancelled_during_playback'): pl += ' CANCELLED'
            lines.append(f"   SENT {u['iso']} [{u['kind']}|{(u['block'] or '-')[:40]}|{u['dom_origin'] or '-'}] {u['text']!r}  -> {pl}")
lines.append(f"interrupts (DEFAULT: Interrupting presentation): {[fmt(i['t']) for i in interrupts]}")
lines.append(f"speechd CANCEL commands: {[(fmt(c['t']), c['playing']) for c in cancels]}")
open(os.path.join(d, 'speech.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
