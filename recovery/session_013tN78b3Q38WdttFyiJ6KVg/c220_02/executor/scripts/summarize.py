#!/usr/bin/python3 -I
"""Compact per-step summary across scenarios (reads evidence/<id>/speech.json, driver.json).
usage: summarize.py EVIDENCE_DIR ID [ID...]  -> writes EVIDENCE_DIR/summary.json and prints it."""
import hashlib, json, os, sys, time

ev = sys.argv[1]
def fmt(ms):
    return (time.strftime('%H:%M:%S', time.gmtime(ms / 1000)) + f".{int(ms % 1000):03d}") if ms else None
def label(u):
    o = u.get('dom_origin')
    if o and o.startswith('widget:') and u.get('log_source') != 'speechd-only':
        return 'WIDGET live region'
    if o in ('#museum-state', '#conditional-route-status', '.current-state'):
        return f'outer status {o} (C201-AT-01 scope)'
    if u['kind'] == 'say-all':
        return 'Say All (document reading)'
    b = u.get('block') or ''
    ev = u.get('event') or ''
    if u['text'] == 'Screen reader on.':
        return 'Orca startup'
    if 'window:activate' in ev or 'window:activate' in b:
        return 'window activation'
    if u.get('log_source') == 'speechd-only':
        base = 'WIDGET live region' if (u.get('dom_origin') or '').startswith('widget:') else 'unattributed'
        return base + '; from speech-dispatcher log only (Orca debug log tail lost at shutdown)'
    if (u.get('block') or '').startswith("PROCESS ATSPI_KEY_PRESSED_EVENT"):
        return 'Orca key echo'
    if 'state-changed:busy' in ev:
        return 'Orca page-load message'
    if b.startswith('PRESENT LIVE REGION MESSAGE') and u['text'] == '2026-10-13T14:56:55Z':
        return 'outer status: <time> inside .current-state (C201-AT-01 scope)'
    if b.startswith('PRESENT LIVE REGION MESSAGE'):
        return 'other live region'
    if 'focused' in b or 'state-changed:focused' in (u.get('event') or ''):
        return 'focus presentation'
    if 'property-change:accessible-name' in (u.get('event') or '') or 'property-cha' in b:
        return 'name change of focused control'
    if 'state-changed:pressed' in (u.get('event') or '') or 'state-changed:pressed' in b:
        return 'pressed-state change of focused control'
    return (u.get('event') or b or 'other')[:80]
out = {}
for sid in sys.argv[2:]:
    d = os.path.join(ev, sid)
    sp = json.load(open(os.path.join(d, 'speech.json')))
    steps = []
    for w in sp['windows']:
        spoken = []
        for u in sp['orca_sent']:
            if u['step'] != w['step']:
                continue
            played = f"played {fmt(u['played_start_t'])}" if u.get('played_start_t') else 'not played before run end'
            if u.get('cancelled_during_playback'):
                played += ' (CANCEL during playback)'
            spoken.append(f"sent {fmt(u['t'])}, {played} [{label(u)}] {u['text']}")
        compact = [x for x in spoken if '[Say All' not in x]
        n_sa = len(spoken) - len(compact)
        if n_sa:
            compact.append(f"[Say All] {n_sa} document-reading chunk(s) also sent in this step (listed in speech.txt)")
        steps.append({'step': w['step'], 'action': w['action'], 'dom_status': w['widget_status'], 'active': w['active'], 'spoken': compact, 'spoken_all': spoken})
    sha = hashlib.sha256(open(os.path.join(d, 'orca-debug.txt'), 'rb').read()).hexdigest()
    out[sid] = {'steps': steps, 'orca_log_sha256': sha, 'cancels': sp['speechd_cancels'],
                'interrupts': [fmt(i['t']) for i in sp['orca_interrupts']],
                'dom_changes': [f"{fmt(c['t'])} {c['el']}: {c['text']}" for c in sp['dom_status_changes']]}
json.dump(out, open(os.path.join(ev, 'summary.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
