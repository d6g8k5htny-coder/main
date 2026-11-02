#!/usr/bin/env python3
"""Manual-check aids for an assembled tranche file (records + other_items): NO_SCRIPTS consistency, an overclaim lint over
our-voice fields, and a privacy scan of the whole file. Prints findings for human review; exit code is 0 unless NO_SCRIPTS is
inconsistent or a privacy pattern hits."""
import json, re, sys
SCRIPT_EXT = ('.py', '.sage', '.jl', '.m', '.js', '.mjs', '.ts', '.c', '.cc', '.cpp', '.h', '.rs', '.sh', '.ipynb', '.r', '.R', '.go', '.java')
OUR_FIELDS = ('title', 'exact_boundary', 'blocking_reason', 'attempted', 'resumption_cost', 'layer_a_status',
              'classification_reason', 'floating_point', 'source_self_limits', 'finite_checks')
OVER = re.compile(r'\b(certif\w*|proven|proved|verified|verif(?:y|ies)|closed|establish\w*|rigorous\w*|guarantee\w*|discharg\w*|sound)\b', re.I)
NEG = re.compile(r'\b(not|no|never|unverified|without|cannot|nothing|neither|nor|un\w+|only|own|claims?|says?|states?|asserts?|calls?|labels?|reports?|presents?|would|until|before|whether)\b', re.I)
PRIV = {'email': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
        'gh_token': re.compile(r'\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
        'device_code': re.compile(r'\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{4}-(?=[A-Z0-9]*[A-Z0-9])[A-Z0-9]{4}\b(?<![0-9]{4}-[0-9]{4})'),
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'users_path': re.compile(r'/Users/[A-Za-z0-9._-]+'),
        'signed_url': re.compile(r'(X-Amz-Signature|sig=|se=20\d\d|token=)', re.I)}
data = json.load(open(sys.argv[1]))
raw = open(sys.argv[1], encoding='utf-8').read()
bad = 0
recs = data['records']
print(f'{len(recs)} records')
for r in recs:
    files = [f['path'] for f in r['source_binding']['files']]
    scripts = [p for p in files if p.endswith(SCRIPT_EXT) or p.endswith(tuple(e + '.txt' for e in SCRIPT_EXT))]
    fp = r['floating_point'].split()[0].rstrip(':;,.—-')
    if fp == 'NO_SCRIPTS' and scripts:
        bad += 1
        print('NO_SCRIPTS-INCONSISTENT', r['id'], scripts[:3])
    if fp != 'NO_SCRIPTS' and not scripts:
        print('info: no script but', fp, r['id'], '|', r['floating_point'][:140])
hits = []
QUOTED = re.compile(r"'[^'\n]{0,400}'|\"[^\"\n]{0,400}\"|`[^`\n]{0,200}`|\S+\.(?:json|md|py|txt|tex|lean|out)\b")
BENIGN = re.compile(r"closed[- ]form\w*|closed (?:set|superlevel|trap|interval|ball|under|with|curve|loop|subset|domain|by)|"
                    r"verified (?:interval|exact|rational|replay|checker|evaluation|enclosure|exp|inverse|bound|comparison|barcode)|"
                    r"needs? (?:a |an )?(?:certified|verified|rigorous)|(?:a|any) (?:rigorous|certified|verified)|"
                    r"certificate\w*|discharge (?:lives|of)|is certified (?:false|only by)", re.I)
VOICE = re.compile(r"\b(?:is|are|was|were|been|be|now|fully|thereby|hence)\s+(?:certified|proved|proven|verified|closed|established|"
                   r"discharged|sound|rigorous)\b|\b(?:this (?:item|record|object)|the (?:item|record)|we)\s+(?:certif|prov|verif|establish|discharg)\w*", re.I)
broad = 0
for r in recs + data.get('other_items', []):
    for k in OUR_FIELDS:
        v = r.get(k) or r.get('reason' if k == 'classification_reason' else k)
        if not isinstance(v, str):
            continue
        broad += len(OVER.findall(v))
        t = BENIGN.sub(' ', QUOTED.sub(' ', v))
        for m in VOICE.finditer(t):
            ctx = t[max(0, m.start() - 90):m.end() + 60]
            if not NEG.search(t[max(0, m.start() - 30):m.start()]):
                hits.append((r.get('id') or r.get('item_id'), k, ctx.replace('\n', ' ')))
print(f'overclaim lint: {broad} broad keyword hits; {len(hits)} our-voice assertions (quotes, file names and benign phrases removed) to read')
for h in hits:
    print('  OVER', h[0], h[1], '::', h[2])
for name, rx in PRIV.items():
    found = sorted(set(m.group(0) for m in rx.finditer(raw)))
    if found:
        print('PRIVACY', name, len(found), found[:8])
        if name in ('email', 'gh_token', 'ssn', 'signed_url'):
            bad += 1
sys.exit(1 if bad else 0)
