#!/usr/bin/env python3
"""Accounting rows for main files outside the tranche-4 census's file types (.md .tex .py .json .lean .txt), classified by hand
after reading. Bindings come from Git. Usage: python3 -I -B scope_addendum.py MAIN_CLONE OUTPUT"""
import hashlib, json, subprocess, sys
REF = 'd2b0216e15d545c47f25b6a9265dd3298971e9fe'
ROWS = [{
    'object': 'docs/notebooks', 'item_id': 'side24-public-coefficients-notebook', 'classification': 'COPY_OR_EXPOSITION_OF',
    'title': 'SIDE24 public coefficient viewer notebook (NON-CERTIFYING display of published endpoints)',
    'files': ['docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb'],
    'reason': ('Cell 1 fetches Math-@9d7b6802424fb4715b31999066aafca8ee2f3cca:coefficients/side24_v1/ENCLOSURE.json and refuses it '
               'unless it is 1090 bytes with SHA-256 72b6cd92... and scientific_acceptance false. Cell 2 keeps the published decimal '
               'endpoint strings, checks 0 < lower < upper, and prints them; Decimal is used only for a display midpoint. Cell 4 draws '
               'an optional Matplotlib chart of float midpoints, labelled NON-CERTIFYING. Nothing is computed beyond the published '
               'numbers, so the notebook states no mathematics of its own.'),
    'refers_to': [{'ref': 'Math-@9d7b6802424fb4715b31999066aafca8ee2f3cca:coefficients/side24_v1/ENCLOSURE.json',
                   'dispositions': ['ADOPTED:Q-1170']}],
}]
main_clone, out = sys.argv[1:3]
rows = []
for r in ROWS:
    files = []
    for p in r['files']:
        data = subprocess.run(['git', '-C', main_clone, 'cat-file', 'blob', f'{REF}:{p}'], capture_output=True, check=True).stdout
        blob = subprocess.run(['git', '-C', main_clone, 'rev-parse', f'{REF}:{p}'], capture_output=True, text=True, check=True).stdout.strip()
        files.append({'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    row = {k: v for k, v in r.items() if k != 'files'}
    row['source_binding'] = {'repository': 'd6g8k5htny-coder/main', 'commit': REF, 'files': files}
    row['verification'] = {'ok': True, 'problems_fixed': [], 'method': 'read by hand (all six cells); no agent'}
    rows.append(row)
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.t4-scope-addendum.v1', 'main_ref': REF,
               'scope': 'main files at REF outside the census file types and engineering directories that could carry mathematics',
               'rows': rows}, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(len(rows), 'rows')
