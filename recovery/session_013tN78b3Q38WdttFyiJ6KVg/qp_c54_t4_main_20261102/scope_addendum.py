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
    'method': 'read by hand (all six cells); no agent',
}, {
    'object': 'audits/dropbox_import/2026-09-27', 'item_id': 'drive-manifest-part01-txt', 'classification': 'NOT_OBLIGATION',
    'title': 'Condensed Drive manifest, part 01 of 13 (text export): file-title index',
    'files': ['audits/dropbox_import/2026-09-27/files/Github/Drive manifest (condensed, all 964 files flags  folder  title) \u2014 '
              'Part 01 of 13 \u2014 canonical impact NONE (1).pdf.txt'],
    'reason': ('Dropped by main_dirs.py, whose ls-tree parsing without -z C-quotes paths containing U+2014. Read by hand (115 lines): '
               'line 2 calls it a condensed index of all 964 Drive files, and lines 3-4 give the format FLAGS | folder | title and say '
               'it is additive with canonical impact NONE. Every later line is a flag, a folder and a file title. File titles that '
               'name theorems or lemmas are titles only; the file states no mathematics of its own.'),
    'refers_to': [],
    'method': 'read by hand (all 115 lines); no agent',
}, {
    'object': 'audits/dropbox_import/2026-09-27', 'item_id': 'verified-duplicate-log-txt', 'classification': 'NOT_OBLIGATION',
    'title': 'Verified duplicate log for the 02_GAUSSIAN_TRANSVERSALITY same-name set (text export): custody record',
    'files': ['audits/dropbox_import/2026-09-27/files/Github/VERIFIED DUPLICATE LOG \u2014 02_GAUSSIAN_TRANSVERSALITY SAME-NAME '
              'SET (1).pdf.txt'],
    'reason': ('Dropped by main_dirs.py for the same reason. Read by hand (90 lines): it records that three Drive files named '
               '02_GAUSSIAN_TRANSVERSALITY.md were compared by byte count, SHA-256 and pairwise byte equality, were exact duplicates, '
               'and were renamed and kept, with no deletion (sections 1-4), plus a reopening rule (section 6). It is a custody record '
               'with no mathematical claim; the Drive ids it lists are not quoted here.'),
    'refers_to': [],
    'method': 'read by hand (all 90 lines); no agent',
}]
main_clone, out = sys.argv[1:3]
rows = []
for r in ROWS:
    files = []
    for p in r['files']:
        data = subprocess.run(['git', '-C', main_clone, 'cat-file', 'blob', f'{REF}:{p}'], capture_output=True, check=True).stdout
        blob = subprocess.run(['git', '-C', main_clone, 'rev-parse', f'{REF}:{p}'], capture_output=True, text=True, check=True).stdout.strip()
        files.append({'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    row = {k: v for k, v in r.items() if k not in ('files', 'method')}
    row['source_binding'] = {'repository': 'd6g8k5htny-coder/main', 'commit': REF, 'files': files}
    row['verification'] = {'ok': True, 'problems_fixed': [], 'method': r['method']}
    rows.append(row)
with open(out, 'x') as f:
    json.dump({'schema': 'claude54.t4-scope-addendum.v1', 'main_ref': REF,
               'scope': ('main files at REF that the census did not list: the one file outside its file types that could carry mathematics, '
                         'and the two census-type files that its ls-tree parsing dropped because their paths contain U+2014'),
               'rows': rows}, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(len(rows), 'rows')
