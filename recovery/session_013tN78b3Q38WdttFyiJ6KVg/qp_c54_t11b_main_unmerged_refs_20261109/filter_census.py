#!/usr/bin/env python3
"""Reduce the main history census (pr_history_census.py output) to the file entries t11b_objects.py can use: paths outside root
and the excluded directories, with a mathematical extension. Every ref row is kept. t11b_objects.py gives byte-identical output on
the reduced census, because it applies the same filter. Usage: python3 -I -B filter_census.py CENSUS_IN CENSUS_OUT"""
import json, re, sys
EXCL = ('cbz/', 'tests/', 'tools/', '.github/', 'docs/site/', 'docs/retro/', 'governance/', 'ctx/', 'history/', 'interface/',
        'records/', 'architecture/', 'data/', 'recovery/', 'drive/', 'incoming/')
MATHY = re.compile(r'\.(md|tex|py|json|lean|txt)$')
d = json.load(open(sys.argv[1]))
for r in d['prs']:
    r['files'] = [f for f in r['files'] if not f['path'].startswith(EXCL) and '/' in f['path'] and MATHY.search(f['path'])]
    r['files_never_in_main_history_math_areas'] = len(r['files'])
d['schema'] += '+filtered-to-t11b-math-areas'
json.dump(d, open(sys.argv[2], 'x'), indent=1)
print(sum(len(r['files']) for r in d['prs']), 'file entries kept on', len(d['prs']), 'refs')
