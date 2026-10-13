import hashlib, json, pathlib, sys
root = pathlib.Path('.').resolve(); repo = root.parents[1]
sys.path.insert(0, str(root)); import gate
files = sorted(p.name for p in root.iterdir() if p.is_file() and p.name != 'MANIFEST.json')
blob = lambda d: hashlib.sha1(b'blob %d\0' % len(d) + d).hexdigest()
lock = json.loads((root/'lake-manifest.json').read_text())
targets = gate.declarations((root/'CapFirstExit.lean').read_text())
ax = json.load(open('<SCRATCH>/v21m/axioms.json'))
assert sorted(ax) == sorted(targets)
pins = ['reviews/d1_cap_elder_partner_second_pass_claude_20260930/REVIEW.md', 'imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md']
m = {
 'object': 'CAP-FIRST-EXIT-LEAN-20261002-v1',
 'scientific_effect': 'NONE',
 'mathematical_acceptance': False,
 'alignment_review': 'NOT_CLAIMED',
 'module': 'CapFirstExit',
 'toolchain': (root/'lean-toolchain').read_text().strip(),
 'mathlib_rev': next(p['rev'] for p in lock['packages'] if p['name']=='mathlib'),
 'dependency_revisions': {p['name']: p['rev'] for p in lock['packages']},
 'pins_on_main': [{'path': p, 'git_blob': blob((repo/p).read_bytes())} for p in pins],
 'allowed_axioms': ['Classical.choice', 'Quot.sound', 'propext'],
 'targets': targets,
 'expected_axioms': {t: ax[t] for t in targets},
 'files': [{'path': f, 'bytes': len((root/f).read_bytes()), 'sha256': hashlib.sha256((root/f).read_bytes()).hexdigest()} for f in files],
}
(root/'MANIFEST.json').write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n')
print(files); print(m['pins_on_main']); print(m['dependency_revisions'])
