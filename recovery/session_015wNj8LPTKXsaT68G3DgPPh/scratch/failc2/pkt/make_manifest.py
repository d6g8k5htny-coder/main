import hashlib, json, os
PK = '/home/user/wt_fail/frontiers/c8_elder_failure_coefficient_20261001'
files = sorted(f for f in os.listdir(PK) if f != 'SOURCE_FILES.json')
out = []
for f in files:
    data = open(os.path.join(PK, f), 'rb').read()
    out.append({'path': f, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'git_blob': hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()})
doc = {'schema': 1, 'scientific_effect': 'NONE', 'certified': True, 'review_status': 'AUTHOR_SIDE_CERTIFIED_NUMERICS',
       'object': 'CL-C8-ELDER-FAILURE-COEFFICIENT-20261001-v1', 'files': out}
with open(os.path.join(PK, 'SOURCE_FILES.json'), 'w') as fh:
    json.dump(doc, fh, indent=2)
    fh.write('\n')
print('\n'.join('%s %d' % (e['path'], e['bytes']) for e in out))
