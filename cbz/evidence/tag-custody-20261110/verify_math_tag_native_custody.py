"""Verify the fixed, observed five Math tags and 171 supplied source bindings.

Scope is the supplied path set and exact Git object identities; this does not
prove completeness of the independent changed-path enumeration or any theorem.
"""
from collections import Counter
import json
import pathlib
import sys
import tarfile
from tag_custody_verifier_common import (check_binding, check_commits, check_members, check_source,
                                         digest, get, git_oid, read_objects, require, safe_path)

EXPECTED = {
    'refs/tags/cbz-quarantine-q-7522': ('f55f2304f4788b759482eee8e3fc70f01211ab82', '223293d9d26a7a4158026a4225523e8a683a8794', 17),
    'refs/tags/cbz-quarantine-q-7523': ('e30449ce79444baaf6607d03659f3c69db19be9a', 'de872d97fa54eb11c0834b8f532543222900d165', 25),
    'refs/tags/cbz-quarantine-q-7524': ('adc5c7c9219999b7e952b796b4fb72601c51bcd5', '5a30296bfc6c9c886d483c87e80e2eb2d9ea047d', 34),
    'refs/tags/cbz-quarantine-q-7525': ('e7a6918da1ef61402fd412de9d4e5d2a39456fdd', 'fc5ecb67f4a9d6cfb1dec066499ca2daa1147b3b', 43),
    'refs/tags/cbz-quarantine-q-7526': ('86651fa51da980dea8321f8a7b9c3626116b70bf', 'fdddc7e6b1fd584fa566beaccedd64d177104d10', 52),
}

with tarfile.open(pathlib.Path(sys.argv[1]), 'r:gz') as tar:
    names = check_members(tar)
    manifest = json.loads(get(tar, 'MANIFEST.json'))
    require(manifest['schema'] == 'cbz.historical-math-tags-native-custody.v2', 'wrong schema')
    files, tags, bindings = manifest['files'], manifest['tag_objects'], manifest['tree_bindings']
    require(len(files) == 52 and len(tags) == 5 and len(bindings) == 171, 'unexpected record count')
    actual_tags = {r['ref']: (r['tag_object'], r['target_commit']) for r in tags}
    require(len(actual_tags) == 5 and actual_tags == {ref: (v[0], v[1]) for ref, v in EXPECTED.items()},
            'tag identity mismatch')
    require(all(r['file'] == r['ref'].rsplit('/', 1)[-1] + '.tag' for r in tags), 'tag path mismatch')
    require(all(safe_path(r['path']) for r in files), 'invalid source path')
    expected_members = ({'MANIFEST.json'} | {'sources/' + r['path'] for r in files}
                        | {'tags/' + r['file'] for r in tags}
                        | {'git-objects/' + r['kind'] + '/' + r['sha'] for r in manifest['native_objects']})
    require(names == expected_members, 'unexpected or missing archive members')
    objects = read_objects(tar, manifest['native_objects'])
    check_commits(objects, manifest['commit_trees'], [row[1] for row in EXPECTED.values()])
    for row in tags:
        raw = get(tar, 'tags/' + row['file'])
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'] and git_oid('tag', raw) == row['tag_object'],
                'tag object mismatch')
        require(raw.splitlines()[:3] == [('object ' + row['target_commit']).encode(), b'type commit',
                                         ('tag ' + row['ref'].rsplit('/', 1)[-1]).encode()], 'tag header mismatch')
    by_path = {row['path']: row for row in files}
    require(len(by_path) == 52, 'duplicate source path')
    keys = [(row['tag'], row['commit'], row['path']) for row in bindings]
    require(len(set(keys)) == 171, 'duplicate binding')
    require(Counter(row['tag'] for row in bindings) == {ref: value[2] for ref, value in EXPECTED.items()},
            'unexpected tag binding counts')
    require({row['path'] for row in bindings} == set(by_path), 'source and binding paths differ')
    for row in files:
        check_source(tar, 'sources/' + row['path'], row)
    for row in bindings:
        require(row['tag'] in EXPECTED and EXPECTED[row['tag']][1] == row['commit'], 'binding tag mismatch')
        source = by_path[row['path']]
        require(source['git_blob'] == row['git_blob'] and source['mode'] == row['mode'], 'binding source mismatch')
        check_binding(objects, manifest['commit_trees'], row)
    print('PASS: 5 fixed tag objects, 5 commits,', sum(kind == 'tree' for kind, _ in objects),
          'path trees, 52 files and 171 supplied bindings; source custody only')
