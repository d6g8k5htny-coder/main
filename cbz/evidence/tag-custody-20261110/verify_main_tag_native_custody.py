"""Verify the fixed, observed two Main tags and 11 supplied source bindings.

Scope is the supplied path set and exact Git object identities; this does not
prove completeness of the independent changed-path enumeration or any theorem.
"""
from collections import Counter
import json
import pathlib
import sys
import tarfile
from tag_custody_verifier_common import (check_binding, check_commits, check_members, check_source,
                                         get, read_objects, require, safe_path)

EXPECTED = {
    'refs/tags/archive/drive-migration-20261002': ('commit', 'b6ccb5e2363dfe859973bf960c494c65488b8bc2', 'b6ccb5e2363dfe859973bf960c494c65488b8bc2', 9),
    'refs/tags/integrate-697-after-am-9e8d634a': ('tag', '1203b44eeec56e0bc8b4c712368dc39178c7e09e', 'e451fa55363df0bf1c1341691c95187322fbb669', 2),
}

with tarfile.open(pathlib.Path(sys.argv[1]), 'r:gz') as tar:
    names = check_members(tar)
    manifest = json.loads(get(tar, 'MANIFEST.json'))
    require(manifest['schema'] == 'cbz.historical-main-tags-native-custody.v1', 'wrong schema')
    files, tags = manifest['files'], manifest['tags']
    require(len(tags) == 2 and len(files) == 11, 'unexpected record count')
    actual_tags = {r['ref']: (r['type'], r['object'], r['target_commit']) for r in tags}
    require(len(actual_tags) == 2 and actual_tags == {ref: value[:3] for ref, value in EXPECTED.items()},
            'tag identity mismatch')
    expected_members = ({'MANIFEST.json'} | {r['archive_name'] for r in files}
                        | {'git-objects/' + r['kind'] + '/' + r['sha'] for r in manifest['native_objects']})
    require(names == expected_members, 'unexpected or missing archive members')
    objects = read_objects(tar, manifest['native_objects'])
    check_commits(objects, manifest['commit_trees'], [row[2] for row in EXPECTED.values()])
    for row in tags:
        if row['type'] == 'tag':
            raw = objects[('tag', row['object'])]
            require(raw.splitlines()[:3] == [('object ' + row['target_commit']).encode(), b'type commit',
                                             ('tag ' + row['ref'].rsplit('/', 1)[-1]).encode()], 'tag header mismatch')
        else:
            require(row['type'] == 'commit' and row['object'] == row['target_commit'], 'lightweight tag mismatch')
    keys = [(row['tag'], row['commit'], row['path']) for row in files]
    require(len(set(keys)) == 11, 'duplicate binding')
    require(Counter(row['tag'] for row in files) == {ref: value[3] for ref, value in EXPECTED.items()},
            'unexpected tag binding counts')
    for row in files:
        require(row['tag'] in EXPECTED and EXPECTED[row['tag']][2] == row['commit'], 'binding tag mismatch')
        require(safe_path(row['path']) and row['archive_name'] == 'sources/' + row['commit'] + '/' + row['path'],
                'source archive path mismatch')
        check_source(tar, row['archive_name'], row)
        check_binding(objects, manifest['commit_trees'], row)
    print('PASS: 2 fixed tags, 2 commits,', sum(kind == 'tree' for kind, _ in objects),
          'path trees and 11 supplied file bindings; source custody only')
