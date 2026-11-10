"""Common, deliberately bounded verification of exact Git objects in tar archives.

This checks bytes and tree paths against fixed observed tag objects. It does not
establish that a changed-path list is complete or classify mathematical claims.
"""
import hashlib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_oid(kind, data):
    header = kind.encode() + b' ' + str(len(data)).encode() + b'\0'
    return hashlib.sha1(header + data).hexdigest()


def safe_path(name):
    return (isinstance(name, str) and bool(name) and not name.startswith('/')
            and all(part not in ('', '.', '..') for part in name.split('/')))


def tree_entries(raw):
    entries = {}
    pos = 0
    while pos < len(raw):
        space = raw.index(b' ', pos)
        nul = raw.index(b'\0', space)
        mode = raw[pos:space].decode('ascii')
        name = raw[space + 1:nul].decode('utf-8')
        object_bytes = raw[nul + 1:nul + 21]
        require(len(object_bytes) == 20 and safe_path(name) and '/' not in name and name not in entries,
                'invalid or duplicate tree entry')
        entries[name] = (mode, object_bytes.hex())
        pos = nul + 21
    require(pos == len(raw), 'truncated tree')
    return entries


def check_members(tar):
    members = tar.getmembers()
    names = [member.name for member in members]
    require(len(names) == len(set(names)), 'duplicate archive member')
    require(all(member.isfile() and 0 <= member.size <= 16 * 1024 * 1024 and safe_path(member.name)
                for member in members), 'invalid archive member')
    return set(names)


def get(tar, name):
    stream = tar.extractfile(name)
    require(stream is not None, 'missing archive member: ' + name)
    return stream.read()


def read_objects(tar, records):
    objects = {}
    for row in records:
        kind, sha = row['kind'], row['sha']
        require(kind in ('tag', 'commit', 'tree') and (kind, sha) not in objects, 'invalid or duplicate object')
        raw = get(tar, 'git-objects/' + kind + '/' + sha)
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'] and git_oid(kind, raw) == sha,
                'Git object identity mismatch')
        objects[(kind, sha)] = raw
    return objects


def check_commits(objects, commit_trees, expected_commits):
    require(set(commit_trees) == set(expected_commits), 'unexpected commit set')
    for commit, tree in commit_trees.items():
        require(objects[('commit', commit)].splitlines()[0] == ('tree ' + tree).encode(),
                'commit tree mismatch')


def check_binding(objects, commit_trees, row):
    path = row['path']
    require(safe_path(path) and row['mode'] in ('100644', '100755'), 'invalid source path or mode')
    parts = path.split('/')
    tree = commit_trees[row['commit']]
    for name in parts[:-1]:
        mode, tree = tree_entries(objects[('tree', tree)])[name]
        require(mode in ('40000', '040000'), 'expected path tree')
    mode, sha = tree_entries(objects[('tree', tree)])[parts[-1]]
    require(mode == row['mode'] and sha == row['git_blob'], 'tree source binding mismatch')


def check_source(tar, name, row):
    raw = get(tar, name)
    require(len(raw) == row['bytes'] and digest(raw) == row['sha256'] and git_oid('blob', raw) == row['git_blob'],
            'source blob mismatch')
    mode = 0o755 if row['mode'] == '100755' else 0o644
    require(row['mode'] in ('100644', '100755') and tar.getmember(name).mode == mode, 'source mode mismatch')
