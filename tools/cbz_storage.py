#!/usr/bin/env python3
"""CBZ storage tool registry: share exact immutable regular-file copies via hardlinks.

Purpose: reclaim duplicate local immutable toolchain blocks without deleting paths.
Usage: plan --root ABS (repeat) --output plan.json; apply --plan plan.json
--sha256 PRINTED_SHA. Planning is the default safe/dry operation. Apply requires
an independently retained exact plan digest and refuses stale identities. Never
use on mutable files: shared inodes intentionally share all subsequent writes.
Files with any extended attributes, or unreadable xattr state, are refused.
Self verification: python -m unittest discover -s tests -p test_cbz_storage.py
Reports establish storage integrity only, never mathematical certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid


class Refusal(RuntimeError):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    root = Path(root)
    parts = Path(relative).parts
    if Path(relative).is_absolute() or '..' in parts:
        raise Refusal('path escape')
    for item in [root, *root.parents]:
        if item.is_symlink():
            raise Refusal('symlink ancestor')
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise Refusal('symlink path')
    return path


def identity(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_uid, s.st_gid,
            s.st_mtime_ns, s.st_ctime_ns]


def require_no_xattrs(fd):
    try:
        attributes = os.listxattr(fd)
    except OSError as error:
        raise Refusal('cannot establish absence of extended attributes') from error
    if attributes:
        raise Refusal('extended attributes unsupported; file sharing refused')


def file_record(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require_no_xattrs(fd)
        if not stat.S_ISREG(before.st_mode):
            raise Refusal('nonregular file')
        h = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        require_no_xattrs(fd)
        after = os.fstat(fd)
        if identity(before) != identity(after) or identity(after) != identity(os.lstat(path)):
            raise Refusal('file changed while hashing')
        return {'identity': identity(after), 'sha256': h.hexdigest(),
                'blocks': after.st_blocks}
    finally:
        os.close(fd)


ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.zst', '.deb', '.xz')


def scan(roots, archives_only=False):
    entries = []
    for root in roots:
        safe_path(root, '.')
        if not Path(root).is_dir():
            raise Refusal('missing root')
        for current, dirs, files in os.walk(root, followlinks=False):
            if archives_only:
                dirs[:] = [d for d in dirs if d not in ('.git', '.lake') and 'cache' not in d.casefold()]
            for name in sorted(dirs + files):
                path = Path(current) / name
                s = os.lstat(path)
                if archives_only and (not stat.S_ISREG(s.st_mode) or s.st_size <= 1024 * 1024
                                      or not name.lower().endswith(ARCHIVE_SUFFIXES)
                                      or any(p in ('.git', '.lake') or 'cache' in p.casefold()
                                             for p in path.parts)):
                    continue
                base = {'root': root, 'path': str(path.relative_to(root)),
                        'mode': s.st_mode, 'uid': s.st_uid, 'gid': s.st_gid}
                if stat.S_ISREG(s.st_mode):
                    base.update(file_record(path))
                    base['kind'] = 'file'
                elif stat.S_ISLNK(s.st_mode):
                    base.update(kind='symlink', target=os.readlink(path))
                elif stat.S_ISDIR(s.st_mode):
                    base['kind'] = 'directory'
                else:
                    base['kind'] = 'other'
                entries.append(base)
    return sorted(entries, key=lambda x: (x['root'], x['path']))


def make_plan(roots, archives_only=False):
    roots = [str(Path(r).absolute()) for r in roots]
    if len(set(roots)) != len(roots):
        raise Refusal('duplicate roots')
    for a in roots:
        for b in roots:
            if a != b and Path(a) in Path(b).parents:
                raise Refusal('overlapping roots')
    entries = scan(roots, archives_only)
    groups = {}
    operations = []
    for i, entry in enumerate(entries):
        if entry['kind'] != 'file':
            continue
        ident = entry['identity']
        key = (ident[0], ident[2], ident[3], ident[4], ident[5], entry['sha256'])
        if key not in groups:
            groups[key] = i
        elif entries[groups[key]]['identity'][:2] != ident[:2]:
            operations.append([groups[key], i])
    result = {'version': 1, 'roots': roots, 'entries': entries, 'operations': operations}
    if archives_only:
        result['selection'] = 'archives-over-1MiB'
    return result


def same_content(before, after):
    keys = ('root', 'path', 'mode', 'uid', 'gid', 'kind', 'sha256', 'target')
    return [{k: e[k] for k in keys if k in e} for e in before] == [
        {k: e[k] for k in keys if k in e} for e in after]


def check_entry(entry, allow_link_ctime=False):
    path = safe_path(entry['root'], entry['path'])
    actual = file_record(path)
    expected = entry['identity']
    observed = actual['identity']
    # Creating/removing links changes ctime, so compare all other identity fields.
    if ((observed[:-1] != expected[:-1] if allow_link_ctime else observed != expected)
            or actual['sha256'] != entry['sha256']):
        raise Refusal('stale file identity or content: ' + str(path))
    return path


def apply_plan(plan, before_replace=None):
    if plan.get('version') != 1:
        raise Refusal('unsupported plan')
    # Re-derive operations and inventory before any mutation: no injected paths,
    # forged pairs, changed files, added files, missing files, or mode drift.
    archives_only = plan.get('selection') == 'archives-over-1MiB'
    fresh = make_plan(plan['roots'], archives_only)
    if fresh != plan:
        raise Refusal('plan inputs changed')
    free_before = os.statvfs(plan['roots'][0]).f_bavail * os.statvfs(plan['roots'][0]).f_frsize
    applied = 0
    for source_i, target_i in plan['operations']:
        source = plan['entries'][source_i]
        target = plan['entries'][target_i]
        src = check_entry(source, allow_link_ctime=True)
        dst = check_entry(target, allow_link_ctime=True)
        temporary = dst.parent / ('.cbz-hardlink-' + uuid.uuid4().hex)
        try:
            os.link(src, temporary, follow_symlinks=False)
            if before_replace is not None:
                before_replace(src, dst)
            # Hash both fresh paths after creating link and before atomic replace.
            check_entry(source, allow_link_ctime=True)
            check_entry(target, allow_link_ctime=True)
            if os.lstat(temporary).st_ino != os.lstat(src).st_ino:
                raise Refusal('temporary link identity changed')
            os.replace(temporary, dst)
            applied += 1
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
    after = scan(plan['roots'], archives_only)
    if not same_content(plan['entries'], after):
        raise Refusal('post-apply inventory/content/modes mismatch')
    for a, b in plan['operations']:
        ea, eb = plan['entries'][a], plan['entries'][b]
        if not os.path.samefile(safe_path(ea['root'], ea['path']), safe_path(eb['root'], eb['path'])):
            raise Refusal('post-apply inode mismatch')
    v = os.statvfs(plan['roots'][0])
    return {'applied': applied, 'content_paths_modes_verified': True,
            'free_before': free_before, 'free_after': v.f_bavail * v.f_frsize,
            'actual_df_bytes_freed': v.f_bavail * v.f_frsize - free_before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    planning = sub.add_parser('plan')
    planning.add_argument('--root', action='append', required=True)
    planning.add_argument('--output', required=True)
    planning.add_argument('--archives-only', action='store_true',
                          help='Only >1MiB immutable archive candidates; excludes .git/.lake and cache paths')
    applying = sub.add_parser('apply')
    applying.add_argument('--plan', required=True)
    applying.add_argument('--sha256', required=True)
    applying.add_argument('--report', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        plan = make_plan(args.root, args.archives_only)
        data = encoded(plan)
        # Exclusive creation: never overwrite a previous attempted plan.
        with open(args.output, 'xb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        print(json.dumps({'plan_sha256': digest(data), 'operations': len(plan['operations']),
                          'estimated_duplicate_bytes': sum(plan['entries'][b]['identity'][2] for a,b in plan['operations'])}))
    else:
        data = Path(args.plan).read_bytes()
        if digest(data) != args.sha256:
            raise Refusal('plan SHA mismatch')
        plan = json.loads(data)
        if encoded(plan) != data:
            raise Refusal('noncanonical plan')
        report = apply_plan(plan)
        report['plan_sha256'] = args.sha256
        with open(args.report, 'x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
