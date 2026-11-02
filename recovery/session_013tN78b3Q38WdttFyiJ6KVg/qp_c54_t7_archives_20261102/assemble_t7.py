#!/usr/bin/env python3
"""Check tranche-7 items (members of committed archives) against the archive bytes and assemble catalog-schema rows.

Usage: python3 -I -B assemble_t7.py CLONES_JSON FACTS_DIR VERDICTS_JSON OUTPUT_JSON

FACTS_DIR holds abatch<N>.json (the objects given to the workflow). VERDICTS_JSON is the workflow result {"batches": [{"batch",
"kind", "verdicts": [{"object", "ok", "problems", "items"}]}]}. Archives are read from Git into memory; nothing is extracted to
disk and nothing is executed.

Fails (exit 1, nothing written) on: a missing, duplicate or unknown object; an archive whose blob, bytes or SHA-256 differ from the
facts; a member whose bytes, SHA-256 or Git blob (recomputed from the archive) differ from the facts; items that do not partition
an object's members (a member covered by no item or by two; a file spec matching nothing); duplicate item ids within an object; an
OBLIGATION without statements, with a statement not verbatim at its lines of a member in the item, or with an unclassified
floating-point field; a REVIEW_OF or COPY_OR_EXPOSITION_OF item without refers_to.

Output: records (OBLIGATION items, ids QP-C54-T7-NNN, in the catalog's proposal schema), other_items (every other item) and
mechanical_copies (one row per object listing its members resolved as copies). Every row binds the archive file (repository,
commit, path, blob, bytes, SHA-256) and lists each member it covers with its own bytes, SHA-256 and Git blob.
"""
import glob, gzip, hashlib, io, json, os, re, subprocess, sys, tarfile, zipfile


def blob_id(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def read_member(name, data, member):
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.read(member)
    if re.search(r'\.(tgz|tar\.gz|tar)$', name, re.I):
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            return t.extractfile(t.getmember(member)).read()
    if name.lower().endswith('.gz'):
        return gzip.decompress(data)
    raise ValueError(f'not an archive: {name}')


def main(clones_path, facts_dir, verdicts_path, output):
    clones = json.load(open(clones_path))
    facts = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'abatch*.json'))):
        for o in json.load(open(f))['objects']:
            facts[o['object']] = o
    verdicts = [v for b in json.load(open(verdicts_path))['batches'] for v in (b['verdicts'] or [])]
    problems, seen = [], {}
    for v in verdicts:
        if v['object'] in seen:
            problems.append(f'duplicate object {v["object"]}')
        seen[v['object']] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    archive_cache, member_cache = {}, {}

    def archive_bytes(fact):
        key = (fact['repository'], fact['commit'], fact['archive'])
        if key not in archive_cache:
            clone = clones[fact['repository'].split('/', 1)[1]]
            archive_cache[key] = subprocess.run(['git', '-C', clone, 'cat-file', 'blob', f'{fact["commit"]}:{fact["archive"]}'],
                                                capture_output=True, check=True).stdout
        return archive_cache[key]

    def member_bytes(fact, spec):
        key = (fact['archive'], tuple(spec))
        if key not in member_cache:
            b = read_member(fact['archive'], archive_bytes(fact), spec[0])
            if len(spec) > 1:
                b = read_member(spec[0], b, spec[1])
            member_cache[key] = b
        return member_cache[key]

    records, others, copies, n = [], [], [], 0
    for obj in sorted(facts):
        fact = facts[obj]
        data = archive_bytes(fact)
        if (blob_id(data), hashlib.sha256(data).hexdigest()) != (fact['archive_git_blob'], fact['archive_sha256']):
            problems.append(f'{obj}: archive bytes differ from the facts')
            continue
        archive_binding = {'path': fact['archive'], 'git_blob': fact['archive_git_blob'], 'bytes': len(data), 'sha256': fact['archive_sha256']}
        members = {m['member']: m for m in fact['members']}
        for m in fact['members'] + [c for c in fact['copies'] if not c['copy_of'].startswith('nested archive')]:
            b = member_bytes(fact, m['member_spec'])
            if (len(b), hashlib.sha256(b).hexdigest(), blob_id(b)) != (m['bytes'], m['sha256'], m['git_blob']):
                problems.append(f'{obj}: member {m["member"]} differs from the facts')
        copies.append({'object': obj, 'classification': 'MECHANICAL_COPY', 'reason': (
            'Each listed member is byte-identical (same Git blob) to the named main or Math- tree file or to a member classified '
            'in another tranche-7 object, or is a nested archive whose members are listed individually; its disposition is that '
            'of its original.'),
            'source_binding': {'repository': fact['repository'], 'commit': fact['commit'], 'files': [archive_binding],
                               'members': [{k: c[k] for k in ('member', 'bytes', 'sha256', 'git_blob', 'copy_of')} |
                                           ({'copy_disposed_location': c['copy_disposed_location']} if c.get('copy_disposed_location') else {})
                                           for c in fact['copies']]}})
        if obj not in seen:
            continue
        v = seen[obj]
        cover = {p: [] for p in members}
        ids = [it['item_id'] for it in v['items']]
        if len(ids) != len(set(ids)):
            problems.append(f'{obj}: duplicate item ids')
        for it in v['items']:
            if it['object'] != obj:
                problems.append(f'{obj}: item {it["item_id"]} names object {it["object"]}')
            mine = set()
            for spec in it['files']:
                hits = [p for p in members if p == spec or (spec.endswith('/') and p.startswith(spec))]
                if not hits:
                    problems.append(f'{obj}:{it["item_id"]}: file spec {spec} matches no member')
                mine.update(hits)
            for p in mine:
                cover[p].append(it['item_id'])
            it['_files'] = sorted(mine)
        for p, c in cover.items():
            if len(c) != 1:
                problems.append(f'{obj}: {p} covered by {len(c)} items {c}')
        for it in v['items']:
            cls = it['classification']
            for st in it['statements']:
                if st['file'] not in it['_files']:
                    problems.append(f'{obj}:{it["item_id"]}: statement member {st["file"]} not in the item')
                    continue
                lines = member_bytes(fact, members[st['file']]['member_spec']).decode('utf-8').split('\n')
                a, e = st['line_start'], st['line_end']
                if not (1 <= a <= e <= len(lines)) or '\n'.join(lines[a - 1:e]) != st['text'] or not st['text'].strip():
                    problems.append(f'{obj}:{it["item_id"]}: statement {st["file"]}:{a}-{e} is not verbatim')
            if cls == 'OBLIGATION':
                if not it['statements']:
                    problems.append(f'{obj}:{it["item_id"]}: OBLIGATION without statements')
                if it['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
                    problems.append(f'{obj}:{it["item_id"]}: floating_point unclassified')
            if cls in ('REVIEW_OF', 'COPY_OR_EXPOSITION_OF') and not it['refers_to']:
                problems.append(f'{obj}:{it["item_id"]}: {cls} without refers_to')
            binding = {'repository': fact['repository'], 'commit': fact['commit'], 'files': [archive_binding],
                       'members': [{k: members[p][k] for k in ('member', 'bytes', 'sha256', 'git_blob')} for p in it['_files']]}
            verification = {'ok': v['ok'], 'problems_fixed': v['problems']}
            if cls != 'OBLIGATION':
                others.append({'object': obj, 'item_id': it['item_id'], 'classification': cls, 'title': it['title'],
                               'reason': it['classification_reason'], 'refers_to': it['refers_to'], 'source_binding': binding,
                               'verification': verification})
                continue
            n += 1
            records.append({
                'id': f'QP-C54-T7-{n:03d}', 'title': it['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
                'proposal_kind': 'NEW_RECORD',
                'location': [f'{fact["repository"]}@{fact["commit"]}:{fact["archive"]}#{p}' for p in it['_files']],
                'layer_a_status': it['layer_a_status'], 'exact_boundary': it['exact_boundary'], 'blocking_reason': it['blocking_reason'],
                'attempted': it['attempted'], 'resumption_cost': it['resumption_cost'],
                'isolation': ('The item is a member of a committed archive, so nothing in the CLOSED domain can import it. The '
                              'composer must adopt this record before it becomes a terminal quarantine.'),
                'owner': 'none', 'maintenance': 'zero', 'object': obj, 'item_id': it['item_id'],
                'classification_reason': it['classification_reason'], 'source_statements': it['statements'],
                'source_hypotheses': it['hypotheses'], 'source_dependencies': it['dependencies'],
                'source_self_limits': it['source_self_limits'], 'finite_checks': it['finite_checks'],
                'floating_point': it['floating_point'], 'refers_to': it['refers_to'], 'source_binding': binding,
                'verification': verification})
    if problems:
        print('\n'.join(problems[:60]))
        print(f'{len(problems)} problem(s); nothing written')
        return 1
    from collections import Counter
    with open(output, 'x') as stream:
        json.dump({'schema': 'claude54.quarantine-proposals.t7.v1', 'records': records, 'other_items': others,
                   'mechanical_copies': copies}, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    print(f'{len(records)} obligation records; other items {dict(Counter(o["classification"] for o in others))}; '
          f'{sum(len(c["source_binding"]["members"]) for c in copies)} members resolved as copies in {len(copies)} objects')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
