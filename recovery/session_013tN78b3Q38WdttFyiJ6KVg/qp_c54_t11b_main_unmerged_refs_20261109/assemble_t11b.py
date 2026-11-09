#!/usr/bin/env python3
"""Check tranche-11b items (mathematics on main pull-request heads and branch tips whose bytes occur in no commit of main's
history; assemble_t10.py with the repository set to d6g8k5htny-coder/main) against main bytes at each object's ref head, and assemble catalog-schema rows.

Usage: python3 -I -B assemble_t11b.py MAIN_REPO FACTS_DIR CANDIDATE_CATALOG OUTPUT_JSON VERDICTS_JSON [VERDICTS_JSON...]

FACTS_DIR holds qbatch<N>.json from t10_batches.py. Each object has an id "<directory>@<ref>", its ref (refs/pull/N/head or
refs/heads/<branch>), the ref's head commit, pr and pr_open, and its files with full Git blobs. The heads must be present in
MATH_REPO. CANDIDATE_CATALOG is the catalog the workflows checked claim_held_by against. Each VERDICTS_JSON is a workflow result
{"batches": [{"batch", "kind", "verdicts": [{"object", "ok", "problems", "items"}]}]}. Bindings come only from Git at each
object's commit.

Fails (exit 1, nothing written) on: a missing, duplicate or unknown object; a head absent from MATH_REPO; a file whose blob at the
object's commit differs from the facts; items that do not partition an object's files (a file covered by no item or by two; a file
spec matching nothing); duplicate item ids within an object; an item naming another object; an OBLIGATION without statements, with
a statement whose file is not in the object or whose text differs from that file's lines at the commit, or with an unclassified
floating-point field; a REVIEW_OF or COPY_OR_EXPOSITION_OF item without refers_to; a claim_held_by id that is no adopted or
proposed record, or claim_held_by on a non-OBLIGATION item. A statement may quote a file that another item of the same object
holds; that file is then bound under source_binding.context_files and reported.

Kinds: an OBLIGATION whose verified claim_held_by names only adopted records becomes a PINNING_AMENDMENT to them. One held by a
QP-C54 proposal stays a NEW_RECORD and carries claim_overlaps (SAME_CLAIM_AS_PROPOSAL). Any other OBLIGATION is a NEW_RECORD.
Locations are file-level, d6g8k5htny-coder/main@<head>:<path>. No archive tag exists for these heads, so every record has
archive_ref null and says so in its isolation text.
"""
import glob, hashlib, json, os, subprocess, sys
from collections import Counter

REPO = 'd6g8k5htny-coder/main'


def git(repo, *a):
    return subprocess.run(['git', '-C', repo, *a], capture_output=True)


def refbind(fact):
    b = {'repository': REPO, 'commit': fact['commit'], 'ref': fact['ref']}
    if fact['pr'] is not None:
        b.update(pull_request=f'main#{fact["pr"]}', pull_state='open' if fact['pr_open'] else 'closed unmerged')
    else:
        b['branch'] = fact['ref'][len('refs/heads/'):]
    return b


def where(fact):
    if fact['pr'] is not None:
        return (f'the head {fact["commit"]} of {"open" if fact["pr_open"] else "closed, unmerged"} pull request main#{fact["pr"]} '
                f'({fact["ref"]})')
    return f'the tip {fact["commit"]} of branch {fact["ref"][len("refs/heads/"):]} (no pull-request head carries these bytes)'


def main(math_repo, facts_dir, cand_path, output, *verdict_paths):
    facts = {}
    for f in sorted(glob.glob(os.path.join(facts_dir, 'qbatch*.json'))):
        for o in json.load(open(f))['objects']:
            if o['object'] in facts:
                raise SystemExit(f'duplicate object in facts: {o["object"]}')
            facts[o['object']] = o
    cand = json.load(open(cand_path))
    adopted = {r['id'] for r in cand['quarantines']}
    proposed = {r['id'] for k in ('proposed_quarantines', 'proposed_quarantines_federation', 'proposed_pinning_amendments')
                for r in cand.get(k, [])}
    verdicts = [v for p in verdict_paths for b in json.load(open(p))['batches'] for v in (b['verdicts'] or [])]
    problems, seen = [], {}
    for v in verdicts:
        if v['object'] in seen:
            problems.append(f'duplicate object {v["object"]}')
        seen[v['object']] = v
    problems += [f'missing: {o}' for o in sorted(set(facts) - set(seen))]
    problems += [f'unknown: {o}' for o in sorted(set(seen) - set(facts))]
    for c in sorted({o['commit'] for o in facts.values()}):
        if git(math_repo, 'cat-file', '-e', f'{c}^{{commit}}').returncode:
            problems.append(f'PR head {c} is not in {math_repo}')
    records, others, mech, cross_item, n = [], [], [], [], 0
    cache = {}

    def blob_at(commit, p):
        if (commit, p) not in cache:
            r = git(math_repo, 'cat-file', 'blob', f'{commit}:{p}')
            b = git(math_repo, 'rev-parse', f'{commit}:{p}').stdout.decode().strip()
            cache[(commit, p)] = (None, None) if r.returncode else (r.stdout, b)
        return cache[(commit, p)]

    def bind(commit, p):
        data, blob = blob_at(commit, p)
        return {'path': p, 'git_blob': blob, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    for obj in sorted(facts):
        if obj not in seen:
            continue
        v, fact = seen[obj], facts[obj]
        commit, pr, ref = fact['commit'], fact['pr'], fact['ref']
        files = {f['path']: f for f in fact['files']}
        copies = {f['path']: f for f in fact.get('copies', [])}
        for p, f in list(files.items()) + list(copies.items()):
            data, blob = blob_at(commit, p)
            if data is None or not blob.startswith(f['git_blob']):
                problems.append(f'{obj}: {p} at {commit[:8]} does not match the facts blob {f["git_blob"]}')
        cover = {p: [] for p in files}
        ids = [it['item_id'] for it in v['items']]
        if len(ids) != len(set(ids)):
            problems.append(f'{obj}: duplicate item ids')
        for it in v['items']:
            if it['object'] != obj:
                problems.append(f'{obj}: item {it["item_id"]} names object {it["object"]}')
            mine = set()
            for spec in it['files']:
                hits = [p for p in files if p == spec or (spec.endswith('/') and p.startswith(spec))]
                if not hits:
                    problems.append(f'{obj}:{it["item_id"]}: file spec {spec} matches no object file')
                mine.update(hits)
            for p in mine:
                cover[p].append(it['item_id'])
            it['_files'] = sorted(mine)
        for p, c in cover.items():
            if len(c) != 1:
                problems.append(f'{obj}: {p} covered by {len(c)} items {c}')
        for it in v['items']:
            cls = it['classification']
            it['_context'] = []
            for st in it['statements']:
                if st['file'] not in it['_files']:
                    if st['file'] in files or st['file'] in copies:
                        if st['file'] not in it['_context']:
                            it['_context'].append(st['file'])
                            cross_item.append(f'{obj}:{it["item_id"]} quotes {st["file"]} (another item\'s file or a resolved main copy)')
                    else:
                        problems.append(f'{obj}:{it["item_id"]}: statement file {st["file"]} not in the object')
                        continue
                data, _ = blob_at(commit, st['file'])
                lines = data.decode('utf-8').split('\n') if data is not None else []
                a, b = st['line_start'], st['line_end']
                if not (1 <= a <= b <= len(lines)) or '\n'.join(lines[a - 1:b]) != st['text'] or not st['text'].strip():
                    problems.append(f'{obj}:{it["item_id"]}: statement {st["file"]}:{a}-{b} is not verbatim at {commit[:8]}')
            held = list(it.get('claim_held_by') or [])
            for h in held:
                if h not in adopted and h not in proposed:
                    problems.append(f'{obj}:{it["item_id"]}: claim_held_by {h} is no adopted or proposed record')
            if cls == 'OBLIGATION':
                if not it['statements']:
                    problems.append(f'{obj}:{it["item_id"]}: OBLIGATION without statements')
                if it['floating_point'].split()[0].rstrip(':;,.') not in ('USED', 'NONE', 'NO_SCRIPTS'):
                    problems.append(f'{obj}:{it["item_id"]}: floating_point unclassified')
            elif held:
                problems.append(f'{obj}:{it["item_id"]}: claim_held_by on a {cls} item')
            if cls in ('REVIEW_OF', 'COPY_OR_EXPOSITION_OF') and not it['refers_to']:
                problems.append(f'{obj}:{it["item_id"]}: {cls} without refers_to')
            if any(blob_at(commit, p)[0] is None for p in it['_files'] + it['_context']):
                continue
            binding = dict(refbind(fact), files=[bind(commit, p) for p in it['_files']])
            if it['_context']:
                binding['context_files'] = [bind(commit, p) for p in it['_context']]
            verification = {'ok': v['ok'], 'problems_fixed': v['problems']}
            if cls != 'OBLIGATION':
                others.append({'object': obj, 'item_id': it['item_id'], 'classification': cls, 'title': it['title'],
                               'reason': it['classification_reason'], 'refers_to': it['refers_to'], 'source_binding': binding,
                               'verification': verification})
                continue
            n += 1
            held_adopted = [h for h in held if h in adopted]
            held_prop = [h for h in held if h not in adopted]
            kind = 'PINNING_AMENDMENT' if held_adopted and not held_prop else 'NEW_RECORD'
            lean = any(p.endswith('.lean') for p in it['_files'])
            rec = {
                'id': f'QP-C54-T11B-{n:03d}', 'title': it['title'], 'status': 'QUARANTINED', 'proposal_status': 'PROPOSED',
                'proposal_kind': kind, 'target_records': held_adopted if kind == 'PINNING_AMENDMENT' else [],
                'location': [f'{REPO}@{commit}:{p}' for p in it['_files']],
                'layer_a_status': it['layer_a_status'], 'exact_boundary': it['exact_boundary'], 'blocking_reason': it['blocking_reason'],
                'attempted': it['attempted'], 'resumption_cost': it['resumption_cost'],
                'isolation': (f'These bytes occur in no commit of main\'s history; they are at {where(fact)}, so nothing in the '
                              'CLOSED domain imports them. '
                              + ('The item contains Lean source that hosts no CLOSED declaration; it must not be imported into a '
                                 'CLOSED root. ' if lean else 'The item contains no Lean source. ')
                              + 'No durable tag retains this commit: before the ref is archived, bundled or deleted, the composer '
                              'needs an annotated archive tag at it, as for Q-8336, Q-8340 and Q-8429; if the work is integrated, the '
                              'location becomes the same paths on main. '
                              'The composer must adopt this record before it becomes a terminal quarantine.'),
                'archive_ref': None, 'owner': 'none', 'maintenance': 'zero', 'object': obj, 'item_id': it['item_id'],
                'classification_reason': it['classification_reason'], 'claim_held_by': held, 'source_statements': it['statements'],
                'source_hypotheses': it['hypotheses'], 'source_dependencies': it['dependencies'],
                'source_self_limits': it['source_self_limits'], 'finite_checks': it['finite_checks'],
                'floating_point': it['floating_point'], 'refers_to': it['refers_to'], 'source_binding': binding,
                'verification': verification}
            if held_prop:
                rec['claim_overlaps'] = [{'record': h, 'relation': 'SAME_CLAIM_AS_PROPOSAL' if h in proposed else 'HOLDS_CLAIM',
                                          'note': 'The verifiers found this record already holds the claim; the composer merges the two.'}
                                         for h in held]
            records.append(rec)
        if copies:
            mech.append({'object': obj, 'reason': 'byte copies of files on the main tip; each copy\'s disposition is that of '
                         'the main file named in copy_of', 'source_binding': {
                             **refbind(fact), 'files': [dict(bind(commit, p), copy_of=copies[p]['copy_of']) for p in sorted(copies)]}})
    if problems:
        print('\n'.join(problems[:80]))
        print(f'{len(problems)} problem(s); nothing written')
        return 1
    with open(output, 'x') as stream:
        json.dump({'schema': 'claude54.quarantine-proposals.t11b.v1', 'main_ref': 'd2b0216e15d545c47f25b6a9265dd3298971e9fe',
                   'ref_heads': {o['ref']: o['commit'] for o in sorted(facts.values(), key=lambda o: o['ref'])},
                   'records': records, 'other_items': others, 'mechanical_copies': mech}, stream, indent=1, ensure_ascii=False)
        stream.write('\n')
    for c in cross_item:
        print('cross-item quote:', c)
    print(f'{len(records)} obligation records ({dict(Counter(r["proposal_kind"] for r in records))}); other items '
          f'{dict(Counter(o["classification"] for o in others))}')
    print('floating point', dict(Counter(r['floating_point'].split()[0].rstrip(':;,.') for r in records)))
    print('mechanical copies', sum(len(m['source_binding']['files']) for m in mech), 'in', len(mech), 'objects')
    print('records by ref', dict(Counter(r['source_binding']['ref'] for r in records)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 6:
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
