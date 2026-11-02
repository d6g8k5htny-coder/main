#!/usr/bin/env python3
"""Cross-reference Claude S's comment-math census (#745) with CBZ dispositions.

Usage: python3 -I -B census_dispositions.py CENSUS_JSON CANDIDATE_CATALOG CLOSED_JSON MATH_REPO MAIN_REPO T3_DIRS_JSON OUTPUT_JSON

T3_DIRS_JSON lists the Math- reviews/ directories that no record names (classified separately in tranche 3).

CANDIDATE_CATALOG is the QP-C54 candidate (staging 32ab7d90 catalog plus proposed_* keys). For each artifact
every committed carrier is classified, best first:

  CLOSED_SOURCE         the carrier is a .lean file that hosts a CLOSED declaration (same repository and path)
  ADOPTED_QUARANTINE    the carrier's path, an ancestor directory, or a location inside its packet directory is an
                        adopted quarantine location in the same repository (anchors such as #claims/x are recorded
                        as ANCHORED: the record names one claim inside that file, not the whole file)
  PROPOSED_QP_C54       the same test against proposed_quarantines and proposed_pinning_amendments
  NOT_OBLIGATION_QP_C54 the carrier lies in a proposed_non_obligations object
  COPY_OF_DISPOSED      the carrier blob equals the blob of a file at a disposed location (any repository, at the
                        location's own commit when it is in the local clone)
  NONE

Path matching only applies to carriers on main or Math- at a commit whose path the record names (the catalog's
commits are older cuts; a path match there is reported as path coverage, as in the federation census). Carriers on
branch-only commits are matched by blob only. Committed artifacts left with NONE, or whose best match is only a
SIBLING (a record names a different file in the same directory), get a routing lane: T3_REVIEW_DIR, LEAN,
MAIN_SIDE, MATH_REVIEW_PARTLY_NAMED or OTHER. Separately, every artifact whose comment ids or key occur in the
candidate's text is listed as MENTIONED: a lead, not a disposition.
"""
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict

RANK = ['CLOSED_SOURCE', 'ADOPTED_QUARANTINE', 'PROPOSED_QP_C54', 'NOT_OBLIGATION_QP_C54', 'COPY_OF_DISPOSED', 'NONE']
REPOS = {'d6g8k5htny-coder/Math-': 'Math-', 'd6g8k5htny-coder/main': 'main', 'Math-': 'Math-', 'main': 'main'}


def git(repo, *args):
    r = subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def loc_entries(record):
    """(repo short name, commit, path, anchor) for each string or dict location of a record."""
    out = []
    locs = record.get('location')
    for loc in locs if isinstance(locs, list) else [locs]:
        if isinstance(loc, dict):
            repo = REPOS.get(loc.get('repository', ''))
            for p in ([loc['path']] if 'path' in loc else []) + (loc.get('paths') or []):
                out.append((repo, loc.get('commit'), p.rstrip('/'), None))
            continue
        if not isinstance(loc, str) or ':' not in loc:
            continue
        repo_commit, _, path = loc.partition(':')
        repo, _, commit = repo_commit.partition('@')
        path, _, anchor = path.partition('#')
        if REPOS.get(repo) and path:
            out.append((REPOS[repo], commit or None, path.rstrip('/'), anchor or None))
    return out


def main(census_path, cand_path, closed_path, math_repo, main_repo, t3_path, output):
    t3 = set(json.load(open(t3_path)))
    census = json.load(open(census_path))
    cand = json.load(open(cand_path))
    closed = json.load(open(closed_path))['entries']
    clones = {'Math-': math_repo, 'main': main_repo}

    closed_paths = {(REPOS.get(e['repository'], e['repository']), e['path']): e['id'] for e in closed}
    index = {'ADOPTED_QUARANTINE': defaultdict(list), 'PROPOSED_QP_C54': defaultdict(list), 'NOT_OBLIGATION_QP_C54': defaultdict(list)}
    disposed_locs = []
    for cls, rows in (('ADOPTED_QUARANTINE', cand['quarantines']),
                      ('PROPOSED_QP_C54', cand['proposed_quarantines'] + cand['proposed_pinning_amendments'])):
        for r in rows:
            for repo, commit, path, anchor in loc_entries(r):
                index[cls][(repo, path)].append((r['id'], anchor))
                disposed_locs.append((repo, commit, path))
    for n in cand['proposed_non_obligations']:
        b = n['source_binding']
        obj = n['object']
        index['NOT_OBLIGATION_QP_C54'][('Math-', obj.rstrip('/'))].append((f'NOT_OBLIGATION:{obj}', None))

    # Blobs at disposed locations, read at the location's commit (all files under a directory location).
    disposed_blobs = {}
    for repo, commit, path in sorted(set(disposed_locs), key=lambda t: tuple(x or "" for x in t)):
        clone = clones.get(repo)
        if not clone or not commit or not re.fullmatch(r'[0-9a-f]{40}', commit):
            continue
        listing = git(clone, 'ls-tree', '-r', commit, '--', path)
        for line in (listing or '').splitlines():
            meta, p = line.split('\t', 1)
            disposed_blobs.setdefault(meta.split()[2], f'{repo}@{commit[:8]}:{p}')

    def packet_dirs(repo, commit, path):
        parts = path.split('/')
        return ['/'.join(parts[:i]) for i in range(len(parts), 0, -1)]

    def path_cover(cls, repo, path):
        idx = index[cls]
        for anc in packet_dirs(repo, None, path):
            if (repo, anc) in idx:
                hits = idx[(repo, anc)]
                anchored = all(a for _, a in hits) and anc == path
                return anc, [h for h, _ in hits], anchored, 'EXACT' if anc == path else 'ANCESTOR'
        # Packet level: a disposed location inside the carrier's own directory.
        parent = path.rsplit('/', 1)[0] if '/' in path else ''
        if parent and parent.count('/') >= 1:
            for (r, p), hits in idx.items():
                if r == repo and p.startswith(parent + '/'):
                    return p, [h for h, _ in hits], False, 'SIBLING'
        return None

    cuts = {'Math-': census['source_cuts']['Math-'], 'main': census['source_cuts']['main']}
    cuts = {k: (v['commit'] if isinstance(v, dict) else v) for k, v in cuts.items()}

    text = json.dumps({k: cand[k] for k in ('quarantines', 'proposed_quarantines', 'proposed_pinning_amendments',
                                            'proposed_non_obligations', 'main_census_independent_validation',
                                            'non_obligation_support')}, ensure_ascii=False)
    rows = []
    for art in census['artifacts']:
        results = []
        for car in art['custody']['carriers']:
            repo, path, blob, commit = car['repo'], car['path'], car.get('blob'), car.get('commit', '')
            res = None
            on_cut = repo in cuts and commit == cuts[repo]
            if on_cut and path.endswith('.lean') and (repo, path) in closed_paths:
                res = ('CLOSED_SOURCE', closed_paths[(repo, path)], False)
            if not res and on_cut:
                for cls in ('ADOPTED_QUARANTINE', 'PROPOSED_QP_C54', 'NOT_OBLIGATION_QP_C54'):
                    hit = path_cover(cls, repo, path)
                    if hit:
                        where, ids, anchored, how = hit
                        res = (cls, f'{",".join(sorted(set(ids))[:4])} ({where})', anchored, how)
                        break
            if res and res[0] == 'CLOSED_SOURCE':
                res = res + ('EXACT',)
            if not res and blob in disposed_blobs:
                res = ('COPY_OF_DISPOSED', disposed_blobs[blob], False, 'BLOB')
            results.append({'repo': repo, 'commit': commit, 'path': path, 'role': car.get('role'),
                            'disposition': res[0] if res else 'NONE', 'by': res[1] if res else None,
                            'anchored_only': bool(res and res[2]), 'match': res[3] if res else None})
        best = min((RANK.index(r['disposition']) for r in results), default=len(RANK) - 1)
        ids = set()
        for v in art['versions']:
            m = re.search(r'/(\d{6,})$', v['comment'])
            if m:
                ids.add(m.group(1))
        mentioned = sorted(i for i in ids if i in text) + ([art['key']] if art['key'] in text else [])
        rows.append({'key': art['key'], 'custody': art['custody']['class'], 'kind': art['kind'], 'family': art['family'],
                     'implementation': art['implementation']['class'] if isinstance(art['implementation'], dict) else art['implementation'],
                     'disposition': RANK[best], 'carriers': results, 'mentioned_in_catalog_text': mentioned})
    summary = defaultdict(Counter)
    for r in rows:
        summary[r['custody']][r['disposition']] += 1
    total = Counter(r['disposition'] for r in rows)
    def lane(r):
        paths = [(c['repo'], c['path']) for c in r['carriers']]
        for repo, p in paths:
            if repo == 'Math-' and p.startswith('reviews/'):
                return 'T3_REVIEW_DIR' if '/'.join(p.split('/')[:2]) in t3 else 'MATH_REVIEW_PARTLY_NAMED'
        if any(p.endswith(('.lean', '.lean.txt')) for _, p in paths):
            return 'LEAN'
        return 'MAIN_SIDE' if paths and all(repo == 'main' for repo, _ in paths) else 'OTHER'
    for r in rows:
        best_matches = {c['match'] for c in r['carriers'] if c['disposition'] == r['disposition']}
        r['match'] = min(best_matches, key=['EXACT', 'ANCESTOR', 'BLOB', 'SIBLING'].index) if r['disposition'] != 'NONE' else None
    lanes = Counter()
    for r in rows:
        if r['custody'] == 'COMMITTED_ON_MAIN' and (r['disposition'] == 'NONE' or r['match'] == 'SIBLING'):
            r['lane'] = lane(r)
            lanes[f"{'SIBLING_ONLY' if r['match'] == 'SIBLING' else 'NONE'}:{r['lane']}"] += 1
    by_match = defaultdict(Counter)
    for r in rows:
        if r['match']:
            by_match[r['disposition']][r['match']] += 1
    anchored = sum(1 for r in rows if r['disposition'] in ('ADOPTED_QUARANTINE', 'PROPOSED_QP_C54') and all(
        c['anchored_only'] for c in r['carriers'] if c['disposition'] == r['disposition']))
    mentioned_none = sum(1 for r in rows if r['disposition'] == 'NONE' and r['mentioned_in_catalog_text'])
    report = {'schema': 'claude54.comment-census-dispositions.v1', 'census_cuts': cuts,
              'summary_by_custody': {k: dict(v) for k, v in sorted(summary.items())}, 'summary': dict(total),
              'summary_by_match': {k: dict(v) for k, v in by_match.items()},
              'committed_residual_lanes': dict(sorted(lanes.items())),
              'best_disposition_anchored_only': anchored, 'none_but_mentioned_in_catalog_text': mentioned_none, 'rows': rows}
    with open(output, 'x') as s:
        json.dump(report, s, indent=1, ensure_ascii=False)
        s.write('\n')
    print(json.dumps({k: report[k] for k in ('summary', 'summary_by_custody', 'summary_by_match', 'committed_residual_lanes', 'best_disposition_anchored_only',
                                             'none_but_mentioned_in_catalog_text')}, indent=1))


if __name__ == '__main__':
    if len(sys.argv) != 8:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
