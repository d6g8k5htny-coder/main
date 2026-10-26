#!/usr/bin/env python3
"""Recheck the math coverage ledger rows whose proof/lemma cell is MISSING.

For every ledger row with a literal ``MISSING`` proof/lemma cell this lists the
unit's statement documents at the row's pinned cut (blob, size, SHA-256), gives
each a structural class from regex markers, merges the reader classes from the
companion ``.reviewed.json``, and attaches the landed post-cut Math- carriers
that touch the row.  It writes ``OUT.json`` and ``OUT.md``; with ``--check`` it
regenerates both in memory and fails if the committed files differ.

Read-only: only ``git rev-parse``, ``ls-tree``, ``cat-file``, ``show``, ``diff``
and ``merge-base`` run against the two local clones.  No network access.

Usage:
  RETRO_SOURCE_CELL_RECHECK_20261026.py MAIN_REPO MATH_REPO [--check]
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'RETRO_SOURCE_CELL_RECHECK_20261026'
REVIEWED = HERE / 'RETRO_SOURCE_CELL_RECHECK_20261026.reviewed.json'
MAIN_REF = '99adffc8902b1fbfb7ab487020f9c0267f28f88d'
LEDGER = 'data/hypergraph/math_coverage_ledger_20261023.md'
CUTS = {'Math-': '9f3fb6918b28e6378c034ebc19e76d432f5391cb',
        'main': 'a487cfd41bfe095842d9d9f5cec26fc23877b028'}
MATH_TIP = 'b29c4b769cac85437b409d152651d5cce4140015'
DOC = re.compile(r'\.(md|tex|txt|lean)$', re.I)
MARKERS = re.compile(r'\b(Theorem|Lemma|Proposition|Corollary|Proof|Claim|Definition|Remark)\b'
                     r'|∎|\\begin\{(theorem|lemma|proof)\}|\$\$|\\\(|\\\[|≤|≥|∫|∑', re.I)
MATH_CLASSES = {'MATH_STATEMENT', 'MATH_REVIEW', 'FORMAL_SOURCE'}
# Landed Math- changes after the cut that touch ledger rows.  Each entry was
# checked with `git diff MERGE^1 MERGE`, and every merge descends from the cut
# and is an ancestor of MATH_TIP (asserted below).
POSTCUT = [
    (262, 'b29c4b769cac85437b409d152651d5cce4140015', 'Math-#366', 'NEW_CARRIER',
     ['tools/lm006_sign_coefficient_replay.py', 'tests/test_lm006_sign_coefficient_replay.py',
      '.github/workflows/lm006-sign-coefficient.yml'],
     'replay tool pins the unit NOTE.md a79f0808 and runs the unit enclose.py'),
    (215, 'b29c4b769cac85437b409d152651d5cce4140015', 'Math-#366', 'DEPENDENCY_PIN_ONLY',
     ['tools/lm006_sign_coefficient_replay.py'],
     'pins reviews/lm006_wrong_sign_asymptotic_20261006/NOTE.md ea914c1b; runs no code of this unit'),
    (261, '67adf884353e1e5e6e453a412ea0dd01fcc4a466', 'Math-#316', 'REVISED_EXISTING_CARRIER',
     ['tools/d3_l4_replay.py', 'tests/test_d3_l4_replay.py'],
     'both files existed at the cut (aed704e7, 47c538b1) but are absent from the row implementation cell'),
    (257, '96633d22e385969cd91ce6b836576e6104680e01', 'Math-#501', 'REVISED_EXISTING_CARRIER',
     ['imports/lifetime_parent_20260925/verify.py'], 'listed in the row; revised after the cut (was 74235319)'),
    (257, '96633d22e385969cd91ce6b836576e6104680e01', 'Math-#501', 'NEW_CARRIER',
     ['imports/lifetime_parent_20260925/test_verify.py'], 'new negative-control test for verify.py'),
    (34, 'c99ad22f33150abc69e3e54b9143a34169971369', 'Math-#500', 'MANIFEST_REPIN',
     ['frontiers/c8_window_coefficient_d6_20261001/SOURCE_FILES.json'],
     'declarative manifest; one blob id expanded to 40 hex'),
]
REPIN_MERGES = [('10f138512fe285e6ef1b0d145c5f4f07f2c1f9d5', 'Math-#508'),
                ('dd2d13b4bf3325232dd008e9256c7262b4481b51', 'Math-#509')]


def git(repo, *args, ok=(0,)):
    proc = subprocess.run(['git', '-C', str(repo), *args], capture_output=True)
    if proc.returncode not in ok:
        raise SystemExit(f'git {" ".join(args)} failed: {proc.stderr.decode()}')
    return proc


def text(repo, *args):
    return git(repo, *args).stdout.decode().strip()


def heuristic(path, body):
    name = path.rsplit('/', 1)[-1].upper()
    hits = len(MARKERS.findall(body))
    if re.search(r'\.(LEAN|LEAN\.TXT)$', name):
        decls = len(re.findall(r'^\s*(?:private\s+|protected\s+|noncomputable\s+)*(?:theorem|lemma)\s',
                               body, re.M))
        return 'FORMAL' if decls else 'FORMAL_NO_THEOREM_DECLS'
    if name.endswith('.TEX'):
        return 'FORMAL' if hits >= 6 else 'NOTE_UNMARKED'
    if name.startswith('README') or 'INDEX' in name:
        return 'README/INDEX' if hits < 6 else 'README/INDEX+MATH'
    if re.match(r'(SCOPE|REPLAY_SCOPE|PROTOCOL|COORDINATION|ATTRIBUTION|BUILD_STATUS|D0_CI)', name):
        return 'PROTOCOL/SCOPE'
    if re.match(r'(AUDIT|RECONCILIATION|REPORT|VALIDATION|RETRACTION|LEDGER|CANDIDATES|CORRECTIONS)', name):
        return 'REVIEW/AUDIT' if hits < 6 else 'REVIEW/AUDIT+MATH'
    return 'MATH_NOTE' if hits >= 6 else 'NOTE_UNMARKED'


def ledger_rows(main_repo):
    body = text(main_repo, 'show', f'{MAIN_REF}:{LEDGER}')
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) >= 8 and cells[0].isdigit():
            yield cells


def build(main_repo, math_repo):
    repos = {'main': main_repo, 'Math-': math_repo}
    for merge, _ in REPIN_MERGES:
        assert git(math_repo, 'merge-base', '--is-ancestor', CUTS['Math-'], merge, ok=(0, 1)).returncode == 0
    for entry in POSTCUT:
        assert git(math_repo, 'merge-base', '--is-ancestor', CUTS['Math-'], entry[1], ok=(0, 1)).returncode == 0
        assert git(math_repo, 'merge-base', '--is-ancestor', entry[1], MATH_TIP, ok=(0, 1)).returncode == 0
    reviewed = json.loads(REVIEWED.read_text())
    by_doc = {(d['row'], d['path']): d for d in reviewed['documents']}
    used = set()
    carriers = {}
    for row, merge, pr, role, paths, note in POSTCUT:
        for p in paths:
            cut_blob = git(math_repo, 'rev-parse', f'{CUTS["Math-"]}:{p}', ok=(0, 128)).stdout.decode().strip()
            carriers.setdefault(row, []).append({
                'path': p, 'landing': pr, 'merge': merge, 'role': role, 'note': note,
                'blob': text(math_repo, 'rev-parse', f'{merge}:{p}'),
                'blob_at_cut': cut_blob if re.fullmatch(r'[0-9a-f]{40}', cut_blob) else None})
    rows, all_cells = [], list(ledger_rows(main_repo))
    math_units = [(int(c[0]), c[3].strip('`')) for c in all_cells if c[2] == 'Math-']
    repins, workflow_repins = [], []
    for merge, pr in REPIN_MERGES:
        for line in text(math_repo, 'diff', '--raw', '--no-abbrev', f'{merge}^1', merge).splitlines():
            meta, p = line.split('\t', 1)
            _, _, before, after, status = meta.split()
            entry = {'path': p, 'status': status, 'landing': pr, 'merge': merge,
                     'blob_before': before, 'blob': after}
            if p.startswith('.github/workflows/'):
                entry['rows_citing'] = [int(c[0]) for c in all_cells if p in c[6]]
                workflow_repins.append(entry)
            else:
                entry['rows'] = [row for row, unit in math_units
                                 if p == unit or p.startswith(unit.rstrip('/') + '/')]
                repins.append(entry)
    for cells in all_cells:
        if cells[5] != 'MISSING':
            continue
        row, repo, unit = int(cells[0]), cells[2], cells[3].strip('`')
        rec = {'row': row, 'repo': repo, 'unit': unit, 'lane': cells[4], 'ledger_impl_cell': cells[6]}
        if cells[4].startswith('PENDING_OPEN_PR'):
            rec.update(cut=None, members_at_cut=None, statement_documents=[],
                       row_result='NOT_CHECKED_PINNED_TO_OPEN_PR')
            rows.append(rec)
            continue
        cut = CUTS[repo]
        members = []
        for line in text(repos[repo], 'ls-tree', '-r', '-l', cut, '--', unit).splitlines():
            meta, path = line.split('\t', 1)
            members.append((path, meta.split()[2], int(meta.split()[3])))
        docs = []
        for path, blob, size in members:
            if not DOC.search(path):
                continue
            data = git(repos[repo], 'cat-file', 'blob', blob).stdout
            body = data.decode('utf-8', 'replace')
            rv = by_doc.get((row, path))
            if rv is None:
                raise SystemExit(f'no reader class for row {row} {path}')
            used.add((row, path))
            docs.append({'path': path, 'blob': blob, 'bytes': size, 'sha256': hashlib.sha256(data).hexdigest(),
                         'marker_hits': len(MARKERS.findall(body)), 'heuristic_class': heuristic(path, body),
                         'reviewed_class': rv['content_class'], 'reviewed_detail': rv['content_detail'],
                         'heuristic_verdict': rv['verdict'], 'evidence': rv['evidence']})
        math_docs = [d['path'] for d in docs if d['reviewed_class'] in MATH_CLASSES]
        rec.update(cut=cut, members_at_cut=len(members), statement_documents=docs,
                   row_result='SOURCE_DOCUMENT_PRESENT' if math_docs else 'NO_MATH_SOURCE_DOCUMENT',
                   math_source_documents=math_docs)
        if str(row) in reviewed['weak_rows']:
            rec['weak_note'] = reviewed['weak_rows'][str(row)]
        if str(row) in reviewed['rule_sensitive_rows']:
            rec['rule_sensitive_note'] = reviewed['rule_sensitive_rows'][str(row)]
        if row in carriers:
            rec['postcut_carriers'] = carriers[row]
        rp = [{k: v for k, v in r.items() if k != 'rows'} for r in repins if row in r['rows']]
        if rp:
            rec['postcut_manifest_repins'] = rp
        rows.append(rec)
    if used != set(by_doc):
        raise SystemExit(f'reader classes without a document: {sorted(set(by_doc) - used)}')
    missing = {r['row'] for r in rows}
    unit_of = {int(c[0]): c[3].strip('`') for c in all_cells}
    other = [{'row': row, 'unit': unit_of[row], **c} for row, cs in sorted(carriers.items())
             if row not in missing for c in cs]
    counts = {}
    for r in rows:
        counts[r['row_result']] = counts.get(r['row_result'], 0) + 1
    return {
        'schema': 'retro-source-cell-recheck/v1', 'not_a_result': True, 'scientific_effect': 'NONE',
        'performer': 'Anthropic Claude, session_01S16677s1SGeAJmN3QMz4ip (Claude S)',
        'pickup': 'https://github.com/d6g8k5htny-coder/main/issues/619#issuecomment-6405600087',
        'ledger': {'repo': 'main', 'commit': MAIN_REF, 'path': LEDGER,
                   'blob': text(main_repo, 'rev-parse', f'{MAIN_REF}:{LEDGER}')},
        'cuts': CUTS, 'math_tip': MATH_TIP, 'marker_regex': MARKERS.pattern,
        'reader_method': reviewed['method'], 'row_rule': reviewed['row_rule'],
        'missing_rows': len(rows), 'row_result_counts': dict(sorted(counts.items())),
        'rows': rows, 'postcut_carriers_on_rows_with_sources': other,
        'postcut_repins': repins, 'postcut_workflow_repins': workflow_repins,
    }


def render(doc):
    rows = doc['rows']
    present = [r for r in rows if r['row_result'] == 'SOURCE_DOCUMENT_PRESENT']
    absent = [r for r in rows if r['row_result'] == 'NO_MATH_SOURCE_DOCUMENT']
    rc = {}
    for r in rows:
        for d in r['statement_documents']:
            rc[d['reviewed_class']] = rc.get(d['reviewed_class'], 0) + 1
    verdicts = {}
    for r in rows:
        for d in r['statement_documents']:
            verdicts[d['heuristic_verdict']] = verdicts.get(d['heuristic_verdict'], 0) + 1
    out = [
        '# Ledger MISSING source-cell recheck — 26 October 2026',
        '',
        '> **NOT A RESULT.** Engineering source accounting only. Scientific effect **NONE**. This record does not edit the',
        '> ledger, RETRO-MATH-BIND-01, any GB-8 file, the graph, the v2 register or any count; the ledger owner decides adoption.',
        '> "Source document present" means a file in the unit states mathematics. It is not a claim that the mathematics is correct.',
        '',
        '## Pins',
        '',
        f"- Ledger: `main:{doc['ledger']['path']}` at `{doc['ledger']['commit']}`, blob `{doc['ledger']['blob']}`.",
        f"- Unit cuts: Math- `{doc['cuts']['Math-']}`, main `{doc['cuts']['main']}` (the cuts the ledger rows were built on).",
        f"- Post-cut carriers: landed Math- merges up to `{doc['math_tip']}`.",
        f"- Pickup: [619/6405600087]({doc['pickup']}). Performer: {doc['performer']}.",
        '',
        '## Result',
        '',
        f"- The ledger has **{doc['missing_rows']}** rows whose proof/lemma cell is `MISSING`.",
        f"- **{len(present)}** of them hold at least one document that states mathematics (reader-assessed).",
        f"- **{len(absent)}** hold no mathematical source document; "
        f"{sum(1 for r in absent if not r['statement_documents'])} of these have no statement document at all.",
        f"- **{doc['row_result_counts'].get('NOT_CHECKED_PINNED_TO_OPEN_PR', 0)}** row is pinned to an open PR and was not checked.",
        '',
        'Reader classes of the ' + str(sum(rc.values())) + ' statement documents: '
        + ', '.join(f'{k} {v}' for k, v in sorted(rc.items())) + '.',
        '',
        '## Method',
        '',
        '1. The script lists every member of each row unit at its pinned cut and takes the `.md`, `.tex`, `.txt` and `.lean` files',
        '   as statement documents, with blob, byte size and SHA-256.',
        '2. It gives each document a structural class from marker counts (`marker_regex` in the JSON). This first pass is',
        '   reproducible but coarse: against the reader pass it was '
        + ', '.join(f'{k} {v}' for k, v in sorted(verdicts.items()))
        + '. ASCII mathematics (`<=`, `integral_`) pushes it too low; README link labels push it too high.',
        f"3. {doc['reader_method']}",
        f"4. {doc['row_rule']}",
        '',
        'Regenerate and check: `python3 docs/retro/RETRO_SOURCE_CELL_RECHECK_20261026.py <main clone> <Math- clone> --check`.',
        '',
        '## Rows with a mathematical source document',
        '',
        '| Row | Unit | Mathematical source documents | Note |',
        '|---|---|---|---|',
    ]
    for r in present:
        docs = '; '.join(f"`{p.rsplit('/', 1)[-1]}` `{next(d['blob'] for d in r['statement_documents'] if d['path'] == p)[:8]}`"
                         for p in r['math_source_documents'])
        note = r.get('weak_note', '')
        out.append(f"| {r['row']} | `{r['unit']}` | {docs} | {note} |")
    out += ['', '## Rows without a mathematical source document', '',
            '| Row | Unit | Documents present (reader class) | Note |', '|---|---|---|---|']
    for r in absent:
        docs = '; '.join(f"`{d['path'].rsplit('/', 1)[-1]}` {d['reviewed_class']}" for d in r['statement_documents'][:6])
        if len(r['statement_documents']) > 6:
            docs += f"; … {len(r['statement_documents']) - 6} more"
        out.append(f"| {r['row']} | `{r['unit']}` | {docs or 'none'} | {r.get('rule_sensitive_note', '')} |")
    out += ['', '## Landed post-cut carriers', '',
            'These Math- merges landed after the cut and touch ledger rows; none of them supplies a proof or lemma source.', '',
            '| Row | Path | Landing | Merge | Blob | Role | Note |', '|---|---|---|---|---|---|---|']
    carried = [(r['row'], c) for r in rows for c in r.get('postcut_carriers', [])]
    carried += [(c['row'], c) for c in doc['postcut_carriers_on_rows_with_sources']]
    for row, c in sorted(carried, key=lambda x: (x[0], x[1]['path'])):
        out.append(f"| {row} | `{c['path']}` | {c['landing']} | `{c['merge'][:8]}` | `{c['blob'][:8]}` | {c['role']} | {c['note']} |")
    out += ['', 'Rows 257 and 261 are not `MISSING` rows; they appear because their implementation carriers changed after the cut.',
            '', '## Manifest and test re-pins (Math-#508, Math-#509)', '',
            'Every non-workflow path these two merges changed, with the ledger rows whose unit contains it. A row that is not a',
            '`MISSING` row is listed here only. None of these changes touches mathematical content.', '',
            '| Rows | Path | Landing | Merge | Blob before → after |', '|---|---|---|---|---|']
    for c in doc['postcut_repins']:
        rows_txt = ', '.join(str(x) for x in c['rows']) or 'no ledger unit'
        out.append(f"| {rows_txt} | `{c['path']}` | {c['landing']} | `{c['merge'][:8]}` | "
                   f"`{c['blob_before'][:8]}` → `{c['blob'][:8]}` |")
    wf = doc['postcut_workflow_repins']
    cited = sorted({r for c in wf for r in c['rows_citing']})
    out += ['', f"The same merges changed {len(wf)} workflow files under `.github/workflows/`, listed with blobs in",
            '`postcut_workflow_repins` in the JSON. #508 changes only `uses:` action pins; #509 changes one',
            '`timeout-minutes` value (25 → 60) in `d2-square-schur.yml`. '
            f"{len(cited)} ledger rows cite one of them in their implementation cell; for those rows:", '',
            '| Workflow | Landing | Blob before → after | Ledger rows citing it |', '|---|---|---|---|']
    for c in wf:
        if c['rows_citing']:
            out.append(f"| `{c['path'].rsplit('/', 1)[-1]}` | {c['landing']} | `{c['blob_before'][:8]}` → `{c['blob'][:8]}` | "
                       + ', '.join(str(x) for x in c['rows_citing']) + ' |')
    out += ['', '## Limits', '',
            '- Only the units of `MISSING` rows were read; other ledger cells, rows and counts were not rechecked.',
            '- The reader classes are a model-assisted judgment and can be wrong; each document carries its quoted evidence in the JSON.',
            '- Formal statement maps (rows 280, 283, 284, 289, 290) and import-only Lean roots (rows 288, 313) are recorded but not',
            '  counted as source documents; see `rule_sensitive_note`.',
            '- Open-PR content is not promoted. Drive, Dropbox and comment-only sources are outside this record.',
            '', 'credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN', '']
    return '\n'.join(out)


def main(argv):
    if len(argv) not in (3, 4) or (len(argv) == 4 and argv[3] != '--check'):
        raise SystemExit(__doc__)
    doc = build(Path(argv[1]), Path(argv[2]))
    payload = json.dumps(doc, indent=1, ensure_ascii=False) + '\n'
    md = render(doc)
    if len(argv) == 4:
        same = (OUT.with_suffix('.json').read_text() == payload and OUT.with_suffix('.md').read_text() == md)
        print('check:', 'OK' if same else 'DIFFERS')
        return 0 if same else 1
    OUT.with_suffix('.json').write_text(payload)
    OUT.with_suffix('.md').write_text(md)
    print(json.dumps({'missing_rows': doc['missing_rows'], 'row_result_counts': doc['row_result_counts']}))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
