#!/usr/bin/env python3
"""Compute the measured values for the batch-4 scope note, PR body and delivery comment.

Usage: fill_b4.py WORKTREE RUNADIR CONTROLDIR VALDIR OUT.json

Reads run A's records, the control records, the merged contract, the GREEN and RED
test logs, the replay log and the full-suite log (whichever exist), and writes a
JSON map placeholder -> text. Placeholders whose inputs are missing are left out
and listed on stderr. Prose that depends on judgment is not generated here.
"""
import collections
import hashlib
import json
import pathlib
import re
import sys

LIFETIME, D3 = 'c8-lifetime-difference-constant', 'c8-theorem-a-constants-d3'
SHORT = {LIFETIME: 'lifetime', D3: 'd3'}
ROWS = [('test_crashes_each_mutant_each_mode', 'crashes'),
        ('test_silent_exits_each_mutant_each_mode', 'silent exits'),
        ('test_wrong_reasons_each_mutant_each_mode', 'wrong reasons'),
        ('test_wrong_mutant_label', 'wrong mutant label'),
        ('test_baseline_contract', 'baseline faults (stderr, wrong value, missing key or newline, malformed)'),
        ('test_json_contract', 'JSON faults (stderr, duplicate, NaN, trailing, extra, missing, malformed)'),
        ('test_stderr_contract', 'stderr-channel faults (stdout, stderr)'),
        ('test_text_baseline_contract', 'text-baseline faults (reformatted line, extra stdout)'),
        ('test_rebound_sources_need_a_new_characterization', 'rebound sources (checker; and a loaded file)')]
TRIGGER = ('test_paths_cover_the_helper_its_inputs_and_these_tests', 'test_contract_step_runs_in_both_modes_before_the_replay',
           'test_the_replay_is_the_only_import_and_names_its_family')


def minutes(s):
    return '%.1f minutes' % (s / 60) if s >= 120 else ('%d s' % round(s) if s >= 10 else '%.1f s' % s)


def main():
    wt, runa, control, val, out = map(pathlib.Path, sys.argv[1:6])
    fills, missing = {}, []
    records = {}
    for fam in (LIFETIME, D3):
        p = runa / (fam + '.json')
        if p.exists():
            records[fam] = {(r['mode'], r['label']): r for r in json.loads(p.read_text())}
    if len(records) == 2:
        lt, d3 = records[LIFETIME], records[D3]
        # mutant tables
        def rows(fam, recs, stream):
            lines = collections.OrderedDict()
            order = [k[1] for k in recs if k[1] is not None]
            for label in order:
                text = recs[('normal', label)][stream]
                reason = text.rstrip('\n') if stream == 'stderr' else json.loads(text)['reason']
                lines.setdefault(reason, []).append(label)
            return lines
        lt_rows = rows(LIFETIME, lt, 'stderr')
        d3_rows = rows(D3, d3, 'stdout')
        fills['@@LIFETIME_MUTANTS@@'] = '\n'.join('| %s | `%s` |' % (', '.join('`%s`' % x for x in labels), reason)
                                                  for reason, labels in lt_rows.items())
        fills['@@D3_MUTANTS@@'] = '\n'.join('| %s | `%s` |' % (', '.join('`%s`' % x for x in labels), reason)
                                            for reason, labels in d3_rows.items())
        fills['@@LIFETIME_MUTANTS_INLINE@@'] = '; '.join('`%s` for %s' % (reason, ', '.join('`%s`' % x for x in labels))
                                                         for reason, labels in lt_rows.items()) + '.'
        fills['@@D3_MUTANTS_INLINE@@'] = '; '.join('`%s` for %s' % (reason, ', '.join('`%s`' % x for x in labels))
                                                   for reason, labels in d3_rows.items()) + '.'
        fills['@@D3_SAMPLE_N@@'] = str(json.loads(d3[('optimized', None)]['stdout'])['bands_replayed'])
        fills['@@D3_FULL_N@@'] = str(json.loads(d3[('normal', None)]['stdout'])['bands_replayed'])
        # timings
        def timing(fam, recs):
            full = recs[('normal', None)]['seconds']
            sample = recs[('optimized', None)]['seconds']
            muts = [r['seconds'] for k, r in recs.items() if k[1] is not None]
            return ('%s\'s `--check-full` took %s beside the other family\'s children, its sampled `--check` %s, '
                    'and each mutant %s to %s' % (SHORT[fam], minutes(full), minutes(sample),
                                                  minutes(min(muts)), minutes(max(muts))))
        fills['@@TIMINGS@@'] = timing(LIFETIME, lt) + '; ' + timing(D3, d3) + '.'
        fills['@@TIMINGS_SHORT@@'] = ('Side by side, the lifetime `--check-full` took %s and the d3 `--check-full` %s.'
                                      % (minutes(lt[('normal', None)]['seconds']), minutes(d3[('normal', None)]['seconds'])))
        fills['@@RUNA_STARTED@@'] = min(r['started'] for recs in records.values() for r in recs.values())
    else:
        missing += ['run A records']
    # control
    ctl = {}
    for fam in (LIFETIME, D3):
        p = control / (fam + '.jsonl')
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r['label'] == 'procs-2':
                    ctl[fam] = r
    if len(ctl) == 2 and len(records) == 2:
        same = all(ctl[f]['stdout'] == records[f][('optimized', None)]['stdout'] and ctl[f]['stderr'] == records[f][('optimized', None)]['stderr']
                   and ctl[f]['returncode'] == 0 for f in ctl)
        fills['@@PROCS_CONTROL@@'] = ('byte-identical stdout and stderr to the `--procs 4` baselines (exit 0; lifetime %s s, d3 %s s)'
                                      % (ctl[LIFETIME]['seconds'], ctl[D3]['seconds'])) if same else 'DIFFERENT OUTPUT: inspect'
    else:
        missing += ['control records']
    # contract
    cpath = wt / 'tools/check_mode_contracts.json'
    data = cpath.read_bytes()
    doc = json.loads(data)
    if LIFETIME in doc['families'] and D3 in doc['families']:
        fills['@@CONTRACT_BYTES@@'] = format(len(data), ',')
        fills['@@CONTRACT_SHA@@'] = hashlib.sha256(data).hexdigest()
        canonical = json.dumps(doc['families'], sort_keys=True, separators=(',', ':')).encode()
        fills['@@FAMILIES_DIGEST@@'] = hashlib.sha256(canonical).hexdigest()
    else:
        missing += ['merged contract']
    # tests
    def ran(path):
        text = path.read_text(errors='replace')
        m = re.search(r'^Ran (\d+) tests? in ([0-9.]+)s', text, re.M)
        ok = re.search(r'^OK$', text, re.M) is not None
        return (int(m.group(1)), float(m.group(2)), ok) if m else None
    greens = {}
    for fam in (LIFETIME, D3, 'all'):
        for mode in ('normal', 'optimized'):
            p = val / ('green_%s_%s.log' % (fam, mode))
            if p.exists() and ran(p):
                greens[(fam, mode)] = ran(p)
    if len(greens) == 6:
        counts = {g[0] for g in greens.values()}
        allok = all(g[2] for g in greens.values())
        fills['@@TEST_COUNT@@'] = ('%d' % greens[(LIFETIME, 'normal')][0]) + ('' if allok else ' (NOT ALL OK)')
        fills['@@TEST_SECONDS@@'] = 'about %d s per mode for lifetime and %d s for d3' % (
            round((greens[(LIFETIME, 'normal')][1] + greens[(LIFETIME, 'optimized')][1]) / 2),
            round((greens[(D3, 'normal')][1] + greens[(D3, 'optimized')][1]) / 2))
        fills['@@TEST_SECONDS_ALL@@'] = '%d s' % round((greens[('all', 'normal')][1] + greens[('all', 'optimized')][1]) / 2)
        fills['@@TEST_COUNTS_ALL@@'] = '%d' % greens[('all', 'normal')][0]
    else:
        missing += ['green logs (%d of 6)' % len(greens)]
    # red
    reds = {}
    for fam in (LIFETIME, D3):
        for mode in ('normal', 'optimized'):
            p = val / ('red_%s_%s.log' % (fam, mode))
            if p.exists():
                c = collections.Counter()
                for line in p.read_text(errors='replace').splitlines():
                    m = re.match(r'^(FAIL|ERROR): (test_\w+) \(', line)
                    if m:
                        c[m.group(2)] += 1
                reds[(fam, mode)] = c
    if len(reds) == 4:
        same_modes = all(reds[(f, 'normal')] == reds[(f, 'optimized')] for f in (LIFETIME, D3))
        table = []
        totals = {f: 0 for f in (LIFETIME, D3)}
        for name, label in ROWS:
            vals = []
            for f in (LIFETIME, D3):
                n = reds[(f, 'normal')].get(name, 0)
                totals[f] += n
                vals.append(str(n) if n else '—')
            if any(v != '—' for v in vals):
                table.append('  | %s | %s | %s |' % (label, vals[0], vals[1]))
        table.append('  | **false acceptances** | **%d** | **%d** |' % (totals[LIFETIME], totals[D3]))
        fills['@@RED_TABLE@@'] = '\n'.join(table)
        fills['@@RED_LIFETIME@@'] = str(totals[LIFETIME])
        fills['@@RED_D3@@'] = str(totals[D3])
        trig = {f: sum(reds[(f, 'normal')].get(t, 0) for t in TRIGGER) for f in (LIFETIME, D3)}
        fills['@@RED_TRIGGER@@'] = '%d/%d' % (trig[LIFETIME], trig[D3])
        other = {f: {k: v for k, v in reds[(f, 'normal')].items() if k not in dict(ROWS) and k not in TRIGGER} for f in (LIFETIME, D3)}
        fills['@@RED_OTHER@@'] = json.dumps(other)
        fills['@@RED_SAME_MODES@@'] = str(same_modes)
    else:
        missing += ['red logs (%d of 4)' % len(reds)]
    # replays
    p = val / 'replay.log'
    if p.exists():
        oks = re.findall(r'REAL_REPLAY_OK (\S+) both-modes (\d+) (True|False) ([0-9.]+)', p.read_text(errors='replace'))
        if len(oks) == 2:
            parts = ['%s %s/7 children `protocol_ok` in %s s' % (SHORT[f], n, sec) for f, n, match, sec in oks]
            allmatch = all(m == 'True' for _, _, m, _ in oks)
            fills['@@REAL_REPLAYS@@'] = ', '.join(parts) + ('; every stdout and stderr digest equals the characterized reference.'
                                                             if allmatch else '; DIGEST MISMATCH')
            fills['@@REAL_REPLAYS_SHORT@@'] = ' and '.join('%s %s/7 `protocol_ok`' % (SHORT[f], n) for f, n, _, _ in oks) + (
                ', every digest equal to the reference' if allmatch else ' DIGEST MISMATCH')
        else:
            missing += ['replay log (%d of 2 jobs)' % len(oks)]
    else:
        missing += ['replay log']
    # suite
    p = val / 'full_suite.log'
    if p.exists() and ran(p):
        n, sec, ok = ran(p)
        text = p.read_text(errors='replace')
        failing = sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+ \([^)]*\))', text, re.M)))
        base = pathlib.Path('/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14/char/real3/full_suite.log')
        base_failing = sorted(set(re.findall(r'^(?:FAIL|ERROR): (test_\w+ \([^)]*\))', base.read_text(errors='replace'), re.M))) if base.exists() else None
        fills['@@FULL_SUITE_N@@'] = str(n)
        fills['@@FULL_SUITE_FAILING@@'] = str(len(failing))
        fills['@@FULL_SUITE_SAME_AS_BASE@@'] = str(failing == base_failing)
        fills['@@FULL_SUITE_NEW_FAILS@@'] = json.dumps(sorted(set(failing) - set(base_failing or [])))
        fills['@@FULL_SUITE_SECONDS@@'] = str(round(sec))
        # every failing block must be the old-git capability failure
        blocks = re.split(r'^=+$', text, flags=re.M)
        fail_blocks = [b for b in blocks if re.match(r'\s*(FAIL|ERROR): test_', b)]
        lazy = sum('no-lazy-fetch' in b for b in fail_blocks)
        fills['@@FULL_SUITE_LAZY@@'] = '%d/%d' % (lazy, len(fail_blocks))
        same = failing == base_failing
        base_n = re.search(r'^Ran (\d+) tests', base.read_text(errors='replace'), re.M).group(1) if base.exists() else '?'
        if same and lazy == len(fail_blocks):
            fills['@@FULL_SUITE@@'] = ('%s tests on this tree against %s on the unchanged tree of `19a55750` (the batch-3 head\'s tree, '
                                       'run on 29 October). The same %d tests fail on both, all at git\'s `--no-lazy-fetch` capability '
                                       'check (local git is too old). None is in this PR\'s files; all 16 check-mode tests and the '
                                       'legacy trigger tests pass.' % (format(n, ','), format(int(base_n), ','), len(failing)))
            fills['@@FULL_SUITE_SHORT@@'] = ('%s tests, the same %d failing as on the unchanged tree, all at git\'s `--no-lazy-fetch` check'
                                             % (format(n, ','), len(failing)))
        else:
            fills['@@FULL_SUITE@@'] = 'SUITE DIFFERS FROM BASE: inspect (%d failing, %d base, lazy %d/%d, new %s)' % (
                len(failing), len(base_failing or []), lazy, len(fail_blocks), fills['@@FULL_SUITE_NEW_FAILS@@'])
            fills['@@FULL_SUITE_SHORT@@'] = 'SUITE DIFFERS FROM BASE: inspect'
    else:
        missing += ['full suite log']
    out.write_text(json.dumps(fills, indent=1, sort_keys=True) + '\n')
    print('fills', len(fills), 'written to', out)
    if missing:
        print('missing:', '; '.join(missing), file=sys.stderr)


if __name__ == '__main__':
    main()
