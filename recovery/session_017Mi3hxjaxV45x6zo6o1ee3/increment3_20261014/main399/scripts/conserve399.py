"""Conservation check for the main#399 successor binding (reviewer-private).

usage: python3 -I conserve399.py REPO OLD_BASE REVIEWED_HEAD NEW_BASE NEW_HEAD MERGE EXPECTED_TREE
Claims checked:
  * new head parents are [REVIEWED_HEAD, NEW_BASE]; merge parents are [NEW_BASE, NEW_HEAD];
    head and merge trees equal EXPECTED_TREE;
  * OLD_BASE..NEW_BASE (main's movement) and REVIEWED_HEAD..NEW_HEAD (the carry) touch the
    same paths, and every carried path has NEW_BASE's exact entry (mode+type+blob);
  * the PR's changed-path set is identical before and after the carry, and every PR path keeps
    its exact entry on both sides (REVIEWED_HEAD == NEW_HEAD, OLD_BASE == NEW_BASE).
"""
import subprocess
import sys

REPO, A, E, F, H, M, TREE = sys.argv[1:8]
SEMANTIC = ('docs/site/atlas.mjs', 'docs/site/current-model.mjs', 'tests/test_current_site.mjs',
            'tools/current_site_browser_check.py')


def git(*args):
    return subprocess.run(('git', '-C', REPO) + args, capture_output=True, text=True, check=True).stdout


def entries(commit):
    out = {}
    for line in git('ls-tree', '-r', '-z', '--full-tree', commit).split('\0'):
        if line:
            meta, path = line.split('\t', 1)
            out[path] = tuple(meta.split())
    return out


def changed(x, y):
    return {p for p in set(x) | set(y) if x.get(p) != y.get(p)}


bad = []


def check(ok, msg):
    print(('OK   ' if ok else 'FAIL ') + msg)
    if not ok:
        bad.append(msg)


full = {k: git('rev-parse', v).strip() for k, v in (('A', A), ('E', E), ('F', F), ('H', H), ('M', M))}
print('commits', full)
check(git('rev-list', '--parents', '-n1', full['H']).split()[1:] == [full['E'], full['F']], 'head parents [reviewed head, new base]')
check(git('rev-list', '--parents', '-n1', full['M']).split()[1:] == [full['F'], full['H']], 'merge parents [new base, head]')
th, tm = git('rev-parse', full['H'] + '^{tree}').strip(), git('rev-parse', full['M'] + '^{tree}').strip()
check(th == tm == git('rev-parse', TREE).strip(), 'head tree %s == merge tree %s == expected' % (th[:8], tm[:8]))
check(subprocess.run(('git', '-C', REPO, 'merge-base', '--is-ancestor', full['A'], full['F'])).returncode == 0,
      'old base is an ancestor of new base')
ea, ee, ef, eh = entries(full['A']), entries(full['E']), entries(full['F']), entries(full['H'])
main_moved, carry = changed(ea, ef), changed(ee, eh)
print('main movement old->new base: %d paths %s' % (len(main_moved), sorted(main_moved)))
print('carry reviewed->new head:    %d paths %s' % (len(carry), sorted(carry)))
check(main_moved == carry, 'carry touches exactly the paths main moved')
for p in sorted(carry):
    check(eh.get(p) == ef.get(p) and ee.get(p) == ea.get(p),
          'carried %s: head entry == new base %s; reviewed head == old base %s' % (
              p, (eh.get(p) or ('-', '-', 'absent'))[2][:8], (ee.get(p) or ('-', '-', 'absent'))[2][:8]))
pr_old, pr_new = changed(ea, ee), changed(ef, eh)
check(pr_old == pr_new, 'PR path set unchanged by the carry (%d before, %d after)' % (len(pr_old), len(pr_new)))
check(not (pr_new & carry), 'no PR path is a carried path')
head_kept = [p for p in pr_new if eh.get(p) != ee.get(p)]
base_kept = [p for p in pr_new if ef.get(p) != ea.get(p)]
check(not head_kept, 'all %d PR paths keep the reviewed head entry (mode+blob)' % len(pr_new))
check(not base_kept, 'all %d PR paths keep the old base entry on the new base side' % len(pr_new))
sem = [p for p in sorted(pr_new) if p in SEMANTIC]
print('semantic paths in PR: %d; other: %d' % (len(sem), len(pr_new) - len(sem)))
for p in SEMANTIC:
    print('  %s head %s base %s' % (p, eh[p][2][:8], ef[p][2][:8]))
print('RESULT', 'PASS' if not bad else 'FAIL (%d)' % len(bad))
sys.exit(1 if bad else 0)
