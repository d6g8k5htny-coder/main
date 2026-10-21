"""Proof-dependency graph of the SIDE24 V3 supplement, with a cycle check.

For every numbered statement (theorem, lemma, proposition, corollary) the script finds its proofs
(the next ``proof`` environment; every ``\\begin{proof}[Proof of ...~\\ref{label}...]`` naming it, joint
proofs included; and a ``\\subsection{Proof of ...~\\ref{label}}`` up to the next sectioning command), collects
every ``\\ref``/``\\eqref`` in the statement and in its proof, maps equation labels to the statement
that owns them (the statement or proof in which the display appears), and builds the directed graph
"A's proof uses B". It then reports:
  1. strongly connected components of size > 1, each with its internal references in context, so that a
     reader can see whether the component is a circularity or a statement-level artefact (a pointer, a
     definition used from another statement, or a dependency that is acyclic item by item), and
  2. statements that have no proof of their own.
Self-references (a proof citing its own statement's displays) are not edges. The script does not judge
mathematics: an edge is a reference, not a verified logical use.

The seven statement-level components of draft 13 are listed in KNOWN with the reason each is acyclic at
the level of items, definitions and pointers; the script exits 1 only if a component outside that list
appears, or if the internal edges of a listed component change (KNOWN_EDGES), so it is a regression check
for later drafts, not a proof of anything. It exits 2, before building the graph, if the directory or any of
the thirteen section files is missing, or if a \\label is defined twice (a duplicate would overwrite a
statement and could hide a component).

Exit codes: 0 no unexplained component; 1 an unexplained or changed component; 2 invalid input.

Usage: python3 dep_graph_check.py SUPPDIR [--dot out.dot]
"""
import os
import re
import sys
from collections import defaultdict

STMT_ENVS = ('theorem', 'lemma', 'proposition', 'corollary')
OTHER_ENVS = ('definition', 'remark', 'convention', 'hypothesis', 'example', 'claim')
ALL_ENVS = STMT_ENVS + OTHER_ENVS

KNOWN = {
    frozenset(['prop:parent-KR', 'prop:parent-ledger']):
        'S12.23 uses S12.22(a),(b); S12.22(c) uses S12.23: acyclic item by item (stated in both proofs).',
    frozenset(['lem:parent-generic', 'prop:parent-push', 'prop:parent-near', 'thm:parent-A', 'thm:parent-B', 'thm:parent-C']):
        'two pointers close the loop: S12.24 names the density of Theorem B (its proof uses only S12.23(b)), and S12.19(b) '
        'remarks that the proof of Theorem C uses it; neither is a logical use.',
    frozenset(['lem:selection-slope', 'lem:selection-capture', 'lem:selection-robust']):
        'S5.30 (robustness) states margins for the inequalities of S5.26-S5.29 and is proved from them; the closing '
        'sentence of the proof of S5.26 points forward to S5.30 for the perturbed field, which is not used in S5.26.',
    frozenset(['lem:selection-collarchart', 'lem:selection-collarfactor', 'lem:selection-singchart', 'lem:selection-singfactor',
               'lem:selection-finsing', 'lem:selection-fincol']):
        'the finite-scale display of S5.20 is proved by S5.40(d),(e), whose proof uses only the chart, Jacobian, residual limits '
        'and covariances of S5.16-S5.21, proved there (stated in the proof of S5.20); the statement of S5.16 points to S5.41.',
    frozenset(['prop:coefficient-iso', 'cor:coefficient-BF']):
        'the proof of S3.19 points to S3.20 for a comparison; S3.20 uses the displays of S3.19.',
    frozenset(['prop:correction-firstvar', 'thm:correction-main']):
        'the statement of S4.24 uses the polynomial P_d defined in a display of the statement of S4.4; the proof of S4.4 uses S4.24.',
    frozenset(['lem:remainder-W', 'prop:remainder-Z', 'thm:remainder-vanish', 'thm:remainder-rate']):
        'S8.4(ii),(iii) and the finiteness of B are proved directly; S8.9 uses S8.14 and those items; the limit in S8.4(i) '
        'uses (S8.16) and (S8.20); the elder paragraph of S8.6 then uses that limit (stated in S8.6 of draft 12). '
        'S8.14 reuses a decomposition display from the proof of S8.6, not its conclusion.',
}

FILES = ['supp_s1_setting.tex', 'supp_s2_contact.tex', 'supp_s3_coefficient.tex', 'supp_s4_correction.tex',
         'supp_s5_selection.tex', 'supp_s6_limit.tex', 'supp_s7_rate.tex', 'supp_s8_remainder.tex',
         'supp_s9_cusp.tex', 'supp_s10_third.tex', 'supp_s11_ctwo.tex', 'supp_s12_parent.tex', 'supp_s13_cap.tex']

# Number of graph edges inside each KNOWN component when its explanation was written. A different count means
# the explanation must be reviewed again, so the script then treats the component as unexplained.
KNOWN_EDGES = {
    frozenset(['prop:parent-KR', 'prop:parent-ledger']): 2,
    frozenset(['lem:parent-generic', 'prop:parent-push', 'prop:parent-near', 'thm:parent-A', 'thm:parent-B', 'thm:parent-C']): 7,
    frozenset(['lem:selection-slope', 'lem:selection-capture', 'lem:selection-robust']): 4,
    frozenset(['lem:selection-collarchart', 'lem:selection-collarfactor', 'lem:selection-singchart', 'lem:selection-singfactor',
               'lem:selection-finsing', 'lem:selection-fincol']): 12,
    frozenset(['prop:coefficient-iso', 'cor:coefficient-BF']): 2,
    frozenset(['prop:correction-firstvar', 'thm:correction-main']): 2,
    frozenset(['lem:remainder-W', 'prop:remainder-Z', 'thm:remainder-vanish', 'thm:remainder-rate']): 7,
}


def strip_comments(s):
    return re.sub(r'(?<!\\)%.*', '', s)


def load_numbers(aux_path):
    nums = {}
    if not os.path.exists(aux_path):
        return nums
    for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^{}]*)\}', open(aux_path, encoding='utf-8', errors='replace').read()):
        nums[m.group(1)] = m.group(2)
    return nums


def main():
    d = sys.argv[1]
    dot_out = None
    if '--dot' in sys.argv:
        dot_out = sys.argv[sys.argv.index('--dot') + 1]
    aux_nums = load_numbers(os.path.join(d, 'supp_main.aux'))
    nums = {}   # computed from the sources: one shared counter per section, S<n>.<i>

    # Input validation: every section file must exist, and no \\label may be defined twice.
    if not os.path.isdir(d):
        print('INPUT_ERROR: not a directory: %s' % d)
        sys.exit(2)
    missing = [fn for fn in FILES if not os.path.exists(os.path.join(d, fn))]
    if missing:
        print('INPUT_ERROR: missing section files: %s' % ', '.join(missing))
        sys.exit(2)
    seen = {}
    dups = []
    for fn in FILES:
        for lab in re.findall(r'\\label\{([^}]*)\}', strip_comments(open(os.path.join(d, fn), encoding='utf-8').read())):
            if lab in seen:
                dups.append('%s (%s, %s)' % (lab, seen[lab], fn))
            else:
                seen[lab] = fn
    if dups:
        print('INPUT_ERROR: duplicate labels: %s' % '; '.join(dups))
        sys.exit(2)

    # Pass 1: statements (label, env, file, span) and proofs (span, explicit target).
    statements = []          # dicts with label, env, file, start, end, text
    proofs = []              # dicts with file, start, end, text, explicit (label or None)
    eq_owner = {}            # equation label -> statement label (by containment)
    stmt_by_label = {}
    for sec_no, fn in enumerate(FILES, 1):
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            continue
        s = strip_comments(open(path, encoding='utf-8').read())
        counter = 0
        for m in re.finditer(r'\\begin\{(' + '|'.join(ALL_ENVS) + r')\}(\[[^\]]*(?:\[[^\]]*\][^\]]*)*\])?(.*?)\\end\{\1\}', s, re.S):
            env = m.group(1)
            body = m.group(3)
            lab = re.search(r'\\label\{([^}]*)\}', body)
            label = lab.group(1) if lab else '%s@%s:%d' % (env, fn, m.start())
            counter += 1
            nums[label] = 'S%d.%d' % (sec_no, counter)
            st = dict(label=label, env=env, file=fn, start=m.start(), end=m.end(), text=body)
            statements.append(st)
            stmt_by_label[label] = st
            for e in re.finditer(r'\\label\{([^}]*)\}', body):
                if e.group(1) != label:
                    eq_owner.setdefault(e.group(1), label)
        for m in re.finditer(r'\\begin\{proof\}(\[(.*?)\])?(.*?)\\end\{proof\}', s, re.S):
            head = m.group(2) or ''
            refs = re.findall(r'\\ref\{([^}]*)\}', head)
            proofs.append(dict(file=fn, start=m.start(), end=m.end(), text=m.group(3), explicit=refs, head=head))
        # a subsection titled "Proof of ...\ref{X}" is a proof of X up to the next sectioning command
        for m in re.finditer(r'\\(?:sub)*section\{((?:[^{}]|\{[^{}]*\})*?Proof of(?:[^{}]|\{[^{}]*\})*?)\}', s):
            refs = re.findall(r'\\ref\{([^}]*)\}', m.group(1))
            if not refs:
                continue
            nxt = re.search(r'\\(?:sub)*section\{', s[m.end():])
            end = m.end() + (nxt.start() if nxt else len(s) - m.end())
            proofs.append(dict(file=fn, start=m.start(), end=end, text=s[m.end():end], explicit=refs, head=m.group(1)))

    if aux_nums:
        mism = [(l, nums[l], aux_nums[l]) for l in nums if l in aux_nums and nums[l] != aux_nums[l]]
        print('numbering: %d statements numbered from the sources; %d differ from supp_main.aux' % (len(nums), len(mism)))
        for l, a, b in mism[:10]:
            print('   MISMATCH', l, a, b)

    # Pass 2: attach proofs to statements.
    proof_of = defaultdict(list)
    for p in proofs:
        named = [r for r in p['explicit'] if r in stmt_by_label]
        if named:
            for r in named:
                proof_of[r].append(p)
            continue
        # nearest preceding statement in the same file
        cands = [st for st in statements if st['file'] == p['file'] and st['end'] <= p['start']]
        if not cands:
            print('WARNING: proof without a statement in', p['file'], 'at', p['start'])
            continue
        st = max(cands, key=lambda t: t['end'])
        proof_of[st['label']].append(p)
    for st in statements:
        for p in proof_of.get(st['label'], []):
            for e in re.finditer(r'\\label\{([^}]*)\}', p['text']):
                eq_owner.setdefault(e.group(1), st['label'])

    # Pass 3: edges.
    def targets(text):
        out = set()
        for r in re.findall(r'\\(?:eq)?ref\{([^}]*)\}', text):
            for lab in r.split(','):
                lab = lab.strip()
                if lab in stmt_by_label:
                    out.add(lab)
                elif lab in eq_owner:
                    out.add(eq_owner[lab])
        return out

    # statements proved together (one joint proof) are not dependencies of one another
    joint = defaultdict(set)
    for p in proofs:
        named = [r for r in p['explicit'] if r in stmt_by_label]
        if len(named) > 1:
            for r in named:
                joint[r] |= set(named)
    edges = defaultdict(set)
    numbered = [st for st in statements if st['env'] in STMT_ENVS]
    for st in numbered:
        deps = targets(st['text'])
        for p in proof_of.get(st['label'], []):
            deps |= targets(p['text'])
        deps.discard(st['label'])
        deps -= joint.get(st['label'], set())
        edges[st['label']] = {t for t in deps if stmt_by_label[t]['env'] in STMT_ENVS}
    if joint:
        groups = sorted({tuple(sorted(v)) for v in joint.values()})
        print('joint proofs (statements proved together, no edges among them): %d' % len(groups))

    # Tarjan SCC.
    index = {}
    low = {}
    stack = []
    onstack = set()
    sccs = []
    counter = [0]
    sys.setrecursionlimit(10000)

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        onstack.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for st in numbered:
        if st['label'] not in index:
            strong(st['label'])

    def name(label):
        st = stmt_by_label[label]
        return '%s %s' % (st['env'].capitalize(), nums.get(label, '?')) + ' [' + label + ']'

    n_edges = sum(len(v) for v in edges.values())
    print('statements: %d numbered (%s), %d other; proofs: %d; edges: %d' % (
        len(numbered), ', '.join('%d %s' % (sum(1 for s in numbered if s['env'] == e), e) for e in STMT_ENVS),
        len(statements) - len(numbered), len(proofs), n_edges))
    def numkey(label):
        m = re.match(r'S(\d+)\.(\d+)', nums.get(label, ''))
        return ((int(m.group(1)), int(m.group(2))) if m else (999, 0), label)
    # deterministic order: members by statement number, components by their first member
    bad = sorted((sorted(c, key=numkey) for c in sccs if len(c) > 1), key=lambda c: numkey(c[0]))
    def internal(c):
        cs = set(c)
        return sum(1 for a in c for b in edges[a] if b in cs)
    unknown = [c for c in bad if frozenset(c) not in KNOWN or (KNOWN_EDGES and KNOWN_EDGES.get(frozenset(c)) != internal(c))]
    print('nontrivial strongly connected components: %d (%d explained in KNOWN, %d not)' % (len(bad), len(bad) - len(unknown), len(unknown)))
    for c in bad:
        print('  COMPONENT:', '; '.join(name(x) for x in c))
        print('     internal edges: %d' % internal(c))
        if frozenset(c) in KNOWN and c not in unknown:
            print('     explained:', KNOWN[frozenset(c)])
        elif frozenset(c) in KNOWN:
            print('     CHANGED: internal edges differ from KNOWN_EDGES (%s); review the explanation' % KNOWN_EDGES.get(frozenset(c)))
        else:
            print('     NOT EXPLAINED: a new statement-level component')
        cs = set(c)
        for a in c:
            texts = [('statement', stmt_by_label[a]['text'])] + [('proof', p['text']) for p in proof_of.get(a, [])]
            for kind, t in texts:
                for m in re.finditer(r'\\(?:eq)?ref\{([^}]*)\}', t):
                    lab = m.group(1).strip()
                    tgt = lab if lab in stmt_by_label else eq_owner.get(lab)
                    if tgt in cs and tgt != a:
                        ctx = ' '.join(t[max(0, m.start() - 90):m.end() + 30].split())
                        print('     %s (%s) -> %s via %s: ...%s...' % (nums.get(a, a), kind, nums.get(tgt, tgt), lab, ctx))
    unproved = [st for st in numbered if not proof_of.get(st['label'])]
    print('numbered statements without a proof of their own: %d' % len(unproved))
    for st in unproved:
        print('  ', name(st['label']), '(%s)' % st['file'])
    # forward references: A's proof uses B with B numbered after A (allowed, reported for information)
    def key(label):
        n = nums.get(label, '')
        m = re.match(r'S(\d+)\.(\d+)', n)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 0)
    fwd = [(a, b) for a in edges for b in edges[a] if key(b) > key(a)]
    print('forward uses (a proof uses a later-numbered statement; allowed when acyclic): %d' % len(fwd))
    for a, b in sorted(fwd, key=lambda ab: (key(ab[0]), key(ab[1]), ab[0], ab[1])):
        print('   %s uses %s' % (name(a), name(b)))
    if dot_out:
        with open(dot_out, 'w', encoding='utf-8') as f:
            f.write('digraph supp {\n rankdir=LR; node [shape=box, fontsize=9];\n')
            for a in sorted(edges, key=numkey):
                for b in sorted(edges[a], key=numkey):
                    f.write(' "%s" -> "%s";\n' % (nums.get(a, a), nums.get(b, b)))
            f.write('}\n')
        print('dot written to', dot_out)
    print('RESULT:', 'NO_UNEXPLAINED_COMPONENT' if not unknown else 'UNEXPLAINED_COMPONENT_FOUND')
    sys.exit(1 if unknown else 0)


if __name__ == '__main__':
    main()
