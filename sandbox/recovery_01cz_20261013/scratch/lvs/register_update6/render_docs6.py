"""Round-6 re-render (usage: render_docs6.py RUN6 M6): as round 5, with the eight P15Interpolation theorems bound to their hosted receipts
(run RUN6 on PR head 0dcb0016, first receipt run 38149524759 on PR head 88c21058; base 79731b96, the head of PR #422). Derived from render_docs5 of the Declarations and Claims sections of README.md and SCOPE.md of the statement packet from
REGISTER.json after the round-3 proof-packet theorems (hazard interpolation and the F3 enclosure), with the prose edits
those theorems make necessary. Theorems with no hosted receipt yet (added after PR head 79731b96) are marked † and said so.

Scientific effect: NONE. The tables transcribe the register; they decide nothing. Everything outside the two
regenerated sections is preserved byte for byte except the exact-string prose edits listed in PROSE.
"""
import importlib.util
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path, PurePosixPath

SCRATCH = Path('<SCRATCH>')
import os
PK = Path(os.environ.get('RD5_PK', str(SCRATCH / 'repos/math/specifications/lean_statements_20261010')))
PACKAGE = 'specifications/lean_statements_20261010'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_SIDE24 = 'd6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)'
MAIN_P15 = 'd6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)'
BASE = 'b2d70d77876a10c365f5db5d3b6cafcf180af4e2'
RUN1, C1, M1 = '38116277597', '00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'
RUN2, C2, M2 = '38121713840', '910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'
RUN3, C3, M3 = '38122386459', 'd9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'
RUN4, C4, M4 = '38124262242', '79731b96', '318ff194eecc42c39a6b27c1d7bf03f30d6a34d9'   # last head with a 52-theorem receipt
RUN6, M6 = sys.argv[1], sys.argv[2]
assert RUN6.isdigit() and re.fullmatch(r'[0-9a-f]{40}', M6)
C6, RUNA, C5A, M5A = '0dcb0016', '38149524759', '88c21058', '3306233bbdeadb15d9b4b07b94cc52a80ed8caeb'
BASE2 = '79731b961fac555f33be3cd90e379b246a5a80e3'
HEADER_BIND = (f'52 kernel-checked by hosted run {RUN2}: PR head {C2}, actual tested merge {M2[:8]}… with base {BASE[:8]}…; the 8 of P15Interpolation by '
               f'run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base {BASE2[:8]}… (the head of PR #422); same trees as the heads')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check = load_module('stmt_check', PK / 'check.py')
import json as _json
RECEIPTED52 = set(_json.loads((SCRATCH / 'lvs/register_update5/receipted_52.json').read_text())['all_52'])
assert len(RECEIPTED52) == 52, len(RECEIPTED52)
RECEIPTED = RECEIPTED52 | {s['declaration'] for r in _json.loads((PK / 'REGISTER.json').read_text(encoding='utf-8'))['claims'] for s in r['statements']
                           if s['package'] == PROOF_PACKAGE and s['module'] == 'SpecificationProofs/P15Interpolation.lean'}
assert len(RECEIPTED) == 60, len(RECEIPTED)
MODULES = list(check.MODULES)
reg = json.loads((PK / 'REGISTER.json').read_text(encoding='utf-8'))
rows = reg['claims']
sources, _ = check.load_sources(PK, None, reg)


def cell(s):
    return str(s).replace('|', '\\|').replace('\n', ' ')


# ----------------------------------------------------------------------------- declarations in module order (live scan, not a cached file)
scan = []
for m in MODULES:
    for d in check.scan_module((PK / m).read_text(encoding='utf-8'), PurePosixPath(m).stem, sources):
        scan.append({'module': m, **d})
order = {(r['module'], r['name']): (MODULES.index(r['module']), r['line']) for r in scan}
src_of = {(r['module'], r['name']): r['source_id'] for r in scan if r['kind']}

decls = OrderedDict()
for r in rows:
    for s in r['statements']:
        if s['package'] != PACKAGE:
            continue
        key = (s['module'], s['declaration'])
        if key not in decls:
            decls[key] = {'kind': s['kind'], 'anchor': s['anchor'], 'dnc': s['does_not_claim'], 'claims': [], 'proofs': []}
        decls[key]['claims'].append(r['claim_id'])
# proof-packet theorems per Prop (declaration name -> theorem names, deduplicated, in row order)
for r in rows:
    for s in r['statements']:
        if s['package'] == PROOF_PACKAGE:
            prop = None
            # the theorem's Prop is the registered local statement of the same row whose anchor it copies
            for loc in r['statements']:
                if loc['package'] == PACKAGE and loc['anchor'] == s['anchor'] and loc['kind'] == 'def_prop':
                    prop = (loc['module'], loc['declaration'])
            assert prop is not None, s['declaration']
            if s['declaration'] not in decls[prop]['proofs']:
                decls[prop]['proofs'].append(s['declaration'])
keys = sorted(decls, key=lambda k: order[k])
assert len(keys) == 150, len(keys)
n_def = sum(1 for k in keys if decls[k]['kind'] == 'def_prop')
n_struct = sum(1 for k in keys if decls[k]['kind'] == 'structure')
n_aux = sum(1 for r in scan if r['kind'] is None)
proved_props = [k for k in keys if decls[k]['proofs']]
proof_theorems = sorted({t for k in keys for t in decls[k]['proofs']})
assert all(decls[k]['kind'] == 'def_prop' for k in proved_props)
PENDING = sorted(set(proof_theorems) - RECEIPTED)
assert RECEIPTED <= set(proof_theorems), sorted(RECEIPTED - set(proof_theorems))
NT, NP, NPEND = len(proof_theorems), len(proved_props), len(PENDING)
dups = [k for k in keys if len(decls[k]['proofs']) > 1]
DUP_TXT = '; `TriangleNeedsThree` is proved twice' if [k[1].split('.', 3)[-1] for k in dups] == ['TriangleNeedsThree'] else \
    ('; proved twice: ' + ', '.join('`' + k[1].split('.', 3)[-1] + '`' for k in dups) if dups else '')
assert NT == NP + sum(len(decls[k]['proofs']) - 1 for k in dups)
assert NPEND == 0, PENDING
N3 = len(RECEIPTED - RECEIPTED52)
PEND_TXT = (f'; the {N3} theorems of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4} with base {BASE}, receipt theorem_count 52), '
            f'are kernel-checked by the hosted run {RUN6} on PR head {C6} (actual tested merge {M6} with base {BASE2}, the head of PR #422; receipt theorem_count 60, '
            f'leanchecker PASS) and were first receipted by run {RUNA} on PR head {C5A} (actual tested merge {M5A} with the same base); same trees as the heads')

intro = (f'{len(keys)} registered declarations ({n_def} `def … : Prop`, {n_struct} interface `structure`), every one with formal progress '
         f'`specified` in this package; the {n_aux} auxiliary definitions and abbreviations of the modules are inventory only and are not '
         'register statements. Names are given without the module namespace `UniversalLaw.Spec.<Module>.`; the source column is the pinned '
         'source named by the docstring\'s `Source:` line. Anchors and "Not established" cells are copied verbatim from the Lean docstrings '
         '(`Anchor:` and `Does not claim:` lines) through REGISTER.json. '
         f'The last column names the theorems of the sibling proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly '
         f'the declaration (namespace `UniversalLaw.Proofs.<Family>.`): {NP} of the {n_def} Props have one ({NT} '
         f'theorems{DUP_TXT}), each listed in the register under the claim rows as a statement entry of kind '
         '`theorem` with label `proved` (the label of a committed proof package; `kernel-checked` appears only in a run receipt: the hosted '
         f'`lean-specification-proofs.yml` run {RUN2} receipted the 52 theorems of PR head {C2} and run {RUN1} the first 38; a pull_request run checks out the '
         f'synthetic merge of the PR head into the base, so each receipt records as checked_commit that merge and not the head: run {RUN2} PR head {C2}, '
         f'actual tested merge {M2} with base {BASE}; run {RUN1} PR head {C1}, actual tested merge {M1} with the same base; run {RUN3} corroborates on PR head {C3}, '
         f'actual tested merge {M3}; each merge has the same tree as its head{PEND_TXT}). That label belongs to the theorem, not to the declaration here, which stays '
         '`specified`, and not to any claim row; a kernel-checked theorem is evidence about its Prop only, and alignment of every statement is '
         '`PENDING_INDEPENDENT_REVIEW`.')


def proofs_cell(d):
    return ', '.join('`' + t.split('.', 2)[-1] + ('†' if t not in RECEIPTED else '') + '`' for t in d['proofs']) if d['proofs'] else '—'


def decl_table(scope: bool):
    out = [intro, '']
    for m in MODULES:
        ks = [k for k in keys if k[0] == m]
        out.append(f'### `{m}` ({len(ks)} registered declarations)')
        out.append('')
        if scope:
            out.append('| declaration | kind | source | exact coverage (verbatim anchor) | Not established (verbatim `Does not claim:`) | proved by (proof packet, `proved`; ' + HEADER_BIND + ') |')
            out.append('|---|---|---|---|---|---|')
        else:
            out.append('| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; ' + HEADER_BIND + ') |')
            out.append('|---|---|---|---|---|')
        for k in ks:
            d = decls[k]
            short = k[1].split('.', 3)[-1]
            if scope:
                out.append(f'| `{short}` | {d["kind"]} | `{src_of[k]}` | {cell(d["anchor"])} | {cell(d["dnc"])} | {cell(proofs_cell(d))} |')
            else:
                out.append(f'| `{short}` | {d["kind"]} | `{src_of[k]}` | {cell(", ".join("`" + c + "`" for c in d["claims"]))} | {cell(proofs_cell(d))} |')
        out.append('')
    return '\n'.join(out).rstrip()


# ----------------------------------------------------------------------------- claims
kinds = Counter(r['kind'] for r in rows)
fps = Counter(r['formal_progress'] for r in rows)
packet_rows = [r for r in rows if r['not_established'].startswith('Statement not extracted in this programme')]
other_rows = [r for r in rows if r not in packet_rows]
n_local_entries = sum(1 for r in rows for s in r['statements'] if s['package'] == PACKAGE)
n_side24 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_SIDE24)
n_p15 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_P15)
n_proof = sum(1 for r in rows for s in r['statements'] if s['package'] == PROOF_PACKAGE)
n_proved = sum(1 for r in rows for s in r['statements'] if s['formal_progress'] == 'proved')
assert n_proved == n_side24 + n_p15 + n_proof
assert fps['proved'] == 0 and fps['kernel-checked'] == 0
assert len({s['declaration'] for r in rows for s in r['statements'] if s['package'] == MAIN_P15}) == 28
claim_intro = (
    f'{len(rows)} claim rows: ' + ', '.join(f'{v} `{k}`' for k, v in kinds.items()) + f'. Claim-level formal progress: {fps["specified"]} '
    f'`specified` (a `def … : Prop` stating the claim itself is registered), {fps["none"]} `none`; no research claim is `proved` or '
    f'`kernel-checked`. {n_local_entries} statement entries point at the {len(keys)} declarations of this package (all `specified`). '
    f'{n_proved} statement entries carry the label `proved` and are theorems of other packages, listed as the arithmetic skeleton or the '
    f'Prop-level proof of the claim they sit under and never as the claim: {n_side24} are the 31 Side24 arithmetic targets of main\'s `formal/` '
    f'at origin/main `f6deeba7`; {n_p15} are the 28 P15 arithmetic targets of main\'s `formal/` at `91cffbb5` (main PR #358, not yet on main; '
    'kernel evidence the hosted workspace-landing run 38096827443), each listed on every P15 row whose layer 0 source it anchors to; '
    f'{n_proof} are theorems of the proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly a registered Prop of '
    f'this package ({NT} theorems, {NP} distinct Props, each listed on every row that registers its Prop; the '
    f'hosted `lean-specification-proofs.yml` run {RUN2} receipted the 52 theorems of PR head {C2} as kernel-checked and run {RUN1} the first 38; a pull_request run checks out the '
    f'synthetic merge of the PR head into the base, which the receipt records as checked_commit: run {RUN2} PR head {C2}, actual tested merge {M2} with base {BASE}; '
    f'run {RUN1} PR head {C1}, actual tested merge {M1} with the same base; run {RUN3} corroborates on PR head {C3}, actual tested merge {M3}; same trees as the heads{PEND_TXT}). Exactly one Prop registered as stating a '
    'claim itself has such a theorem: `DemandOneRefutation` on rows `p15-price-boundary` and `math.p15-price-boundary`; their claim-level label '
    'stays `specified` (the register admits only none/specified at claim level, alignment of the Prop with the source paragraph is '
    'PENDING_INDEPENDENT_REVIEW, and a receipt is about the Prop only), as the rows\' `not established` cells record. `disposition` is the verbatim '
    'LANDING_CLAIMS value or `—` (null). Nothing in these tables is a scientific status, a review verdict or an acceptance; alignment of every '
    'statement is `PENDING_INDEPENDENT_REVIEW`.')


def statements_cell(r):
    loc = [s for s in r['statements'] if s['package'] == PACKAGE]
    s24 = [s for s in r['statements'] if s['package'] == MAIN_SIDE24]
    p15 = [s for s in r['statements'] if s['package'] == MAIN_P15]
    prf = [s for s in r['statements'] if s['package'] == PROOF_PACKAGE]
    assert len(loc) + len(s24) + len(p15) + len(prf) == len(r['statements']), r['claim_id']
    parts = []
    if loc:
        parts.append(f'{len(loc)} specified: ' + ', '.join('`' + s['declaration'].split('.', 3)[-1] + '`' for s in loc))
    if s24:
        parts.append(f'{len(s24)} proved (main, origin/main f6deeba7): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in s24))
    if p15:
        parts.append(f'{len(p15)} proved (main 91cffbb5, PR #358, not on main): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + '`' for s in p15))
    if prf:
        parts.append(f'{len(prf)} proved (proof packet): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + ('†' if s['declaration'] not in RECEIPTED else '') + '`' for s in prf))
    return '; '.join(parts) if parts else '—'


def claim_table_readme():
    out = [claim_intro, '', '| claim_id | kind | disposition (LANDING_CLAIMS, verbatim) | formal_progress | layer 0 source | statements |', '|---|---|---|---|---|---|']
    for r in other_rows:
        src = r['layer0']['source_id'] or '—'
        out.append(f'| `{r["claim_id"]}` | {r["kind"]} | {r["disposition_transcribed"] or "—"} | `{r["formal_progress"]}` | `{src}` | {cell(statements_cell(r))} |')
    out.append('')
    out.append(f'Packets listed by PROOF_INDEX by id only ({len(packet_rows)} rows, kind `proof_index_result`, formal progress `none`, no statements, bytes not pinned here; '
               'each row\'s `not_established` says the statement was not extracted and no verdict is implied):')
    out.append('')
    out.append(', '.join('`' + r['claim_id'] + '`' for r in packet_rows) + '.')
    return '\n'.join(out)


def claim_table_scope():
    out = [claim_intro, '', '| claim_id | kind | disposition | formal_progress | statements | not established | next step |', '|---|---|---|---|---|---|---|']
    for r in other_rows:
        out.append(f'| `{r["claim_id"]}` | {r["kind"]} | {r["disposition_transcribed"] or "—"} | `{r["formal_progress"]}` | {cell(statements_cell(r))} | {cell(r["not_established"])} | {cell(r["next_step"])} |')
    out.append('')
    out.append(f'### Packets listed by id only ({len(packet_rows)} rows)')
    out.append('')
    out.append('Kind `proof_index_result`, formal progress `none`, no statements, no kernel evidence, layer 0 source null (bytes not pinned here). Shared `not_established`: '
               + cell(packet_rows[0]['not_established']) + ' Shared `next_step` (per packet path): ' + cell(packet_rows[0]['next_step']).replace(packet_rows[0]['claim_id'], '<id>'))
    out.append('')
    out.append('| claim_id | title |')
    out.append('|---|---|')
    for r in packet_rows:
        out.append(f'| `{r["claim_id"]}` | {cell(r["title"])} |')
    return '\n'.join(out)


# ----------------------------------------------------------------------------- splice
def splice(text, start_heading, end_heading, body):
    i = text.index('\n' + start_heading + '\n') + 1
    j = text.index('\n' + end_heading + '\n', i) + 1
    return text[:i] + start_heading + '\n\n' + body + '\n\n' + text[j:]


PEND_SHORT = (f'; the {N3} of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4[:8]}…), by run {RUN6}:\n'
              f'PR head {C6}, actual tested merge {M6[:8]}… with base {BASE2[:8]}…, the head of PR #422, same tree as the head')
PEND_BULLET = (f'; the {N3} theorems\n'
               f'  of `P15Interpolation`, added after PR head {C4} (run {RUN4}, actual tested merge {M4[:8]}…), are receipted by run {RUN6}: PR head {C6}, actual tested\n'
               f'  merge {M6[:8]}… with base {BASE2[:8]}…, the head of PR #422, and first by run {RUNA}: PR head {C5A}, actual tested merge {M5A[:8]}…, same base')
PEND_SCOPE = ''
PEND_SCOPE_END = (f'; the 8 of `P15Interpolation` by run {RUN6}: PR head {C6}, actual tested merge {M6[:8]}… with base\n'
                  f'{BASE2[:8]}…, the head of PR #422')
PROSE = {
    'README.md': [
        ('Props (51 of the 141 Props, 52 theorems); the register lists them under the claim rows as\n'
         f'statement entries of kind `theorem` with label `proved` (all of them kernel-checked by the hosted run {RUN2}: PR head {C2},\n'
         f'actual tested merge {M2[:8]}… with base {BASE[:8]}…, same tree as the head),\n',
         f'Props ({NP} of the {n_def} Props, {NT} theorems); the register lists them under the claim rows as\n'
         f'statement entries of kind `theorem` with label `proved` (52 of them kernel-checked by the hosted run {RUN2}: PR head {C2},\n'
         f'actual tested merge {M2[:8]}… with base {BASE[:8]}…, same tree as the head{PEND_SHORT}),\n'),
        ('  packet by path and holds 52 theorems whose type is exactly one of 51 registered Props (P15, Side24\n',
         f'  packet by path and holds {NT} theorems whose type is exactly one of {NP} registered Props (P15, Side24\n'),
        (f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run {RUN2} receipted all of its\n'
         f'  theorems and run {RUN1} the first 38; each receipt\'s checked_commit is the synthetic merge of the PR head into the\n'
         f'  base that a pull_request run checks out, not the head itself: run {RUN2} PR head {C2}, actual tested merge {M2[:8]}…; run {RUN1}\n'
         f'  PR head {C1}, actual tested merge {M1[:8]}…; both with base {BASE[:8]}… and each with the same tree as its head).\n',
         f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run {RUN2} receipted the 52\n'
         f'  theorems of PR head {C2} and run {RUN1} the first 38; each receipt\'s checked_commit is the synthetic merge of the PR head into the\n'
         f'  base that a pull_request run checks out, not the head itself: run {RUN2} PR head {C2}, actual tested merge {M2[:8]}…; run {RUN1}\n'
         f'  PR head {C1}, actual tested merge {M1[:8]}…; both with base {BASE[:8]}… and each with the same tree as its head{PEND_BULLET}).\n'),
    ],
    'SCOPE.md': [
        ('`specifications/lean_statement_proofs_20261011` (52 theorems typed exactly as 51 of the 141 Props;\n'
         'the last column of the declaration tables and the `proved (proof packet …)` entries of the claims table);\n',
         f'`specifications/lean_statement_proofs_20261011` ({NT} theorems typed exactly as {NP} of the {n_def} Props;\n'
         f'the last column of the declaration tables and the `proved (proof packet …)` entries of the claims table{PEND_SCOPE});\n'),
        (f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
         f'hosted run {RUN2} receipted all of its theorems: PR head {C2}, actual tested merge {M2[:8]}… with base {BASE[:8]}…, the\n'
         'synthetic merge a pull_request run checks out, same tree as the head).',
         f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
         f'hosted run {RUN2} receipted the 52 theorems of PR head {C2}: actual tested merge {M2[:8]}… with base {BASE[:8]}…, the\n'
         f'synthetic merge a pull_request run checks out, same tree as the head{PEND_SCOPE_END}).'),
    ],
}

for name, sections in (('README.md', (('## Declarations', '## Claims', decl_table(False)), ('## Claims', '## What this packet is not', claim_table_readme()))),
                       ('SCOPE.md', (('## Declarations', '## Claims and what is not established', decl_table(True)),
                                     ('## Claims and what is not established', '## Not established (by construction)', claim_table_scope())))):
    p = PK / name
    text = p.read_text(encoding='utf-8')
    for start, end, body in sections:
        text = splice(text, start, end, body)
    for old, new in PROSE[name]:
        assert text.count(old) == 1, (name, old[:60])
        text = text.replace(old, new)
    assert not re.search(r'<<[^<>\n]*>>', text), name
    for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW'):
        assert phrase in text
    for para in re.split(r'\n\s*\n', text):
        for run, merge in ((RUN1, M1[:8]), (RUN2, M2[:8]), (RUN3, M3[:8]), (RUN4, M4[:8]), (RUN6, M6[:8]), (RUNA, M5A[:8])):
            if run in para:
                assert merge in para, (name, run, para[:100])
    for line in text.split('\n'):
        if line.startswith('|'):
            for c in line.split(' | '):
                for run, merge in ((RUN1, M1[:8]), (RUN2, M2[:8]), (RUN3, M3[:8]), (RUN4, M4[:8]), (RUN6, M6[:8]), (RUNA, M5A[:8])):
                    assert run not in c or merge in c, (name, run, c[-160:])
    assert '†' not in text and 'no hosted receipt' not in text, name
    p.write_text(text, encoding='utf-8')
    print(name, len(text.encode()), 'bytes')
print({'props_with_proof': NP, 'proof_theorems': NT, 'pending_receipt': NPEND, 'duplicates': [k[1] for k in dups], 'entries': {'local': n_local_entries, 'side24': n_side24, 'p15': n_p15, 'proof': n_proof, 'proved_total': n_proved}})
