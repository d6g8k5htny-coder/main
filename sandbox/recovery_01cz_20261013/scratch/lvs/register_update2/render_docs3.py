"""Re-render the Declarations and Claims sections of README.md and SCOPE.md of the statement packet from REGISTER.json,
and apply the prose edits that the proof packet makes necessary.

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
PK = SCRATCH / 'repos/math/specifications/lean_statements_20261010'
PACKAGE = 'specifications/lean_statements_20261010'
PROOF_PACKAGE = 'd6g8k5htny-coder/Math- specifications/lean_statement_proofs_20261011'
MAIN_SIDE24 = 'd6g8k5htny-coder/main formal/ (UniversalLaw, side24-arithmetic-skeleton)'
MAIN_P15 = 'd6g8k5htny-coder/main formal/ at 91cffbb51d1636f382e9d61a1ecb6f7cd38e1d06 (PR #358, not yet on main)'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


check = load_module('stmt_check', PK / 'check.py')
import json as _json
RECEIPTED = {s['declaration'] for r in _json.loads((PK / 'REGISTER.json').read_text(encoding='utf-8'))['claims'] for s in r['statements'] if s['package'] == PROOF_PACKAGE}
assert len(RECEIPTED) == 52, len(RECEIPTED)
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

intro = (f'{len(keys)} registered declarations ({n_def} `def … : Prop`, {n_struct} interface `structure`), every one with formal progress '
         f'`specified` in this package; the {n_aux} auxiliary definitions and abbreviations of the modules are inventory only and are not '
         'register statements. Names are given without the module namespace `UniversalLaw.Spec.<Module>.`; the source column is the pinned '
         'source named by the docstring\'s `Source:` line. Anchors and "Not established" cells are copied verbatim from the Lean docstrings '
         '(`Anchor:` and `Does not claim:` lines) through REGISTER.json. '
         f'The last column names the theorems of the sibling proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly '
         f'the declaration (namespace `UniversalLaw.Proofs.<Family>.`): {len(proved_props)} of the {n_def} Props have one ({len(proof_theorems)} '
         'theorems; `TriangleNeedsThree` is proved twice), each listed in the register under the claim rows as a statement entry of kind '
         '`theorem` with label `proved` (the label of a committed proof package; `kernel-checked` appears only in a run receipt: the hosted '
         '`lean-specification-proofs.yml` run 38121713840 on commit 910809d6 receipted all of them, run 38116277597 on commit 00d2ac6e the first 38). That label belongs to the theorem, not to the declaration here, which stays '
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
            out.append('| declaration | kind | source | exact coverage (verbatim anchor) | Not established (verbatim `Does not claim:`) | proved by (proof packet, `proved`; kernel-checked by hosted run 38121713840 at 910809d6) |')
            out.append('|---|---|---|---|---|---|')
        else:
            out.append('| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; kernel-checked by hosted run 38121713840 at 910809d6) |')
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
    f'this package ({len(proof_theorems)} theorems, {len(proved_props)} distinct Props, each listed on every row that registers its Prop; the '
    f'hosted `lean-specification-proofs.yml` run 38121713840 on commit 910809d6 receipted all of them as kernel-checked, run 38116277597 on commit 00d2ac6e the first 38). Exactly one Prop registered as stating a '
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


PROSE = {
    'README.md': [
        ('statement entries of kind `theorem` with label `proved` (38 of them kernel-checked by the hosted run 38116277597 on commit 00d2ac6e;\n'
         'the rest await the next run), which changes no\n',
         f'statement entries of kind `theorem` with label `proved` (all of them kernel-checked by the hosted run 38121713840 on commit 910809d6),\n'
         'which changes no\n'),
        ('  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run 38116277597 on commit 00d2ac6e\n'
         '  receipted the 38 theorems of that commit; later theorems await the next run).\n',
         f'  appears only in that packet\'s hosted run receipts (`lean-specification-proofs.yml`; run 38121713840 on commit 910809d6\n'
         '  receipted all of its theorems, run 38116277597 on commit 00d2ac6e the first 38).\n'),
    ],
    'SCOPE.md': [
        ('identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
         'hosted run 38116277597 on commit 00d2ac6e receipted the 38 theorems of that commit; later theorems await the next run).',
         f'identity), never about the claim row it sits under; `kernel-checked` is a run-receipt word (the proof packet\'s\n'
         f'hosted run 38121713840 on commit 910809d6 receipted all of its theorems).'),
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
    p.write_text(text, encoding='utf-8')
    print(name, len(text.encode()), 'bytes')
print({'props_with_proof': len(proved_props), 'proof_theorems': len(proof_theorems), 'entries': {'local': n_local_entries, 'side24': n_side24, 'p15': n_p15, 'proof': n_proof, 'proved_total': n_proved}})
