def build(rows, pending, receipted, header_bind, pend_txt):
    """The four regenerated section bodies (README decl, README claims, SCOPE decl, SCOPE claims) of a register, render_docs10/11's code."""
    decls = collect(rows)
    keys = sorted(decls, key=lambda k: order[k])
    assert len(keys) == 150, len(keys)
    n_def, NP, NT = counts(decls)
    n_struct = sum(1 for k in keys if decls[k]['kind'] == 'structure')
    n_aux = sum(1 for r in scan if r['kind'] is None)
    proof_theorems = {t for k in keys for t in decls[k]['proofs']}
    assert pending <= proof_theorems and proof_theorems - pending <= receipted and not (pending & receipted)
    dups = [k for k in keys if len(decls[k]['proofs']) > 1]
    DUP_TXT = '; `TriangleNeedsThree` is proved twice' if [k[1].split('.', 3)[-1] for k in dups] == ['TriangleNeedsThree'] else \
        ('; proved twice: ' + ', '.join('`' + k[1].split('.', 3)[-1] + '`' for k in dups) if dups else '')
    assert NT == NP + sum(len(decls[k]['proofs']) - 1 for k in dups)
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
             f'actual tested merge {M3}; each merge has the same tree as its head{pend_txt}). That label belongs to the theorem, not to the declaration here, which stays '
             '`specified`, and not to any claim row; a kernel-checked theorem is evidence about its Prop only, and alignment of every statement is '
             '`PENDING_INDEPENDENT_REVIEW`.')

    def proofs_cell(d):
        return ', '.join('`' + t.split('.', 2)[-1] + ('†' if t in pending else '') + '`' for t in d['proofs']) if d['proofs'] else '—'

    def decl_table(scope):
        out = [intro, '']
        for m in MODULES:
            ks = [k for k in keys if k[0] == m]
            out.append(f'### `{m}` ({len(ks)} registered declarations)')
            out.append('')
            if scope:
                out.append('| declaration | kind | source | exact coverage (verbatim anchor) | Not established (verbatim `Does not claim:`) | proved by (proof packet, `proved`; ' + header_bind + ') |')
                out.append('|---|---|---|---|---|---|')
            else:
                out.append('| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; ' + header_bind + ') |')
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

    kinds = Counter(r['kind'] for r in rows)
    fps = Counter(r['formal_progress'] for r in rows)
    packet_rows = [r for r in rows if r['not_established'].startswith('Statement not extracted in this programme')]
    other_rows = [r for r in rows if r not in packet_rows]
    n_local_entries = sum(1 for r in rows for s in r['statements'] if s['package'] == PACKAGE)
    n_side24 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_SIDE24)
    n_p15 = sum(1 for r in rows for s in r['statements'] if s['package'] == MAIN_P15)
    n_proof = sum(1 for r in rows for s in r['statements'] if s['package'] == PROOF_PACKAGE)
    n_proved = sum(1 for r in rows for s in r['statements'] if s['formal_progress'] == 'proved')
    n_total = sum(len(r['statements']) for r in rows)
    assert n_proved == n_side24 + n_p15 + n_proof
    assert fps['proved'] == 0 and fps['kernel-checked'] == 0 and set(fps) <= {'none', 'specified'}, dict(fps)
    claim_intro = (
        f'{len(rows)} claim rows: ' + ', '.join(f'{v} `{k}`' for k, v in kinds.items()) + f'. Claim-level formal progress: {fps["specified"]} '
        f'`specified` (a `def … : Prop` stating the claim itself is registered), {fps["none"]} `none`; no research claim is `proved` or '
        f'`kernel-checked`. {n_local_entries} statement entries point at the {len(keys)} declarations of this package (all `specified`). '
        f'{n_proved} statement entries carry the label `proved` and are theorems of other packages, listed as the arithmetic skeleton or the '
        f'Prop-level proof of the claim they sit under and never as the claim: {n_side24} are the 31 Side24 arithmetic targets of main\'s `formal/` '
        f'at origin/main `f6deeba7`; {n_p15} are the 28 P15 arithmetic targets of main\'s `formal/` at `91cffbb5` (main PR #358, not yet on main; '
        f'kernel evidence the hosted workspace-landing run {WORKSPACE_LANDING_RUN}), each listed on every P15 row whose layer 0 source it anchors to; '
        f'{n_proof} are theorems of the proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly a registered Prop of '
        f'this package ({NT} theorems, {NP} distinct Props, each listed on every row that registers its Prop; the '
        f'hosted `lean-specification-proofs.yml` run {RUN2} receipted the 52 theorems of PR head {C2} as kernel-checked and run {RUN1} the first 38; a pull_request run checks out the '
        f'synthetic merge of the PR head into the base, which the receipt records as checked_commit: run {RUN2} PR head {C2}, actual tested merge {M2} with base {BASE}; '
        f'run {RUN1} PR head {C1}, actual tested merge {M1} with the same base; run {RUN3} corroborates on PR head {C3}, actual tested merge {M3}; same trees as the heads{pend_txt}). Exactly one Prop registered as stating a '
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
            parts.append(f'{len(prf)} proved (proof packet): ' + ', '.join('`' + s['declaration'].split('.', 2)[-1] + ('†' if s['declaration'] in pending else '') + '`' for s in prf))
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

    stats = {'NP': NP, 'NT': NT, 'n_def': n_def, 'dups': [k[1] for k in dups], 'intro': intro, 'claim_intro': claim_intro,
             'entries': {'total': n_total, 'local': n_local_entries, 'side24': n_side24, 'p15': n_p15, 'proof': n_proof, 'proved_total': n_proved},
             'claim_formal_progress': dict(fps), 'claim_rows': len(rows), 'statements_cell': statements_cell, 'other_rows': other_rows}
    return decl_table(False), claim_table_readme(), decl_table(True), claim_table_scope(), stats


def splice(text, start_heading, end_heading, body):
    i = text.index('\n' + start_heading + '\n') + 1
    j = text.index('\n' + end_heading + '\n', i) + 1
    return text[:i] + start_heading + '\n\n' + body + '\n\n' + text[j:]


def outside(text, headings):
    for start, end in headings:
        i = text.index('\n' + start + '\n') + 1
        j = text.index('\n' + end + '\n', i) + 1
        text = text[:i] + start + '\n\n' + text[j:]
    return text


HEADINGS = {'README.md': (('## Declarations', '## Claims'), ('## Claims', '## What this packet is not')),
            'SCOPE.md': (('## Declarations', '## Claims and what is not established'),
                         ('## Claims and what is not established', '## Not established (by construction)'))}


def render_all(rows, pending, receipted, header_bind, pend_txt, base_docs):
    rd, rc, sd, sc, stats = build(rows, pending, receipted, header_bind, pend_txt)
    bodies = {'README.md': (rd, rc), 'SCOPE.md': (sd, sc)}
    out = {}
    for name in ('README.md', 'SCOPE.md'):
        text = base_docs[name]
        for (start, end), body in zip(HEADINGS[name], bodies[name]):
            text = splice(text, start, end, body)
        out[name] = text
    return out, bodies, stats
