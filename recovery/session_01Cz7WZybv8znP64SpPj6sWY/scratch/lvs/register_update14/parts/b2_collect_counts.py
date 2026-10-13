def collect(claim_rows):          # render_docs8/9/10/11 verbatim
    out = OrderedDict()
    for r in claim_rows:
        for s in r['statements']:
            if s['package'] != PACKAGE:
                continue
            key = (s['module'], s['declaration'])
            if key not in out:
                out[key] = {'kind': s['kind'], 'anchor': s['anchor'], 'dnc': s['does_not_claim'], 'claims': [], 'proofs': []}
            out[key]['claims'].append(r['claim_id'])
    for r in claim_rows:
        for s in r['statements']:
            if s['package'] == PROOF_PACKAGE:
                prop = None
                for loc in r['statements']:
                    if loc['package'] == PACKAGE and loc['anchor'] == s['anchor'] and loc['kind'] == 'def_prop':
                        prop = (loc['module'], loc['declaration'])
                assert prop is not None, s['declaration']
                if s['declaration'] not in out[prop]['proofs']:
                    out[prop]['proofs'].append(s['declaration'])
    return out


def counts(d):
    ks = list(d)
    return (sum(1 for k in ks if d[k]['kind'] == 'def_prop'), sum(1 for k in ks if d[k]['proofs']),
            len({t for k in ks for t in d[k]['proofs']}))
