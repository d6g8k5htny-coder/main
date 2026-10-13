

# ----------------------------------------------------------------------------- bands, boxes, runs
TOP_OCTAVES = 7            # octaves [2^-(j+1), 2^-j], j = 6..12 (r up to 1/64), split into 8 sub-bands
LOW_OCTAVES = 16           # octaves j = 13..28, split into 4 sub-bands
R_LAST = Fr(1, 2 ** 29)    # last band [0, 2^-29]
COMBO = (Fr(1, 64), Fr(1, 32))     # (X_0, theta) of NOTE sections 4-5


def bands():
    """Sub-bands [r0, r1] covering (0, 1/64]; each octave split into equal sub-bands; the last band is [0, R_LAST]."""
    out = []
    for j in range(6, 6 + TOP_OCTAVES + LOW_OCTAVES):
        hi = Fr(1, 2 ** j)
        lo = hi / 2
        m = 8 if j < 6 + TOP_OCTAVES else 4
        for i in range(m):
            out.append((lo + (hi - lo) * i / m, lo + (hi - lo) * (i + 1) / m))
    out.append((Fr(0), R_LAST))
    require(min(b[0] for b in out) == 0 and max(b[1] for b in out) == Fr(1, 64), 'band cover')
    out = sorted(out)
    require(all(out[i][1] == out[i + 1][0] for i in range(len(out) - 1)), 'bands contiguous')
    return out


def dec_down(x, places=4):
    x = Fr(x)
    n = (x.numerator * 10 ** places) // x.denominator
    s = str(abs(n)).rjust(places + 1, '0')
    return ('-' if n < 0 else '') + s[:-places] + '.' + s[-places:]


def dec_up(x, places=4):
    x = Fr(x)
    n = -((-x.numerator * 10 ** places) // x.denominator)
    s = str(abs(n)).rjust(places + 1, '0')
    return ('-' if n < 0 else '') + s[:-places] + '.' + s[-places:]


def grid_boxes():
    out = []
    for kb, bg in GRID:
        require(bg[0] == 0 and bg[-1] == 1 and all(bg[i] < bg[i + 1] for i in range(len(bg) - 1)), 'b grid')
        for i in range(len(bg) - 1):
            out.append(((Fr(bg[i]), Fr(bg[i + 1])), (Fr(kb[0]), Fr(kb[1]))))
    ks = [kb for kb, _ in GRID]
    require(ks[0][0] == BAND_K[0] and ks[-1][1] == BAND_K[1] and all(ks[i][1] == ks[i + 1][0] for i in range(len(ks) - 1)),
            'k boxes cover K')
    return tuple(out)


CG3_02_LO = Fr('22424320.655069')     # certified lower end of c_G^(3)(0, 2) (Math-#208)


def band_record(args):
    """Per band: the certified bound of sup Q^W(G_r^c)/r^3 (rounded up, 4 places) and the floor of Z_r/r^2 (rounded down,
    6 places) for each box in BOXES order, and the largest tail part (rounded up, 12 places)."""
    r0, r1 = args
    law = band_law(r0, r1)
    last = (r0 == 0)
    bound, zfloor, tails = [], [], []
    for bb, kb in BOXES:
        res = box_bound(law, bb, kb, r0, r1, last_band=last, x0=COMBO[0], theta=COMBO[1])
        bound.append(dec_up(res['total']))
        zfloor.append(dec_down(36 * kb[0] * kb[0] * res['zt'], 6))
        tails.append(res['tail'])
    return {'r': [str(r0), str(r1)], 'bound': bound, 'zfloor': zfloor, 'tail_max': dec_up(max(tails), 12)}


def full_run(procs):
    bl = bands()
    with multiprocessing.Pool(procs) as pool:
        recs = pool.map(band_record, bl)
    return assemble(recs)


def assemble(recs):
    table = {}
    for rs in R_STARS:
        mx, arg, zmin = Fr(0), None, None
        for rec in recs:
            if Fr(rec['r'][1]) <= rs:
                for (bb, kb), bs, zs in zip(BOXES, rec['bound'], rec['zfloor']):
                    v = Fr(bs)
                    if v > mx:
                        mx, arg = v, {'r': rec['r'], 'b': [str(bb[0]), str(bb[1])], 'k': [str(kb[0]), str(kb[1])]}
                    z = Fr(zs)
                    zmin = z if zmin is None or z < zmin else zmin
        table[str(rs)] = {'C': str(mx), 'argmax': arg, 'C_over_cG3_02': dec_up(mx / CG3_02_LO, 6),
                          'C_rstar3': dec_up(mx * rs ** 3, 10), 'z_star': str(zmin)}
    return {'schema': 1, 'object': 'CL-C8-THEOREM-A-CONSTANTS-D3-20261001-v1', 'scientific_effect': 'NONE',
            'certified': True, 'mutant': MUT, 'L_min': L_MIN, 'dimension': 3,
            'band': {'b': [str(BAND_B[0]), str(BAND_B[1])], 'k': [str(BAND_K[0]), str(BAND_K[1])]},
            'parameters': params(), 'C_table': table, 'bands': recs}


def params():
    return {'PBITS': PBITS, 'NX': NX, 'NB': NB, 'COMBO_X0_THETA': [str(COMBO[0]), str(COMBO[1])], 'THETA_E': str(THETA_E),
            'QS': list(QS), 'R0_COUPLING': R0_COUPLING, 'NET_M': NET_M, 'R_STARS': [str(x) for x in R_STARS],
            'TOP_OCTAVES': TOP_OCTAVES, 'LOW_OCTAVES': LOW_OCTAVES, 'R_LAST': str(R_LAST),
            'GRID': [[[str(kb[0]), str(kb[1])], [str(x) for x in bg]] for kb, bg in GRID]}


def results_text(res):
    """RESULTS.json layout: the summary indented, then one line per band record."""
    head = json.dumps({key: val for key, val in res.items() if key != 'bands'}, indent=1, sort_keys=True)
    lines = ',\n'.join('  ' + json.dumps(rec, sort_keys=True) for rec in res['bands'])
    return head[:-2] + ',\n "bands": [\n' + lines + '\n ]\n}\n'


def write_results(res):
    with open(os.path.join(HERE, 'RESULTS.json'), 'w') as fh:
        fh.write(results_text(res))


def load_results():
    with open(os.path.join(HERE, 'RESULTS.json')) as fh:
        return json.load(fh)


def replay_sample():
    """Band indices replayed by --check: the band ending at each r_*, the bands ending at 2^-14, 2^-20, 2^-26, the last band."""
    bl = bands()
    idx = []
    for rs in R_STARS:
        j = max(i for i, b in enumerate(bl) if b[1] == rs)
        idx.append(j)
    idx += [i for i, b in enumerate(bl) if b[1] in (Fr(1, 2 ** 14), Fr(1, 2 ** 20), Fr(1, 2 ** 26))]
    idx.append(0)
    return sorted(set(idx))


def check_run(procs, full=False):
    ref = load_results()
    with open(os.path.join(HERE, 'RESULTS.json')) as fh:
        require(fh.read() == results_text(ref), 'RESULTS.json is not in the canonical layout')
    require(ref['parameters'] == params(), 'parameters differ from RESULTS.json')
    require(ref['mutant'] is None, 'RESULTS.json produced under a mutant')
    require(abs(PI.mid() - Fr('3.14159265358979323846')) < Fr(1, 10 ** 19) and PI.width() < Fr(1, 10 ** 30), 'pi enclosure')
    require(abs(Phi(IV(1)).mid() - Fr('0.841344746068542948585')) < Fr(1, 10 ** 19), 'Phi(1) enclosure')
    require(abs(m3(0).mid() - (Fr(7, 2) - 2 * SQRT2.mid())) < Fr(1, 10 ** 40), 'm_(3,0) closed form')
    bl = bands()
    require([rec['r'] for rec in ref['bands']] == [[str(a), str(b)] for a, b in bl], 'band list differs')
    idx = list(range(len(bl))) if full else replay_sample()
    with multiprocessing.Pool(procs) as pool:          # in order; the first differing band exits (and stops the pool)
        for i, rec in zip(idx, pool.imap(band_record, [bl[i] for i in idx])):
            require(rec == ref['bands'][i], 'band %d (%s) differs from RESULTS.json' % (i, rec['r']))
    # the table is the maximum over the stored records
    asm = assemble(ref['bands'])
    require(asm['C_table'] == ref['C_table'], 'C table is not the maximum of the stored bounds')
    # sanity: every C exceeds the sharp reference-kernel cap coefficient c_G^(3)(0, 2) of Math-#208 (a lower bound for every
    # d = 3 cap-route constant valid at (0, 2) for the reference kernel, which this certificate covers)
    for rs, row in ref['C_table'].items():
        require(Fr(row['C']) > CG3_02_LO, 'C below the sharp cap coefficient')
    print(json.dumps({'check': 'ok', 'bands_replayed': len(idx), 'C_table': {rs: row['C'] for rs, row in ref['C_table'].items()},
                      'z_star': {rs: row['z_star'] for rs, row in ref['C_table'].items()}, 'mutant': MUT}))


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--check-full', action='store_true')
    ap.add_argument('--mutant', choices=MUTANTS)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--mc', nargs=5, metavar=('R', 'B', 'K', 'N', 'SEED'), help='floating end-to-end control')
    args = ap.parse_args()
    MUT = args.mutant
    if args.mc:
        rr, bb, kk, nn, sd = args.mc
        print(json.dumps(mc_event(float(Fr(rr)), float(Fr(bb)), float(Fr(kk)), int(nn), int(sd))))
        return
    if args.check or args.check_full:
        check_run(args.procs, full=args.check_full)
        return
    res = full_run(args.procs)
    write_results(res)
    print(json.dumps({'C': {rs: row['C'] for rs, row in res['C_table'].items()},
                      'z': {rs: row['z_star'] for rs, row in res['C_table'].items()}}))
