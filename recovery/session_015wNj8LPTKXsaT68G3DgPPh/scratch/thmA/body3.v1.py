

# ----------------------------------------------------------------------------- bands, boxes, runs
B_BOXES = ((Fr(0), Fr(1, 64)), (Fr(1, 64), Fr(1, 16)), (Fr(1, 16), Fr(1, 4)), (Fr(1, 4), Fr(1)))
K_BOXES = ((Fr(1, 2), Fr(1)), (Fr(1), Fr(3, 2)), (Fr(3, 2), Fr(7, 4)), (Fr(7, 4), Fr(15, 8)), (Fr(15, 8), Fr(31, 16)),
           (Fr(31, 16), Fr(63, 32)), (Fr(63, 32), Fr(2)))
HIGH_OCTAVES = 3           # octaves [2^-(j+1), 2^-j], j = 6..8 (r up to 1/64), split into 8 sub-bands
TOP_OCTAVES = 4            # octaves j = 9..12, split into 16 sub-bands
LOW_OCTAVES = 16           # octaves j = 13..28, split into 4 sub-bands
R_LAST = Fr(1, 2 ** 29)    # last band [0, 2^-29]


def bands():
    """Sub-bands [r0, r1] covering (0, 1/64]; each octave split into equal sub-bands; the last band is [0, R_LAST]."""
    out = []
    for j in range(6, 9 + TOP_OCTAVES + LOW_OCTAVES):
        hi = Fr(1, 2 ** j)
        lo = hi / 2
        m = 8 if j < 9 else (16 if j < 9 + TOP_OCTAVES else 4)
        for i in range(m):
            out.append((lo + (hi - lo) * i / m, lo + (hi - lo) * (i + 1) / m))
    out.append((Fr(0), R_LAST))
    require(min(b[0] for b in out) == 0 and max(b[1] for b in out) == Fr(1, 64), 'band cover')
    return sorted(out)


def up(x, digits=6):
    """Decimal string of a rational rounded UP to `digits` significant digits (as a string float)."""
    x = Fr(x)
    if x <= 0:
        return '0'
    e = len(str(x.numerator // x.denominator)) if x >= 1 else -len(str(x.denominator // x.numerator))
    scale = Fr(10) ** (digits - e)
    n = -((-x.numerator * scale.numerator) // (x.denominator * scale.denominator))
    return str(Fr(n) / scale) if scale.denominator == 1 else '%s' % (Fr(n) / scale)


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


BOXES = tuple((bb, kb) for bb in B_BOXES for kb in K_BOXES)      # the order of the per-box lists in a band record


def band_record(args):
    """Per band: the certified bound of sup Q^W(G_r^c)/r^3 (rounded up, 4 places) and the floor of Z_r/r^2 (rounded down,
    6 places) for each of the 28 boxes in BOXES order, and the largest tail part (rounded up, 12 places)."""
    r0, r1 = args
    law = band_law(r0, r1)
    last = (r0 == 0)
    bound, zfloor, tails = [], [], []
    for bb, kb in BOXES:
        res = box_bound(law, bb, kb, r0, r1, last_band=last)
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
        table[str(rs)] = {'C': str(mx), 'argmax': arg, 'C_over_cG02': dec_up(mx / Fr('5324360.4426'), 6),
                          'C_rstar3': dec_up(mx * rs ** 3, 10), 'z_star': str(zmin)}
    return {'schema': 1, 'object': 'CL-C8-THEOREM-A-CONSTANTS-PLANAR-20260930-v1', 'scientific_effect': 'NONE',
            'certified': True, 'mutant': MUT, 'L_min': L_MIN,
            'band': {'b': [str(BAND_B[0]), str(BAND_B[1])], 'k': [str(BAND_K[0]), str(BAND_K[1])]},
            'parameters': params(), 'C_table': table, 'bands': recs}


def params():
    return {'PBITS': PBITS, 'NX': NX, 'NB': NB, 'X0': str(X0), 'THETA': str(THETA), 'THETA_E': str(THETA_E),
            'R_STARS': [str(x) for x in R_STARS], 'HIGH_OCTAVES': HIGH_OCTAVES, 'TOP_OCTAVES': TOP_OCTAVES,
            'LOW_OCTAVES': LOW_OCTAVES, 'R_LAST': str(R_LAST),
            'B_BOXES': [[str(a), str(b)] for a, b in B_BOXES], 'K_BOXES': [[str(a), str(b)] for a, b in K_BOXES]}


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
    bl = bands()
    require([rec['r'] for rec in ref['bands']] == [[str(a), str(b)] for a, b in bl], 'band list differs')
    idx = list(range(len(bl))) if full else replay_sample()
    with multiprocessing.Pool(procs) as pool:
        recs = pool.map(band_record, [bl[i] for i in idx])
    for i, rec in zip(idx, recs):
        require(rec == ref['bands'][i], 'band %d (%s) differs from RESULTS.json' % (i, rec['r']))
    # the table is the maximum over the stored records
    require(assemble(ref['bands'])['C_table'] == ref['C_table'], 'C table is not the maximum of the stored bounds')
    # sanity: every C exceeds the sharp reference-kernel cap coefficient c_G(0, 2) of Math-#203 (a lower bound for every
    # cap-route constant valid for all L >= 10, since c_G^{L,R} -> c_G^{ref} as L -> infinity)
    for rs, row in ref['C_table'].items():
        require(Fr(row['C']) > Fr('5324360.4426'), 'C below the sharp cap coefficient')
    print(json.dumps({'check': 'ok', 'bands_replayed': len(idx), 'C_table': {rs: row['C'] for rs, row in ref['C_table'].items()}, 'mutant': MUT}))


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
    print(json.dumps({rs: row['C'] for rs, row in res['C_table'].items()}))

