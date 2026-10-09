"""Exact axial level connectivity of the source-bound periodic cusp field.

Polynomial arithmetic is unrounded. The verifier checks supplied witnesses;
it never calls the selector or root-isolation producer. Source-exposed same-team
implementation, conditional on the reviewed analytic completeness guards.
"""
from fractions import Fraction as Q
from itertools import count
from surgery_field import Budget, Inconclusive, guard_certificate

Polynomial = tuple[Q, ...]
Interval = tuple[Q, Q]
U0, V0 = Q(3, 2**32), Q(1, 2**47)
A_STAR, R1, DELTA, B = Q(1, 128), Q(1, 32), Q(1, 2**26), Q(4, 9)
ROOT_WIDTH = Q(1, 2**160)
TIE_POLICY = 'smallest_axial_root_ordinal_survives'
_CALL_IDS = count()


def _fraction(x, name):
    if type(x) is not Q:
        raise TypeError(name + ' must be an exact Fraction')


def _interval(x):
    if type(x) is not tuple or len(x) != 2:
        raise TypeError('Interval must be a pair tuple')
    for y in x:
        _fraction(y, 'endpoint')
    if x[0] > x[1]:
        raise ValueError('Reversed interval')


def _polynomial(p):
    if type(p) is not tuple or not p:
        raise TypeError('Nonempty polynomial tuple required')
    for c in p:
        _fraction(c, 'coefficient')
    if p == (Q(0),) or (len(p) > 1 and p[-1] == 0):
        raise ValueError('Nonzero polynomial with nonzero leading coefficient required')


def _controls(u, v, budget):
    _fraction(u, 'u'); _fraction(v, 'v')
    if not -U0 < u < U0 or not -V0 < v < V0:
        raise ValueError('Controls must lie in the strict open rectangle')
    if not isinstance(budget, Budget):
        raise TypeError('Shared Budget required')


def _trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return tuple(p)


def _mul(p, q):
    r = [Q(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i+j] += a*b
    return _trim(r)


def _derivative(p):
    return tuple(Q(i)*p[i] for i in range(1, len(p))) or (Q(0),)


def _divide(p, q, budget, phase):
    budget.charge_division(phase=phase)
    if q == (Q(0),):
        raise ValueError('Zero polynomial divisor')
    r = list(p)
    quotient = [Q(0)] * max(1, len(p)-len(q)+1)
    while len(r) >= len(q) and r != [Q(0)]:
        k = len(r)-len(q)
        c = r[-1]/q[-1]
        quotient[k] += c
        for j, a in enumerate(q):
            r[k+j] -= c*a
        r = list(_trim(r))
    return _trim(quotient), _trim(r)


def _exact_divide(p, q, budget, phase):
    quotient, remainder = _divide(p, q, budget, phase)
    if remainder != (Q(0),):
        raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Nonexact polynomial division')
    return quotient


def _monic(p, budget, phase):
    return _exact_divide(p, (p[-1],), budget, phase + '/scalar_normalization')


def _gcd(p, q, budget, phase):
    while q != (Q(0),):
        budget.charge_gcd(phase=phase)
        _, r = _divide(p, q, budget, phase + '/division')
        p, q = q, r
    return _monic(p, budget, phase)


def _squarefree(p, budget, phase):
    """Yun decomposition in characteristic zero; retain the scalar separately."""
    scalar = p[-1]
    f = _monic(p, budget, phase)
    c = _gcd(f, _derivative(f), budget, phase + '/gcd')
    w = _exact_divide(f, c, budget, phase + '/quotient')
    result, multiplicity = [], 1
    while w != (Q(1),):
        y = _gcd(w, c, budget, phase + '/gcd')
        z = _exact_divide(w, y, budget, phase + '/factor')
        if z != (Q(1),):
            result.append({'factor': z, 'multiplicity': multiplicity})
        w = y
        c = _exact_divide(c, y, budget, phase + '/repeated')
        multiplicity += 1
    return scalar, result


def _eval(p, x, budget, phase):
    budget.charge_polynomial(phase=phase)
    value = Q(0)
    for c in reversed(p):
        value = value*x+c
    return value


def _imul(x, y):
    z = [a*b for a in x for b in y]
    return min(z), max(z)


def _ieval(p, x, budget, phase):
    budget.charge_polynomial(phase=phase)
    value = (Q(0), Q(0))
    for c in reversed(p):
        lo, hi = _imul(value, x)
        value = lo+c, hi+c
    return value


def _sturm(p, budget, phase):
    sequence = [p, _derivative(p)]
    if sequence[-1] == (Q(0),):
        return (p,)
    while True:
        _, remainder = _divide(sequence[-2], sequence[-1], budget, phase + '/division')
        if remainder == (Q(0),):
            return tuple(sequence)
        sequence.append(tuple(-x for x in remainder))


def _variation(sequence, x, budget, phase):
    signs = []
    for i, p in enumerate(sequence):
        y = _eval(p, x, budget, phase + '/p' + str(i))
        if y:
            signs.append(1 if y > 0 else -1)
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _chain_record(p, sequence, lo, hi, budget, phase):
    left = _variation(sequence, lo, budget, phase + '/left')
    right = _variation(sequence, hi, budget, phase + '/right')
    return {'factor': p, 'sequence': sequence, 'left': lo, 'right': hi,
            'left_variations': left, 'right_variations': right, 'root_count': left-right}


def _work(budget):
    # The driver's Budget snapshot owns the full ledger and field records.
    # Nested certificates keep counts and scope without multiplying that ledger.
    snapshot = budget.snapshot()
    names = ('polynomial_evaluations', 'gcd_steps', 'polynomial_divisions',
             'max_branch_splits', 'field_calls', 'primitive_requests',
             'fixture_id', 'attempt_algebra', 'scopes')
    return {name: snapshot[name] for name in names}


def _retain_exact(result, original, entry, point, depth):
    result['roots'].append(dict(factor=original, ordinal=-1, isolator=(point, point),
        exact_value=point, multiplicity=entry['multiplicity'], branch_splits=depth))


def isolate_roots(p: Polynomial, domain: Interval, *, budget: Budget) -> dict:
    _polynomial(p); _interval(domain)
    if not isinstance(budget, Budget):
        raise TypeError('Shared Budget required')
    phase = 'isolate/' + str(next(_CALL_IDS))
    result = dict(status='COMPUTING', polynomial=p, domain=domain, scalar_unit=p[-1],
                  factorization=[], sturm_chains=[], roots=[], checks={}, work={})
    try:
        scalar, factors = _squarefree(p, budget, phase + '/squarefree')
        result['scalar_unit'], result['factorization'] = scalar, factors
        for fi, entry in enumerate(factors):
            original = entry['factor']
            remaining = original
            prefix = phase + '/factor' + str(fi)
            for label, endpoint in (('left', domain[0]), ('right', domain[1])):
                if _eval(remaining, endpoint, budget, prefix + '/' + label) == 0:
                    _retain_exact(result, original, entry, endpoint, 0)
                    remaining = _exact_divide(remaining, (-endpoint, Q(1)), budget, prefix + '/extract')
            if len(remaining) > 1:
                sequence = _sturm(remaining, budget, prefix + '/sturm')
                initial = _chain_record(remaining, sequence, *domain, budget, prefix + '/domain')
                result['sturm_chains'].append(initial)
                pending = [(domain[0], domain[1], 0, initial['root_count'], 'r', remaining, sequence)]
                while pending:
                    lo, hi, depth, number, branch, current, sequence = pending.pop()
                    if number == 0:
                        continue
                    path = prefix + '/' + branch
                    if number == 1 and hi-lo <= ROOT_WIDTH:
                        result['roots'].append(dict(factor=original, ordinal=-1, isolator=(lo, hi),
                            exact_value=None, multiplicity=entry['multiplicity'], branch_splits=depth))
                        result['sturm_chains'].append(_chain_record(current, sequence, lo, hi, budget, path + '/final'))
                        continue
                    budget.branch_splits(depth+1, phase=path + '/split')
                    midpoint = (lo+hi)/2
                    extracted = _eval(current, midpoint, budget, path + '/midpoint') == 0
                    if extracted:
                        _retain_exact(result, original, entry, midpoint, depth+1)
                        current = _exact_divide(current, (-midpoint, Q(1)), budget, path + '/extract')
                        sequence = _sturm(current, budget, path + '/deflated_sturm')
                    left = _chain_record(current, sequence, lo, midpoint, budget, path + '/left')
                    right = _chain_record(current, sequence, midpoint, hi, budget, path + '/right')
                    # Only deflated chains have the extracted point as endpoint.
                    # Continue inside these children without restarting a branch.
                    if extracted and (left['root_count'] or right['root_count']):
                        result['sturm_chains'].extend((left, right))
                    pending.extend([(midpoint, hi, depth+1, right['root_count'], branch+'R', current, sequence),
                                    (lo, midpoint, depth+1, left['root_count'], branch+'L', current, sequence)])
        result['roots'].sort(key=lambda root: root['isolator'][0])
        if any(a['isolator'][1] >= b['isolator'][0] for a, b in zip(result['roots'], result['roots'][1:])):
            raise Inconclusive('INCONCLUSIVE_PRECISION', phase, 'Root isolators not strictly separated')
        for ordinal, root in enumerate(result['roots']):
            root['ordinal'] = ordinal
        result['checks'] = dict(squarefree_identity=True, complete_sturm_counts=True, disjoint_roots=True)
        budget.checkpoint(phase + '/finish')
        result['status'] = 'CERTIFIED'
    except Inconclusive as exc:
        result['status'] = exc.status
        result['failure'] = dict(phase=exc.phase, detail=exc.detail)
    # Partial observations also use sorted contiguous labels; acceptance remains
    # inconclusive when any part of isolation or its final checkpoint failed.
    result['roots'].sort(key=lambda root: root['isolator'][0])
    for ordinal, root in enumerate(result['roots']):
        root['ordinal'] = ordinal
    result['work'] = _work(budget)
    return result


def _require_root_success(record, phase):
    if record['status'] != 'CERTIFIED':
        raise Inconclusive(record['status'], phase, 'Root isolation did not certify')


def _old_criticals():
    return [dict(point=(Q(0), Q(1, 2)), value=Q(2, 9), inertia={'positive':1, 'negative':1, 'zero':0}, superlevel_birth=False),
            dict(point=(Q(1, 2), Q(0)), value=Q(-2, 9), inertia={'positive':1, 'negative':1, 'zero':0}, superlevel_birth=False),
            dict(point=(Q(1, 2), Q(1, 2)), value=Q(-4, 9), inertia={'positive':2, 'negative':0, 'zero':0}, superlevel_birth=False)]


def _height(root, u, v, budget, phase):
    p = (Q(0), v, u/2, Q(0), Q(-1, 4))
    x = root['exact_value']
    if x is not None:
        h = _eval(p, x, budget, phase + '/exact')
        interval = h, h
    elif v == 0 and u > 0:
        # This identity is proved and rechecked modulo x^2-u, not by overlap.
        _, remainder = _divide(p, (-u, Q(0), Q(1)), budget, phase + '/tie_reduction')
        if remainder != (u*u/4,):
            raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Invalid stationary tie identity')
        common = _gcd(root['factor'], (-u, Q(0), Q(1)), budget, phase + '/tie_gcd')
        chain = _sturm(common, budget, phase + '/tie_sturm')
        count_in_box = (_variation(chain, root['isolator'][0], budget, phase + '/tie_left')
                        - _variation(chain, root['isolator'][1], budget, phase + '/tie_right'))
        if count_in_box != 1:
            raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Root does not satisfy tie equality factor')
        h, interval = remainder[0], (remainder[0], remainder[0])
    else:
        h = None
        interval = _ieval(p, root['isolator'], budget, phase + '/interval')
    if root['multiplicity'] > 1:
        axial = 'zero'
    else:
        axial_interval = _ieval((u, Q(0), Q(-3)), root['isolator'], budget, phase + '/hessian')
        if axial_interval[1] < 0:
            axial = 'negative'
        elif axial_interval[0] > 0:
            axial = 'positive'
        else:
            raise Inconclusive('INCONCLUSIVE_PRECISION', phase, 'Axial Hessian sign unresolved')
    transverse = _ieval((Q(-2), Q(0), Q(-1, 2)), root['isolator'], budget, phase + '/transverse')
    if transverse[1] >= 0:
        raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Transverse Hessian not negative')
    return dict(root_ordinal=root['ordinal'], centered_interval=interval, exact_value=h,
                multiplicity=root['multiplicity'], axial_inertia=axial, transverse_inertia='negative', group_id=None)


def _height_groups(heights, u, v, budget, phase):
    ordered = sorted(heights, key=lambda h: h['centered_interval'][0], reverse=True)
    groups = []
    for height in ordered:
        interval = height['centered_interval']
        if groups and interval == groups[-1]['centered_interval'] and interval[0] == interval[1]:
            groups[-1]['root_ordinals'].append(height['root_ordinal'])
        else:
            if groups and groups[-1]['centered_interval'][0] <= interval[1]:
                raise Inconclusive('INCONCLUSIVE_PRECISION', phase, 'Critical heights not separated or proved equal')
            groups.append(dict(id='g'+str(len(groups)), root_ordinals=[height['root_ordinal']], centered_interval=interval, equality_witness=None))
        height['group_id'] = groups[-1]['id']
    for group in groups:
        group['root_ordinals'].sort()
        if len(group['root_ordinals']) > 1:
            if v != 0 or u <= 0:
                raise Inconclusive('INCONCLUSIVE_PRECISION', phase, 'No supported equality witness')
            group['equality_witness'] = dict(factor=(-u, Q(0), Q(1)),
                height_remainder=(u*u/4,), root_ordinals=list(group['root_ordinals']))
            budget.checkpoint(phase + '/equality')
    return groups


def _regular_heights(groups):
    return [DELTA/2] + [(a['centered_interval'][0]+b['centered_interval'][1])/2
        for a, b in zip(groups, groups[1:])] + [-DELTA/2]


def _component_intervals(rootset):
    roots, domain = rootset['roots'], rootset['domain']
    for index in range(len(roots)+1):
        lo = roots[index-1]['isolator'][1] if index else domain[0]
        hi = roots[index]['isolator'][0] if index < len(roots) else domain[1]
        yield index, lo, hi


def _components(rootset, level_id, budget, phase):
    if any(root['multiplicity'] != 1 for root in rootset['roots']):
        raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Regular-level polynomial has a repeated root')
    components, signs = [], []
    for index, lo, hi in _component_intervals(rootset):
        witness = (lo+hi)/2
        value = _eval(rootset['polynomial'], witness, budget, phase + '/interval'+str(index))
        if value == 0:
            raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Regular sign witness vanished')
        signs.append(dict(boundary_root_ordinals=[index-1 if index else None,
                         index if index < len(rootset['roots']) else None], witness=witness, value=value))
        if value > 0:
            if index == 0 or index == len(rootset['roots']):
                raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Positive component meets core boundary')
            components.append(dict(id=level_id+'c'+str(len(components)), boundary_root_ordinals=[index-1,index],
                interior_witness=witness, witness_polynomial_value=value))
    return components, signs


def _inside(witness, component, level):
    left, right = (level['root_certificate']['roots'][j] for j in component['boundary_root_ordinals'])
    return left['isolator'][1] < witness < right['isolator'][0]


def _root_inside(root, component, level):
    left, right = (level['root_certificate']['roots'][j] for j in component['boundary_root_ordinals'])
    return left['isolator'][1] < root['isolator'][0] <= root['isolator'][1] < right['isolator'][0]


def _incidence(upper, lower, budget, phase):
    result = []
    for component in upper['components']:
        witness = component['interior_witness']
        matches = [c for c in lower['components'] if _inside(witness, c, lower)]
        if len(matches) != 1:
            raise Inconclusive('INCONCLUSIVE_PRECISION', phase, 'Unique lower containment unresolved')
        value = _eval(lower['root_certificate']['polynomial'], witness, budget, phase + '/'+component['id'])
        if value <= 0:
            raise Inconclusive('FAIL_IMPLEMENTATION', phase, 'Nonpositive containment witness')
        result.append(dict(upper_level=upper['id'], lower_level=lower['id'], upper_component=component['id'],
                           lower_component=matches[0]['id'], witness=witness, lower_polynomial_value=value))
    return result


def _birth_for(component, lower, group, roots, heights):
    candidates = []
    for ordinal in group['root_ordinals']:
        root = roots[ordinal]
        if _root_inside(root, component, lower) and heights[ordinal]['axial_inertia'] in ('negative', 'zero'):
            candidates.append(ordinal)
    if len(candidates) != 1:
        raise Inconclusive('FAIL_IMPLEMENTATION', 'elder/birth', 'Birth must have one compatible critical root')
    return candidates[0]


def _bar(birth, death, heights, groups):
    h = heights[birth]
    birth_interval, death_interval = h['centered_interval'], death['centered_interval']
    lifetime = (birth_interval[0]-death_interval[1], birth_interval[1]-death_interval[0])
    if lifetime[0] <= 0:
        raise Inconclusive('INCONCLUSIVE_PRECISION', 'elder/lifetime', 'Positive lifetime not separated')
    exact = lifetime[0] if lifetime[0] == lifetime[1] else None
    return dict(birth_root=birth, death_group=death['id'], birth_centered_interval=birth_interval,
        death_centered_interval=death_interval, lifetime_interval=lifetime, lifetime_exact=exact,
        tied_birth=len(next(g for g in groups if g['id'] == h['group_id'])['root_ordinals']) > 1)


def _elder(levels, groups, roots, heights, incidence, budget, *, certificate):
    if levels[0]['components'] or len(levels[-1]['components']) != 1:
        raise Inconclusive('FAIL_IMPLEMENTATION', 'elder/separators', 'Top must be empty and bottom connected')
    # Retain each computed bar and completed group before a checkpoint can fail.
    active, events = {}, certificate['events']
    barcode = certificate['actual_barcode']
    finite = barcode['finite']
    rank = {g['id']: i for i, g in enumerate(groups)}
    for i, group in enumerate(groups):
        upper, lower = levels[i], levels[i+1]
        edges = [e for e in incidence if e['upper_level'] == upper['id']]
        next_active = {}
        event = dict(group_id=group['id'], births=[], continuations=[], merges=[])
        for component in lower['components']:
            predecessors = [e['upper_component'] for e in edges if e['lower_component'] == component['id']]
            incoming = sorted(active[c] for c in predecessors)
            if not incoming:
                birth = _birth_for(component, lower, group, roots, heights)
                event['births'].append(dict(component=component['id'], birth_root=birth))
                survivor = birth
            elif len(incoming) == 1:
                survivor = incoming[0]
                event['continuations'].append(dict(upper_component=predecessors[0], lower_component=component['id'], survivor_birth=survivor))
            else:
                survivor = min(incoming, key=lambda birth: (rank[heights[birth]['group_id']], birth))
                dying = [birth for birth in incoming if birth != survivor]
                event['merges'].append(dict(lower_component=component['id'], incoming_births=incoming, survivor_birth=survivor, dying_births=dying))
                finite.extend(_bar(birth, group, heights, groups) for birth in dying)
            next_active[component['id']] = survivor
        active = next_active
        events.append(event)
        budget.checkpoint('elder/group'+str(i))
    if len(active) != 1:
        raise Inconclusive('FAIL_IMPLEMENTATION', 'elder/bottom', 'Bottom must have one component')
    birth = next(iter(active.values()))
    h = heights[birth]
    essential = [dict(birth_root=birth, birth_centered_interval=h['centered_interval'],
        tied_birth=len(next(g for g in groups if g['id']==h['group_id'])['root_ordinals']) > 1)]
    barcode['essential'] = essential
    barcode['status'] = 'CERTIFIED'


def _projection(u, v, finite):
    discriminant = 4*u**3-27*v**2
    reason = 'EXCLUDED_DEGENERATE' if discriminant == 0 else ('EXCLUDED_TIE' if u > 0 and v == 0 else 'INCLUDED')
    return dict(status=reason if reason != 'INCLUDED' else 'CERTIFIED', count=0 if reason != 'INCLUDED' else len(finite),
                reason=reason, actual_finite_count=len(finite))


def _withdraw(certificate, exc):
    certificate['computed_status_before_withdrawal'] = certificate['status']
    certificate['status'] = exc.status
    certificate['failure'] = dict(phase=exc.phase, detail=exc.detail)
    for key in ('actual_barcode', 'population_projection'):
        record = certificate[key]
        record['computed_status_before_withdrawal'] = record['status']
        record['status'] = exc.status


def select_h0(u: Q, v: Q, *, budget: Budget) -> dict:
    _controls(u, v, budget)
    certificate = dict(schema_version=1, model_id='explicit_periodic_cusp_v1', u=u, v=v, status='COMPUTING',
        guard={}, critical_roots={}, old_criticals=[], critical_heights=[], height_groups=[], levels=[], incidence=[], events=[],
        actual_barcode=dict(status='COMPUTING', essential=[], finite=[], tie_policy=TIE_POLICY),
        population_projection=dict(status='COMPUTING', count=0, reason='UNCOMPUTED', actual_finite_count=0), checks={}, work={})
    try:
        budget.checkpoint('selector/guard')
        certificate['guard'] = guard_certificate()
        certificate['old_criticals'] = _old_criticals()
        p = (-v, -u, Q(0), Q(1))
        roots = isolate_roots(p, (-A_STAR, A_STAR), budget=budget)
        certificate['critical_roots'] = roots
        _require_root_success(roots, 'selector/critical_roots')
        heights = certificate['critical_heights']
        for root in roots['roots']:
            heights.append(_height(root, u, v, budget, 'selector/height'+str(root['ordinal'])))
        groups = _height_groups(heights, u, v, budget, 'selector/groups')
        certificate['height_groups'] = groups
        if not (DELTA/2 > groups[0]['centered_interval'][1] and groups[-1]['centered_interval'][0] > -DELTA/2):
            raise Inconclusive('FAIL_IMPLEMENTATION', 'selector/height_band', 'Critical heights outside separator band')
        for i, h in enumerate(_regular_heights(groups)):
            level = dict(id='l'+str(i), centered_height=h, root_certificate={}, components=[])
            certificate['levels'].append(level)
            q = (-h, v, u/2, Q(0), Q(-1, 4))
            level['root_certificate'] = isolate_roots(q, (-R1/2, R1/2), budget=budget)
            _require_root_success(level['root_certificate'], 'selector/level'+str(i))
            level['components'], level['sign_intervals'] = _components(level['root_certificate'], level['id'], budget, 'selector/level'+str(i))
        for i, (upper, lower) in enumerate(zip(certificate['levels'], certificate['levels'][1:])):
            certificate['incidence'].extend(_incidence(upper, lower, budget, 'selector/incidence'+str(i)))
        _elder(certificate['levels'], groups, roots['roots'], heights, certificate['incidence'], budget, certificate=certificate)
        certificate['population_projection'] = _projection(u, v, certificate['actual_barcode']['finite'])
        certificate['checks'] = dict(analytic_completeness=True, critical_completeness=True,
            regular_level_connectivity=True, nested_component_incidence=True, batched_elder_rule=True)
        budget.checkpoint('selector/final_acceptance')
        certificate['status'] = 'CERTIFIED'
    except Inconclusive as exc:
        _withdraw(certificate, exc)
    certificate['work'] = _work(budget)
    return certificate


class _InvalidCertificate(Exception):
    pass


def _check(condition, name, checks):
    checks[name] = bool(condition)
    if not condition:
        raise _InvalidCertificate(name)


def _same(left, right):
    """Exact record equality must not conflate bool/int/Fraction or list/tuple."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _record_schema(record, required, name, checks):
    _check(type(record) is dict and set(required.split()) <= record.keys(), name+'/required_keys', checks)


def _work_schema(record, name, checks):
    counters = ('polynomial_evaluations', 'gcd_steps', 'polynomial_divisions',
                'max_branch_splits', 'field_calls', 'primitive_requests')
    _record_schema(record, ' '.join(counters)+' scopes', name, checks)
    _check(all(type(record[key]) is int and record[key]>=0 for key in counters), name+'/integer_counts', checks)
    scopes = record['scopes']
    _check(type(scopes) is dict and all(scopes.get(key)=='per_fixture' for key in counters[:4])
           and all(scopes.get(key)=='whole_attempt' for key in counters[4:]), name+'/counter_scopes', checks)


def _verify_roots(record, p, domain, budget, checks, phase):
    """Replay factor identities and counts without invoking isolation."""
    _record_schema(record, 'status polynomial domain scalar_unit factorization sturm_chains roots checks work', phase, checks)
    _check(type(record['status']) is str and type(record['checks']) is dict, phase+'/record_types', checks)
    _work_schema(record['work'], phase+'/work', checks)
    _check(type(record['roots']) is list and type(record['sturm_chains']) is list, phase+'/list_records', checks)
    _check(_same(record['polynomial'], p) and _same(record['domain'], domain), phase+'/input', checks)
    scalar, factors = _squarefree(p, budget, phase+'/squarefree')
    _check(type(record['scalar_unit']) is Q and record['scalar_unit']==scalar and _same(record['factorization'], factors),
           phase+'/factorization', checks)
    rebuilt = (scalar,)
    for entry in factors:
        for _ in range(entry['multiplicity']):
            rebuilt = _mul(rebuilt, entry['factor'])
    _check(rebuilt == p, phase+'/identity', checks)
    roots = record['roots']
    _check(_same([r['ordinal'] for r in roots], list(range(len(roots)))), phase+'/ordinals', checks)
    for i, root in enumerate(roots):
        _interval(root['isolator'])
        lo, hi = root['isolator']
        _check(domain[0] <= lo <= hi <= domain[1] and hi-lo <= ROOT_WIDTH, phase+'/enclosure'+str(i), checks)
        _check(type(root['ordinal']) is int and type(root['multiplicity']) is int and type(root['branch_splits']) is int
               and 0 <= root['branch_splits'] <= 256, phase+'/types'+str(i), checks)
        if i:
            _check(roots[i-1]['isolator'][1] < lo, phase+'/separation'+str(i), checks)
        _polynomial(root['factor'])
        entries = [f for f in factors if _same(f['factor'],root['factor'])]
        _check(len(entries)==1 and entries[0]['multiplicity']==root['multiplicity'], phase+'/multiplicity'+str(i), checks)
        factor = root['factor']
        if root['exact_value'] is not None:
            _fraction(root['exact_value'], 'exact root')
            _check(lo == hi == root['exact_value'] and _eval(factor, lo, budget, phase+'/exact'+str(i))==0,
                   phase+'/exact'+str(i), checks)
        else:
            _check(lo < hi and _eval(factor, lo, budget, phase+'/lo'+str(i)) != 0
                   and _eval(factor, hi, budget, phase+'/hi'+str(i)) != 0, phase+'/nonroot_endpoints'+str(i), checks)
            sequence = _sturm(factor, budget, phase+'/root_sturm'+str(i))
            cnt = _variation(sequence, lo, budget, phase+'/lo_count')-_variation(sequence, hi, budget, phase+'/hi_count')
            _check(cnt == 1, phase+'/one_root'+str(i), checks)
    for i, entry in enumerate(factors):
        factor, endpoints = entry['factor'], 0
        for endpoint in domain:
            if _eval(factor, endpoint, budget, phase+'/domain_endpoint') == 0:
                endpoints += 1
                factor = _exact_divide(factor, (-endpoint,Q(1)), budget, phase+'/endpoint_extract')
        seq = _sturm(factor, budget, phase+'/complete_sturm')
        cnt = endpoints + _variation(seq, domain[0], budget, phase+'/domain_left')-_variation(seq, domain[1], budget, phase+'/domain_right')
        supplied = sum(root['factor']==entry['factor'] for root in roots)
        _check(cnt == supplied, phase+'/complete'+str(i), checks)
    for i, chain in enumerate(record['sturm_chains']):
        factor = chain['factor']
        _polynomial(factor)
        _check(len(factor)>1, phase+'/chain_factor'+str(i), checks)
        # Deflated chain factors must divide a recorded original square-free factor.
        divides = any(_divide(f['factor'], factor, budget, phase+'/chain_factor_division')[1] == (Q(0),) for f in factors)
        _check(divides, phase+'/chain_divisor'+str(i), checks)
        sequence = _sturm(factor, budget, phase+'/chain_sturm')
        _check(_same(chain['sequence'],sequence), phase+'/chain_identity'+str(i), checks)
        lo, hi = chain['left'], chain['right']
        _interval((lo,hi))
        _check(domain[0] <= lo <= hi <= domain[1] and _eval(factor,lo,budget,phase+'/chain_lo') != 0
               and _eval(factor,hi,budget,phase+'/chain_hi') != 0, phase+'/chain_endpoints'+str(i), checks)
        left, right = _variation(sequence,lo,budget,phase+'/chain_left'), _variation(sequence,hi,budget,phase+'/chain_right')
        _check(_same((chain['left_variations'],chain['right_variations'],chain['root_count']), (left,right,left-right)),
               phase+'/chain_count'+str(i), checks)
    for i, root in enumerate(roots):
        if root['exact_value'] is None:
            lo, hi = root['isolator']
            witnesses = [c for c in record['sturm_chains'] if c['left']==lo and c['right']==hi and c['root_count']==1]
            _check(bool(witnesses), phase+'/retained_isolator_witness'+str(i), checks)
    return roots


def _verify_graph(certificate, roots, heights, groups, budget, checks):
    """Independent state replay of births, containment and elder deaths."""
    levels = certificate['levels']
    active, finite, replay_edges = {}, [], []
    rank = {g['id']:i for i,g in enumerate(groups)}
    _check(len(certificate['events']) == len(groups), 'graph/event_count', checks)
    for i, group in enumerate(groups):
        upper, lower, batch = levels[i], levels[i+1], certificate['events'][i]
        _check(batch['group_id']==group['id'], 'graph/group'+str(i), checks)
        edges = []
        for component in upper['components']:
            witness = component['interior_witness']
            matches = [c for c in lower['components'] if _inside(witness,c,lower)]
            _check(len(matches)==1, 'graph/containment/'+component['id'], checks)
            value = _eval(lower['root_certificate']['polynomial'],witness,budget,'verify/containment/'+component['id'])
            _check(value > 0, 'graph/positive/'+component['id'], checks)
            edges.append(dict(upper_level=upper['id'],lower_level=lower['id'],upper_component=component['id'],
                lower_component=matches[0]['id'],witness=witness,lower_polynomial_value=value))
        replay_edges.extend(edges)
        births, continuations, merges, state = [], [], [], {}
        for component in lower['components']:
            predecessor_ids = [e['upper_component'] for e in edges if e['lower_component']==component['id']]
            incoming = sorted(active[x] for x in predecessor_ids)
            if not incoming:
                eligible = []
                for ordinal in group['root_ordinals']:
                    root = roots[ordinal]
                    if _root_inside(root,component,lower) and heights[ordinal]['axial_inertia'] in ('negative','zero'):
                        eligible.append(ordinal)
                _check(len(eligible)==1, 'graph/birth/'+component['id'], checks)
                survivor = eligible[0]
                births.append(dict(component=component['id'],birth_root=survivor))
            elif len(incoming)==1:
                survivor = incoming[0]
                continuations.append(dict(upper_component=predecessor_ids[0],lower_component=component['id'],survivor_birth=survivor))
            else:
                survivor = sorted(incoming,key=lambda n:(rank[heights[n]['group_id']],n))[0]
                deaths = sorted(n for n in incoming if n != survivor)
                merges.append(dict(lower_component=component['id'],incoming_births=incoming,survivor_birth=survivor,dying_births=deaths))
                for birth in deaths:
                    bh, dh = heights[birth]['centered_interval'],group['centered_interval']
                    life = (bh[0]-dh[1],bh[1]-dh[0])
                    _check(life[0]>0,'graph/positive_lifetime/'+str(birth),checks)
                    finite.append(dict(birth_root=birth,death_group=group['id'],birth_centered_interval=bh,
                        death_centered_interval=dh,lifetime_interval=life,lifetime_exact=life[0] if life[0]==life[1] else None,
                        tied_birth=len(next(g for g in groups if g['id']==heights[birth]['group_id'])['root_ordinals'])>1))
            state[component['id']] = survivor
        _check(_same(batch['births'],births) and _same(batch['continuations'],continuations) and _same(batch['merges'],merges),
               'graph/elder_batch'+str(i),checks)
        active=state
    _check(_same(certificate['incidence'],replay_edges),'graph/incidence_witnesses',checks)
    _check(len(active)==1,'graph/one_essential',checks)
    birth=next(iter(active.values()))
    essential=[dict(birth_root=birth,birth_centered_interval=heights[birth]['centered_interval'],
        tied_birth=len(next(g for g in groups if g['id']==heights[birth]['group_id'])['root_ordinals'])>1)]
    barcode=certificate['actual_barcode']
    _check(_same(barcode['finite'],finite) and _same(barcode['essential'],essential) and barcode['tie_policy']==TIE_POLICY,
           'graph/barcode',checks)
    return finite


def verify_h0(u: Q, v: Q, certificate: dict, *, budget: Budget) -> dict:
    _controls(u,v,budget)
    checks={}
    status='FAIL_IMPLEMENTATION'
    try:
        _check(type(certificate) is dict,'schema/dictionary',checks)
        _record_schema(certificate, 'schema_version model_id u v status guard critical_roots old_criticals critical_heights height_groups levels incidence events actual_barcode population_projection checks work', 'schema/selection', checks)
        _check(type(certificate['status']) is str and type(certificate['checks']) is dict
               and all(type(certificate[k]) is list for k in ('old_criticals','critical_heights','height_groups','levels','incidence','events')),
               'schema/record_types', checks)
        _work_schema(certificate['work'], 'schema/work', checks)
        _record_schema(certificate['actual_barcode'], 'status essential finite tie_policy', 'schema/barcode', checks)
        _check(type(certificate['actual_barcode']['status']) is str, 'schema/barcode_status_type', checks)
        _check(type(certificate['schema_version']) is int and certificate['schema_version']==1
               and certificate['model_id']=='explicit_periodic_cusp_v1'
               and _same((certificate['u'],certificate['v']),(u,v)),'schema/identity',checks)
        budget.checkpoint('verify/guard')
        _check(_same(certificate['guard'],guard_certificate()),'guard/source_bound_rational_replay',checks)
        _check(_same(certificate['old_criticals'],_old_criticals()),'guard/old_criticals',checks)
        roots=_verify_roots(certificate['critical_roots'],(-v,-u,Q(0),Q(1)),(-A_STAR,A_STAR),budget,checks,'verify/critical')
        heights=[_height(root,u,v,budget,'verify/height'+str(root['ordinal'])) for root in roots]
        _check(len(certificate['critical_heights'])==len(heights),'heights/count',checks)
        groups=certificate['height_groups']
        _check(all(type(g['id']) is str for g in groups) and len({g['id'] for g in groups})==len(groups) and bool(groups),'groups/unique',checks)
        seen=[]
        for i,group in enumerate(groups):
            ids=group['root_ordinals']; interval=group['centered_interval']
            _interval(interval)
            _check(type(ids) is list and ids==sorted(ids) and bool(ids) and len(set(ids))==len(ids),'groups/ordinals'+str(i),checks)
            for ordinal in ids:
                _check(type(ordinal) is int and 0<=ordinal<len(roots),'groups/root'+str(i),checks)
                _check(heights[ordinal]['centered_interval']==interval,'groups/height'+str(ordinal),checks)
                heights[ordinal]['group_id']=group['id']
            seen.extend(ids)
            if i:
                _check(groups[i-1]['centered_interval'][0]>interval[1],'groups/order'+str(i),checks)
            if len(ids)>1:
                witness=group['equality_witness']
                _check(v==0 and u>0 and type(witness) is dict,'groups/equality_present'+str(i),checks)
                factor=(-u,Q(0),Q(1)); polynomial=(Q(0),v,u/2,Q(0),Q(-1,4))
                _,remainder=_divide(polynomial,factor,budget,'verify/equality/reduction')
                _check(_same(witness,dict(factor=factor,height_remainder=remainder,root_ordinals=ids))
                       and remainder==(u*u/4,) and interval==(u*u/4,u*u/4),'groups/equality_identity'+str(i),checks)
                for ordinal in ids:
                    root=roots[ordinal]
                    if root['exact_value'] is not None:
                        proved=_eval(factor,root['exact_value'],budget,'verify/equality/exact')==0
                    else:
                        common=_gcd(root['factor'],factor,budget,'verify/equality/gcd')
                        seq=_sturm(common,budget,'verify/equality/sturm')
                        proved=_variation(seq,root['isolator'][0],budget,'verify/equality/lo')-_variation(seq,root['isolator'][1],budget,'verify/equality/hi')==1
                    _check(proved,'groups/equality_root'+str(ordinal),checks)
            else:
                _check(group['equality_witness'] is None,'groups/singleton'+str(i),checks)
        _check(sorted(seen)==list(range(len(roots))),'groups/partition',checks)
        _check(_same(heights,certificate['critical_heights']),'heights/values_and_inertia',checks)
        _check(DELTA/2>groups[0]['centered_interval'][1] and groups[-1]['centered_interval'][0]>-DELTA/2,
               'groups/barrier_band',checks)
        levels=certificate['levels']; regular=_regular_heights(groups)
        _check(len(levels)==len(regular),'levels/count',checks)
        for i,(level,h) in enumerate(zip(levels,regular)):
            _check(level['id']=='l'+str(i) and _same(level['centered_height'],h),'levels/height'+str(i),checks)
            q=(-h,v,u/2,Q(0),Q(-1,4))
            rs=level['root_certificate']
            _verify_roots(rs,q,(-R1/2,R1/2),budget,checks,'verify/level'+str(i))
            _check(all(r['multiplicity']==1 for r in rs['roots']),'levels/regular_roots'+str(i),checks)
            expected, signs=[],[]
            for index,lo,hi in _component_intervals(rs):
                witness=(lo+hi)/2
                value=_eval(q,witness,budget,'verify/level'+str(i)+'/sign'+str(index))
                _check(value!=0,'levels/regular_sign'+str(i)+'/'+str(index),checks)
                signs.append(dict(boundary_root_ordinals=[index-1 if index else None,index if index<len(rs['roots']) else None],witness=witness,value=value))
                if value>0:
                    _check(0<index<len(rs['roots']),'levels/boundary_barrier'+str(i),checks)
                    expected.append(dict(id=level['id']+'c'+str(len(expected)),boundary_root_ordinals=[index-1,index],
                        interior_witness=witness,witness_polynomial_value=value))
            _check(_same(level['components'],expected) and _same(level['sign_intervals'],signs),'levels/components'+str(i),checks)
        _check(not levels[0]['components'] and len(levels[-1]['components'])==1,'levels/separators',checks)
        finite=_verify_graph(certificate,roots,heights,groups,budget,checks)
        _check(_same(certificate['population_projection'],_projection(u,v,finite)),'population/projection',checks)
        budget.checkpoint('verify/final_acceptance')
        status='PASS'
    except Inconclusive as exc:
        status=exc.status
        checks[exc.phase]=False
    except (_InvalidCertificate,KeyError,TypeError,ValueError,IndexError,ZeroDivisionError) as exc:
        checks['rejected/'+str(exc)]=False
    return dict(status=status,checks=checks,work=_work(budget))
