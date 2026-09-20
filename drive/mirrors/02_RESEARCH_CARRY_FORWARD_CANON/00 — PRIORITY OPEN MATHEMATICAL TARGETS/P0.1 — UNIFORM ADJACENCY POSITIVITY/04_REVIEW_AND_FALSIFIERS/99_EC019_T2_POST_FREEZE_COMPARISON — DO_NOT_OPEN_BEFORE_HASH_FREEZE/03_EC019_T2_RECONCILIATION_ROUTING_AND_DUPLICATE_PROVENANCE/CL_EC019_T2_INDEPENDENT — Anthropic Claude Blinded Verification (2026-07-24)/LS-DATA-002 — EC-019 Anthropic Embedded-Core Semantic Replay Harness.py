#!/usr/bin/env python3
"""Lead-scientist semantic replay of the human-readable CL-DATA-200 implementation.

This does NOT claim byte-level replay of the declared Anthropic raw implementation,
runner, or frozen JSON, because those raw carriers are absent from the Drive folder.
It tests the mathematical source listing embedded in CL-DATA-200-v1.0.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from fractions import Fraction as Fr
from pathlib import Path
from typing import Any

IMPL_PATH = Path('/mnt/data/cl_ec019_t2_impl_rendered.py')
OUT_PATH = Path('/mnt/data/ls_ec019_cl_semantic_replay_result.json')


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location('cl_impl_rendered', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot construct module spec')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frac_obj(x: Fr) -> dict[str, Any]:
    return {
        'numerator': str(x.numerator),
        'denominator': str(x.denominator),
        'decimal_18': f'{float(x):.18e}',
    }


def iv_obj(x) -> dict[str, Any]:
    return {'lo': frac_obj(x.lo), 'hi': frac_obj(x.hi)}


def mutate(base: dict, key: str, center: Fr, radius: Fr) -> dict:
    out = copy.deepcopy(base)
    out[key] = (Fr(center), Fr(radius))
    return out


def margin_record(margins: dict) -> dict:
    out = {}
    for name, value in margins.items():
        if isinstance(value, Fr):
            out[name] = frac_obj(value)
        else:
            out[name] = {'non_fraction': repr(value)}
    return out


def result_at(module, S: int) -> dict[str, Any]:
    primary = module.compute(copy.deepcopy(module.BASE), S=S)
    margins = primary['margins']

    zero_radius = {}
    for name in module.PARAM_ORDER:
        spec = copy.deepcopy(module.BASE)
        center, _ = spec[name]
        spec[name] = (center, Fr(0))
        try:
            module.compute(spec, S=S)
            zero_radius[name] = {'rejected': False, 'error': None}
        except module.FullDimError as exc:
            zero_radius[name] = {'rejected': True, 'error': str(exc)}

    nc1 = zero_radius['c04']['rejected']
    nc2_m = module.compute(mutate(module.BASE, 'c40', Fr(0), Fr(500)), S=S)['margins']
    nc3_m = module.compute(mutate(module.BASE, 's', Fr(-1), Fr(11, 10)), S=S)['margins']
    nc4_m = module.compute(copy.deepcopy(module.BASE), kappa=Fr(1, 2000), S=S)['margins']
    nc5_m = module.compute(mutate(module.BASE, 'a', Fr(25, 10000), Fr(1, 2)), S=S)['margins']

    controls = {
        'NC1_zero_width': {
            'fires': nc1,
            'all_nine_zero_radius_rejected': all(v['rejected'] for v in zero_radius.values()),
        },
        'NC2_endpoint_typing': {
            'fires': nc2_m['m_saddle_det'] <= 0 or nc2_m['m_max_det'] <= 0,
            'm_saddle_det': frac_obj(nc2_m['m_saddle_det']),
            'm_max_det': frac_obj(nc2_m['m_max_det']),
        },
        'NC3_threshold': {
            'fires': nc3_m['m_threshold'] <= 0,
            'm_threshold': frac_obj(nc3_m['m_threshold']),
        },
        'NC4_cone': {
            'fires': nc4_m['m_cone_slope'] <= 0,
            'm_cone_slope': frac_obj(nc4_m['m_cone_slope']),
        },
        'NC5_capture_chart': {
            'fires': nc5_m['m_chart_X'] <= 0 or nc5_m['m_gersh_1'] <= 0,
            'm_chart_X': frac_obj(nc5_m['m_chart_X']),
            'm_gersh_1': frac_obj(nc5_m['m_gersh_1']),
        },
    }

    # Backend self-tests possible from the embedded mathematical core.
    x = Fr(7, 3)
    lo = module.sqrt_lo(x, S)
    hi = module.sqrt_hi(x, S)
    sqrt_containment = lo * lo <= x <= hi * hi
    even_sq = module.IV(Fr(-2), Fr(3)).powi(2)
    even_power_tight = even_sq.lo == 0 and even_sq.hi == 9

    # Sampled containment test for interval multiplication.
    samples = [Fr(i, 5) for i in range(-5, 6)]
    A = module.IV(Fr(-1), Fr(1))
    B = module.IV(Fr(-2), Fr(2))
    P = A * B
    mul_containment = all(P.lo <= a * b <= P.hi for a in samples for b in samples if -2 <= b <= 2)

    min_name = min(module.MARGIN_ORDER, key=lambda n: margins[n])
    return {
        'S': str(S),
        'primary_all_19_positive': module.all_positive(margins),
        'minimum_margin': {'name': min_name, **frac_obj(margins[min_name])},
        'margins': margin_record(margins),
        'derived': {
            name: iv_obj(value) if hasattr(value, 'lo') and hasattr(value, 'hi') else frac_obj(value)
            if isinstance(value, Fr) else str(value)
            for name, value in primary['derived'].items()
        },
        'controls': controls,
        'all_5_controls_fire': all(v['fires'] for v in controls.values()),
        'zero_radius_mutations': zero_radius,
        'all_9_zero_radius_rejected': all(v['rejected'] for v in zero_radius.values()),
        'self_tests': {
            'sqrt_7_over_3_containment': sqrt_containment,
            'even_power_tight': even_power_tight,
            'sampled_multiplication_containment': mul_containment,
        },
        'all_self_tests_pass': sqrt_containment and even_power_tight and mul_containment,
    }


def main() -> None:
    module = load_module(IMPL_PATH)
    rendered_bytes = IMPL_PATH.read_bytes()

    ladders = {
        'authoritative_reconciled_prereg': [10**40, 10**100, 10**200],
        'reported_duplicate_prereg_and_result_capsule': [10**40, 10**80, 10**120],
    }
    runs = {name: [result_at(module, S) for S in ladder] for name, ladder in ladders.items()}

    def sign_vector(run):
        return tuple(run['margins'][name]['numerator'].startswith('-') is False and run['margins'][name]['numerator'] != '0'
                     for name in module.MARGIN_ORDER)

    sign_stability = {}
    for name, values in runs.items():
        vectors = [sign_vector(v) for v in values]
        sign_stability[name] = all(v == vectors[0] for v in vectors[1:])

    output = {
        'object': 'Semantic replay of the human-readable mathematical core embedded in CL-DATA-200-v1.0',
        'epistemic_limits': {
            'byte_exact_replay_of_declared_raw_impl': False,
            'byte_exact_replay_of_declared_raw_runner': False,
            'byte_exact_replay_of_declared_raw_result': False,
            'reason': 'Raw impl/runner/result carriers were absent from the inspected Drive receipt folder; the embedded plain source is explicitly a human-readable rendering and has different bytes/hash from the declared authoritative impl.',
        },
        'rendered_impl': {
            'path': str(IMPL_PATH),
            'bytes': len(rendered_bytes),
            'sha256': hashlib.sha256(rendered_bytes).hexdigest(),
            'declared_authoritative_impl_bytes': 12832,
            'declared_authoritative_impl_sha256': '6e6e7a532e8e66bb92f80fdc2a225f832c77987e01bdcd63d79a1fb8e99ac9d4',
            'matches_declared_authoritative_bytes_and_hash': False,
        },
        'declared_missing_carriers': {
            'runner': {'bytes': 8737, 'sha256': '5f5a8022279c1ea08914b985e313637acf2c4b87b981a4a66b4f1a10468fe766'},
            'result': {'bytes': 45971, 'sha256': '016c23016b9307dca9f47316cf8322a8bce8eeb0face3a577e5c3fee8301dbeb'},
        },
        'precision_policy_conflict': {
            'authoritative_reconciled_prereg_exponents': [40, 100, 200],
            'reported_result_exponents': [40, 80, 120],
            'exact_match': False,
        },
        'runs': runs,
        'sign_stability': sign_stability,
        'all_runs_primary_positive': all(v['primary_all_19_positive'] for values in runs.values() for v in values),
        'all_runs_controls_fire': all(v['all_5_controls_fire'] for values in runs.values() for v in values),
        'all_runs_zero_radius_reject': all(v['all_9_zero_radius_rejected'] for values in runs.values() for v in values),
        'all_runs_self_tests_pass': all(v['all_self_tests_pass'] for values in runs.values() for v in values),
    }

    OUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'output_path': str(OUT_PATH),
        'output_bytes': OUT_PATH.stat().st_size,
        'output_sha256': hashlib.sha256(OUT_PATH.read_bytes()).hexdigest(),
        'rendered_impl_bytes': len(rendered_bytes),
        'rendered_impl_sha256': hashlib.sha256(rendered_bytes).hexdigest(),
        'all_runs_primary_positive': output['all_runs_primary_positive'],
        'all_runs_controls_fire': output['all_runs_controls_fire'],
        'all_runs_zero_radius_reject': output['all_runs_zero_radius_reject'],
        'all_runs_self_tests_pass': output['all_runs_self_tests_pass'],
        'sign_stability': sign_stability,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
