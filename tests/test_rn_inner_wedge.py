"""Composition controls; synthetic premises here test accounting, not the field."""
import copy
from fractions import Fraction as F
import json
import os
import subprocess
import sys

import pytest

from research.cover.ledger import Box, PartitionError
from tools import rn_inner_wedge_check as check


@pytest.fixture
def synthetic_inputs():
    outer = check.OUTER_BOX
    width = outer.du()/12
    pieces = [dict(box=Box(outer.u0+i*width, outer.u0+(i+1)*width, outer.v0, outer.v1),
                   determinant_lower=check.DETERMINANT_LOWER, q_lower=check.MAHALANOBIS_LOWER)
              for i in range(12)]
    bounds = dict(pieces=pieces, geometry=check.containment(), piece_count=12,
                  pending_pieces=0, conditioning_energy_upper=F(7))
    majorant = dict(determinant_lower=check.DETERMINANT_LOWER, mahalanobis_lower=check.MAHALANOBIS_LOWER,
                    jacobian_upper=F(1), integrand_upper=check.INTEGRAND_UPPER/2)
    return bounds, majorant


def test_exact_wedge_area_once_and_zero_lower(synthetic_inputs):
    bounds, majorant = synthetic_inputs
    receipt = check.assemble_cover(bounds, majorant)
    assert receipt['pending_count'] == 0
    total = receipt['total']
    assert total['certified'] and total['covers_region']
    assert F(total['enclosure_lo']) == 0
    assert F(total['enclosure_hi']) == F(total['area_accounted_hi'])*majorant['integrand_upper']
    assert F(total['enclosure_hi']) < check.INTEGRAL_UPPER/2
    # Auxiliary area is much larger and is not the integration measure.
    assert check.OUTER_BOX.param_area() > F(total['area_accounted_hi'])


@pytest.mark.parametrize('key,value', [('determinant_lower', F(0)), ('q_lower', F(102))])
def test_weak_child_cannot_hide_behind_optimistic_aggregate(synthetic_inputs, key, value):
    bounds, majorant = synthetic_inputs
    bounds[key] = 10**100
    bounds['pieces'][7][key] = value
    with pytest.raises(ValueError, match='uniform determinant'):
        check.assemble_cover(bounds, majorant)


@pytest.mark.parametrize('fault', ['missing', 'gap', 'overlap', 'escape'])
def test_auxiliary_partition_defects_refused(synthetic_inputs, fault):
    bounds, majorant = synthetic_inputs
    if fault == 'missing':
        bounds['pieces'].pop()
    else:
        box = bounds['pieces'][4]['box']
        step = F(1, 10**6)
        bounds['pieces'][4]['box'] = Box(box.u0+(step if fault == 'gap' else -step), box.u1,
                                        box.v0-(step if fault == 'escape' else 0), box.v1)
    with pytest.raises((ValueError, PartitionError)):
        check.assemble_cover(bounds, majorant)


@pytest.mark.parametrize('count', [1, False])
def test_pending_or_boolean_counts_refused(synthetic_inputs, count):
    bounds, majorant = synthetic_inputs
    bounds['pending_pieces'] = count
    with pytest.raises(ValueError, match='complete twelve-piece'):
        check.assemble_cover(bounds, majorant)


@pytest.mark.parametrize('key,value', [('jacobian_upper', F(2)), ('integrand_upper', F(0)),
                                      ('integrand_upper', check.INTEGRAND_UPPER),
                                      ('determinant_lower', 1), ('mahalanobis_lower', 104)])
def test_density_coordinate_or_cap_mismatch_refused(synthetic_inputs, key, value):
    bounds, majorant = synthetic_inputs
    majorant[key] = value
    with pytest.raises(ValueError, match='density-weighted moment'):
        check.assemble_cover(bounds, majorant)


def test_wrong_area_and_changed_geometry_refused(synthetic_inputs, monkeypatch):
    bounds, majorant = synthetic_inputs
    wrong = copy.deepcopy(bounds)
    wrong['geometry']['area_pi_coefficient'] /= 2
    with pytest.raises(ValueError, match='geometry differs'):
        check.assemble_cover(wrong, majorant)
    monkeypatch.setattr(check, 'AREA_PI_COEFFICIENT', check.AREA_PI_COEFFICIENT/2)
    with pytest.raises(ValueError, match='polar area'):
        check.assemble_cover(bounds, majorant)


def test_energy_ceiling_is_strict(synthetic_inputs):
    bounds, majorant = synthetic_inputs
    bounds['conditioning_energy_upper'] = F(8)
    with pytest.raises(ValueError, match='uniform determinant'):
        check.assemble_cover(bounds, majorant)


@pytest.mark.parametrize('optimized', [False, True])
def test_cli_rejects_annulus_scope_promotion(tmp_path, optimized):
    invalid = dict(schema=check.SCHEMA, scope={**check.SCOPE, 'full_annulus_covered': True})
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(invalid))
    command = [sys.executable]+(['-O'] if optimized else [])+[str(check.ROOT/'tools/rn_inner_wedge_check.py'), '--check', str(path)]
    result = subprocess.run(command, cwd=check.ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1 and 'scientific scope' in result.stdout
