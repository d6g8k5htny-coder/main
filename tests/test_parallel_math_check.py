"""Cross-host H3 replay must preserve every mathematical and custody field."""
import copy
import hashlib
import pytest
from tools import parallel_math_check as C


@pytest.fixture
def candidate():
    return C.load_report(C.ROOT/'research/parallel/h3/candidate.json')


def rehash(data):
    data['payload_sha256'] = hashlib.sha256(C.h3_payload_bytes(data['payload'])).hexdigest()
    return data


def test_only_host_and_computational_audit_fields_are_normalized(candidate):
    changed = copy.deepcopy(candidate)
    changed['payload'].update(repository_root='/portable/repo', computational_sources={'new-port.py':'0'*64}, checker_sha256='1'*64)
    C.compare_h3(rehash(changed), candidate)


@pytest.mark.parametrize('field,value', [
    ('imported_floor', '1/100'), ('new_conservative_floor','1'),
    ('fixed_r_floor_reproved_by_this_candidate',1), ('r','1/10'),
    ('original_box_proves_imported_floor',False), ('all_small_r_certified',True),
    ('source',{}), ('negative_controls',[]), ('checks',[]),
])
def test_even_rehashed_mathematical_mutations_fail(candidate, field, value):
    changed = copy.deepcopy(candidate)
    changed['payload'][field] = value
    with pytest.raises(ValueError, match='mathematical reconstruction'):
        C.compare_h3(rehash(changed), candidate)


def test_wrong_envelope_digest_fails(candidate):
    candidate['payload_sha256'] = '0'*64
    with pytest.raises(ValueError, match='digest'):
        C.h3_mathematics(candidate)


def test_missing_audit_fields_fail(candidate):
    del candidate['payload']['checker_sha256']
    with pytest.raises(ValueError, match='audit identities'):
        C.h3_mathematics(rehash(candidate))
