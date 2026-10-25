# Baseline-bound link-migration guard (v2, Claude 01Cz). Replaces model.py line 58 in the
# successor proposal. Bound to the exact original edge set of canonical_dataset.json at
# main 11f7b82e (108 edges) and to exactly one authorized added edge.
import json, hashlib
ORIGINAL_EDGE_COUNT = 108
ORIGINAL_EDGE_SHA256 = 'e1c88cde4b04493170e74c58128f10a85c7c7961bf1daef6ef0f228d2fecd959'
AUTHORIZED_MIGRATION = {'id': 'MIG-P15-T1-PREMISE-T2-20261025', 'obligation': 'OBL:P15-T1',
                        'added_premise': 'P15-T2', 'link_count_before': 108, 'link_count_after': 109}
ALLOWED_MIGRATION_STATUS = {'PROPOSED_NOT_ADOPTED', 'ADOPTED_AFTER_REVIEW'}
SUCCESSOR_EDGE_SHA256 = 'bdd2ea04a24ced3fad835ca09a7bb3ed9b8608fbf02b044098f80e5c68f87523'

def _edge_digest(obs):
    edges = sorted([n, p] for n, o in obs.items() for p in o['premises'])
    return len(edges), hashlib.sha256(json.dumps(edges, ensure_ascii=False, sort_keys=True,
                                                 separators=(',', ':')).encode('utf-8')).hexdigest()

def check_link_migration(data, obs, err):
    migs = data.get('metadata', {}).get('documented_link_migrations', [])
    count, digest = _edge_digest(obs)
    if not migs:
        # unchanged original rule for an unmigrated dataset
        if count != ORIGINAL_EDGE_COUNT: raise err('proof link count changed without documented migration')
        return
    if len(migs) != 1: raise err('only the single authorized link migration is permitted')
    m = migs[0]
    if any(m.get(k) != v for k, v in AUTHORIZED_MIGRATION.items()): raise err('unauthorized link migration')
    if m.get('status') not in ALLOWED_MIGRATION_STATUS: raise err('link migration status not allowed')
    if count != AUTHORIZED_MIGRATION['link_count_after'] or digest != SUCCESSOR_EDGE_SHA256:
        raise err('edge set is not exactly the original 108 edges plus the authorized edge')
