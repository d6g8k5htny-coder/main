"""Controlled graph branches and proposed upstream pin changes.

Branches never approve or install mathematics. Staging calls only the existing
coordinator's proposal API. Engine locks always precede this service's journal.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engine'))
from conductor import Journal, find_item, validate_edit, impact_from, mathematical_fingerprints
from model import canonical, digest

KINDS = frozenset({'statement', 'premise', 'side_condition', 'import', 'correction'})
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}')
IDENTITY = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}')
SHA = re.compile(r'[0-9a-f]{64}')
MAX_EDITS = 16
MAX_BRANCHES = 256
MAX_PATCH_BYTES = 64 * 1024


def _actor(value):
    if not isinstance(value, str) or not 1 <= len(value.encode('utf8')) <= 120 or not value.strip():
        raise ValueError('invalid proposal actor')
    return value


def _sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError('invalid SHA-256 pin')
    return value


def _value(data, kind, target):
    item = find_item(data, kind, target)
    field = {'statement': 'statement', 'premise': 'premises',
             'import': 'scope_description', 'correction': 'role'}.get(kind)
    return deepcopy(item[field] if field else item['side_conditions'][0]['text'])


def _targets(data, kind, target):
    if kind in {'statement', 'premise', 'side_condition'}: return [target]
    field = 'external_theorem_imports' if kind == 'import' else 'source_corrections'
    return sorted(o['conclusion'] for o in data['obligations'] if target in o[field])


def _edits(data, edits):
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        raise ValueError('branch requires 1 to 16 bounded edits')
    try: size = len(canonical(edits))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('edits must be bounded JSON') from exc
    if size > MAX_PATCH_BYTES: raise ValueError('patch byte budget exceeded')
    normalized, keys, candidate = [], set(), deepcopy(data)
    for edit in edits:
        if (not isinstance(edit, dict) or set(edit) != {'kind', 'target', 'replacement'}
                or not isinstance(edit['kind'], str) or edit['kind'] not in KINDS):
            raise ValueError('unsupported branch edit fields or mathematical effect')
        kind, target, replacement = edit['kind'], edit['target'], edit['replacement']
        if not isinstance(target, str) or not 1 <= len(target) <= 128:
            raise ValueError('invalid edit target')
        key = (kind, target)
        if key in keys: raise ValueError('overlapping duplicate edits in branch')
        keys.add(key)
        if kind == 'premise' and not isinstance(replacement, str):
            replacement = json.dumps(replacement, sort_keys=True)
        if not isinstance(replacement, str) or not 1 <= len(replacement.encode('utf8')) <= 20000:
            raise ValueError('replacement text byte budget exceeded')
        candidate = validate_edit(candidate, kind, target, replacement)
        normalized.append({'kind': kind, 'target': target, 'replacement': replacement})
    return normalized, candidate


class EvolutionLab:
    """Durable experimental patches with three-way obligation-context checking."""
    def __init__(self, workdir, coordinator):
        self.coordinator = coordinator
        self.journal = Journal(workdir)
        self.snapshot()

    def _replay(self):
        self.journal._load()
        branches, sources = {}, []
        for event in self.journal.events:
            payload = deepcopy(event['payload'])
            if event['type'] == 'EVOLUTION_BRANCH_CREATED':
                root = payload.pop('branch_root_sha256', None)
                if (payload.get('schema') != 'UL-evolution-branch-1' or root != digest(payload)
                        or payload.get('branch_id') in branches):
                    raise ValueError('invalid evolution branch root or duplicate identity')
                payload['branch_root_sha256'] = root
                payload['status'] = 'experimental_not_staged'
                branches[payload['branch_id']] = payload
            elif event['type'] == 'EVOLUTION_BRANCH_STAGED':
                branch = branches.get(payload.get('branch_id'))
                if branch is None or branch['branch_root_sha256'] != payload.get('branch_root_sha256'):
                    raise ValueError('staged branch root mismatch')
                branch['staging'] = payload
                branch['status'] = 'awaiting_approval'
            elif event['type'] == 'EVOLUTION_SOURCE_PROPOSED':
                if payload.get('status') != 'proposed_not_installed':
                    raise ValueError('source updates cannot be installed by evolution')
                sources.append(payload)
            else: raise ValueError('unknown evolution journal event')
        if len(branches) > MAX_BRANCHES or len(sources) > MAX_BRANCHES:
            raise ValueError('evolution journal inventory budget exceeded')
        return branches, sources

    def _comparison(self, branch, data, state):
        current = mathematical_fingerprints(data, state['certificate_attachments'])
        conflicts = []
        for edit in branch['edits']:
            key = edit['kind'] + ':' + edit['target']
            if _value(data, edit['kind'], edit['target']) != branch['old_values'][key]:
                conflicts.append({'kind': 'overlapping_edit', 'target': edit['target'], 'field': edit['kind']})
        for target, expected in branch['target_fingerprints'].items():
            if current.get(target) != expected:
                conflicts.append({'kind': 'changed_transitive_context', 'target': target})
        if not conflicts:
            try: _edits(data, branch['edits'])
            except ValueError as exc: conflicts.append({'kind': 'current_validation_failed', 'reason': str(exc)})
        return {'branch_id': branch['branch_id'], 'branch_root_sha256': branch['branch_root_sha256'],
                'base_branch_sha256': branch['base_branch_sha256'],
                'current_branch_sha256': state['branch_sha256'],
                'base_input_sha256': branch['base_input_sha256'],
                'current_input_sha256': state['effective_dataset_sha256'],
                'base_changed': branch['base_branch_sha256'] != state['branch_sha256'],
                'conflicts': conflicts, 'can_stage': not conflicts,
                'math_effect': 'proposal_only_not_mathematical_acceptance'}

    def create_branch(self, name, edits, base_hash=None, actor='local-owner'):
        if not isinstance(name, str) or not NAME.fullmatch(name): raise ValueError('invalid branch name')
        _actor(actor)
        with self.coordinator._mutex, self.coordinator.journal.transaction(), self.journal.transaction():
            data, state = self.coordinator._replay()
            if base_hash is not None and _sha(base_hash) != state['branch_sha256']:
                raise ValueError('stale branch base hash')
            branches, _ = self._replay()
            if len(branches) >= MAX_BRANCHES: raise ValueError('branch inventory budget exhausted')
            if any(branch['name'] == name for branch in branches.values()):
                raise ValueError('branch name already exists')
            edits, candidate = _edits(data, edits)
            targets = {rid for edit in edits for rid in _targets(data, edit['kind'], edit['target'])}
            fps = mathematical_fingerprints(data, state['certificate_attachments'])
            branch = {'schema': 'UL-evolution-branch-1', 'branch_id': 'branch-' + uuid.uuid4().hex[:16],
                      'name': name, 'actor': actor, 'base_branch_sha256': state['branch_sha256'],
                      'base_input_sha256': state['effective_dataset_sha256'],
                      'candidate_input_sha256': digest(candidate), 'edits': edits,
                      'old_values': {edit['kind'] + ':' + edit['target']:
                                     _value(data, edit['kind'], edit['target']) for edit in edits},
                      'target_fingerprints': {rid: fps[rid] for rid in sorted(targets)},
                      'preview_affected': impact_from(data, candidate)['affected'],
                      'effect': 'stage_proposals_only'}
            branch['branch_root_sha256'] = digest(branch)
            self.journal.append('EVOLUTION_BRANCH_CREATED', branch)
            return deepcopy(branch)

    def compare(self, branch_id):
        with self.coordinator._mutex, self.coordinator.journal.transaction(), self.journal.transaction():
            data, state = self.coordinator._replay()
            branches, _ = self._replay()
            if not isinstance(branch_id, str) or branch_id not in branches: raise ValueError('unknown branch')
            return self._comparison(branches[branch_id], data, state)

    def stage(self, branch_id, actor='local-owner'):
        _actor(actor)
        with self.coordinator._mutex, self.coordinator.journal.transaction(), self.journal.transaction():
            data, state = self.coordinator._replay()
            branches, _ = self._replay()
            if not isinstance(branch_id, str) or branch_id not in branches: raise ValueError('unknown branch')
            branch = branches[branch_id]
            comparison = self._comparison(branch, data, state)
            if not comparison['can_stage']: raise ValueError('branch conflict: stale or changed obligation context')
            previous = branch.get('staging')
            if previous and previous['staged_against_branch_sha256'] == state['branch_sha256']:
                return deepcopy(previous)
            # Validate the whole sequence first. Each API call creates an independently
            # pending proposal; there is deliberately no atomic approval API here.
            _edits(data, branch['edits'])
            proposals = []
            for index, edit in enumerate(branch['edits']):
                event_id = 'evolution:' + branch_id + ':' + state['branch_sha256'] + ':' + str(index)
                proposal = self.coordinator.propose(**edit, actor=actor, event_id=event_id)
                if proposal['base_branch_sha256'] != state['branch_sha256']:
                    raise ValueError('stale interrupted branch staging')
                proposals.append(proposal['proposal_id'])
            result = {'branch_id': branch_id, 'branch_root_sha256': branch['branch_root_sha256'],
                      'proposal_ids': proposals, 'actor': actor, 'status': 'awaiting_approval',
                      'staged_against_branch_sha256': state['branch_sha256'],
                      'original_base_branch_sha256': branch['base_branch_sha256'],
                      'math_effect': 'proposals_created_only'}
            self.journal.append('EVOLUTION_BRANCH_STAGED', result,
                                event_id='stage:' + branch_id + ':' + state['branch_sha256'])
            return deepcopy(result)

    @staticmethod
    def _source_pins():
        """Read fixed bundled registries only; never follow user-supplied paths/URLs."""
        pins = {}
        registries = [('literature/catalog.json', 'sources'),
                      ('research_catalog.json', 'documents'),
                      ('frontier_sources/PRIMARY_SOURCES.json', 'primary_sources')]
        for relative, group in registries:
            path = ROOT / relative
            if not path.exists(): continue
            if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError('unsafe bundled source pin registry')
            registry = json.loads(path.read_text(encoding='utf8'))
            for item in registry.get(group, []):
                identity = item.get('id')
                pin = item.get('source_sha256') or item.get('sha256')
                if identity and isinstance(pin, str) and SHA.fullmatch(pin):
                    if identity in pins and pins[identity] != pin:
                        raise ValueError('ambiguous bundled source pin identity')
                    pins[identity] = pin
        return pins

    def propose_source_update(self, source_id, old_sha256, new_sha256, url, actor='local-owner'):
        _actor(actor); _sha(old_sha256); _sha(new_sha256)
        if not isinstance(source_id, str) or not IDENTITY.fullmatch(source_id):
            raise ValueError('invalid source identity')
        if old_sha256 == new_sha256: raise ValueError('unchanged source pin')
        if not isinstance(url, str) or len(url.encode('utf8')) > 2048:
            raise ValueError('invalid source URL')
        try:
            parsed = urlsplit(url)
            safe = (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                    and not parsed.password and parsed.port in (None, 443) and not parsed.fragment
                    and not any(ord(c) < 33 for c in url))
        except ValueError: safe = False
        if not safe: raise ValueError('source proposal requires a bounded HTTPS URL')
        with self.coordinator._mutex, self.coordinator.journal.transaction(), self.journal.transaction():
            _, state = self.coordinator._replay()
            _, sources = self._replay()
            if len(sources) >= MAX_BRANCHES: raise ValueError('source proposal budget exhausted')
            if self._source_pins().get(source_id) != old_sha256:
                raise ValueError('old source pin does not match the bundled registry')
            proposal = {'schema': 'UL-source-pin-proposal-1', 'proposal_id': 'source-' + uuid.uuid4().hex[:16],
                        'source_id': source_id, 'old_sha256': old_sha256, 'new_sha256': new_sha256,
                        'url': url, 'actor': actor, 'base_branch_sha256': state['branch_sha256'],
                        'base_input_sha256': state['effective_dataset_sha256'],
                        'status': 'proposed_not_installed', 'source_bytes_fetched': False,
                        'mathematical_applicability': 'not_evaluated'}
            self.journal.append('EVOLUTION_SOURCE_PROPOSED', proposal)
            return deepcopy(proposal)

    def snapshot(self):
        with self.coordinator._mutex, self.coordinator.journal.transaction(), self.journal.transaction():
            data, state = self.coordinator._replay()
            branches, sources = self._replay()
            return {'schema': 'UL-evolution-state-1', 'branches': [
                        {**branch, 'comparison': self._comparison(branch, data, state)}
                        for _, branch in sorted(branches.items())],
                    'source_proposals': sources, 'source_updates_installed': 0,
                    'mathematical_approvals_performed': 0, 'remote_multiuser': 'unconfigured',
                    'journal_head_sha256': self.journal.events[-1]['event_hash'] if self.journal.events else '0' * 64}
