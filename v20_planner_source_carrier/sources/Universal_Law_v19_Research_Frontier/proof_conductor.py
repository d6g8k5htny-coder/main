"""Durable, bounded planning over a validated semantic proof projection.

Claims reserve global effort, not mathematical authority. Failure reports select
registered investigation strategies and cannot certify or refute statements.
"""
from __future__ import annotations

import copy
import math
import os
import stat
import hashlib
import json
import re
from pathlib import Path
import time
import uuid
from contextlib import ExitStack, contextmanager

from frontier_ir import compile_ir, validate_ir, digest, bounded_json, EFFECTS
from conductor import Journal, mathematical_fingerprints
from model import load, fingerprints
from intelligence import MILESTONES

VERSION = 'UL-proof-conductor-1'
PILOT_EFFECT = 'checker.fixed_scalar_replay'
MAX_ATTEMPTS = 4
MAX_EVENTS = 10000
MAX_JOURNAL_BYTES = 32 * 1024 * 1024
STRATEGIES = {
    'native_scalar_profile': 'Run only the fixed reviewed two-goal scalar worker and independent native replay.',
    'dependency_decomposition': 'Inspect every joint premise and explicit side condition.',
    'scope_guidance': 'Restate the exact question and identify missing contextual guidance.',
    'counterexample_audit': 'Investigate the alleged counterexample at the exact scope without accepting it.',
    'hyperedge_reconstruction': 'Reconstruct the complete AND obligation from validated source.',
    'source_custody_audit': 'Locate and authenticate required source bytes and exact versions.',
    'checker_environment_audit': 'Diagnose pinned checker applicability and environment without changing checkers.',
    'bounded_partition': 'Partition the investigation into bounded subquestions without dropping premises.',
    'exact_statement_comparison': 'Compare original statement, scope and inputs against the reported drift.',
    'independent_source_review': 'Compare preserved evidence using an independent source review.',
}
DIAGNOSES = {
    'insufficient_guidance': ('guide', 'scope_guidance'),
    'false_lemma': ('correct', 'counterexample_audit'),
    'invalid_decomposition': ('replan', 'hyperedge_reconstruction'),
    'missing_source': ('guide', 'source_custody_audit'),
    'checker_failure': ('replan', 'checker_environment_audit'),
    'timeout': ('replan', 'bounded_partition'),
    'statement_drift': ('correct', 'exact_statement_comparison'),
}
MILESTONE_MAP = {m['milestone_id']: copy.deepcopy(m) for m in MILESTONES}
INTERFACES = {
    'd5-second-factorial-upper': 'Full collision-stratum upper second factorial moment for the actual field under the original conditioned and endpoint-tilted law; separated tuples and a lower bound do not supply this missing interface.',
    'marked-kac-rice-elder-alignment': 'A single scope-aligned conditional kernel, continuous-mark theorem, global Borel elder event and integrable diagonal exhaustion chain.',
    'normalizer-floor-weighted-transfer': 'Uniform positive floor for the original endpoint-only denominator and its exact weighted-transfer applicability.',
}


def native_backend_identity(toolchain=None):
    """Cheap cache invalidation, never a substitute for pinned native replay.

    Include actual selection/resolution, every pinned file and all directory and
    shadow entries under native search roots. lstat never opens FIFO/device data
    or follows a directory symlink. A changed identity requires full inventory
    and independent replay before any cached positive result is reused.
    """
    from toolchain_setup import default_toolchain
    from formal_lane import _pins
    selected = toolchain if toolchain is not None else default_toolchain()
    selection = {'origin':'explicit' if toolchain is not None else
                 'environment' if 'UL_LEAN_TOOLCHAIN' in os.environ else 'bundled',
                 'value':None if selected is None or selected == '' else os.fspath(selected)}
    if selection['value'] is None:
        return {'signature':digest({'selection':selection}), 'present':False,
                'selected_path':None,'resolved_path':None,'metadata_entries':0}
    path = Path(selection['value']).expanduser().absolute()
    try:
        resolved = path.resolve()
    except (OSError,RuntimeError):
        resolved = path
    rows, errors = {}, []
    def metadata(item):
        try:
            info = item.lstat()
            row = [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,
                   info.st_ctime_ns,info.st_mode]
            if stat.S_ISLNK(info.st_mode):
                row.append(os.readlink(item))
            return row
        except OSError as exc:
            return ['unavailable',exc.errno]
    selected_row, root_row = metadata(path), metadata(resolved)
    truncated = False
    def visit(directory):
        nonlocal truncated
        relative = directory.relative_to(resolved).as_posix()
        info = metadata(directory)
        rows[relative] = info
        if not info or type(info[0]) is not int or not stat.S_ISDIR(info[5]):
            return
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if len(rows)>=20000:
                        truncated = True
                        return
                    child = directory/entry.name
                    key = child.relative_to(resolved).as_posix()
                    info = metadata(child)
                    rows[key] = info
                    if info and type(info[0]) is int and stat.S_ISDIR(info[5]):
                        visit(child)
                    if truncated:
                        return
        except OSError as exc:
            errors.append([relative,exc.errno])
    for name in ('bin','lib'):
        visit(resolved/name)
    try:
        pinned = [[name,rows.get(name,['missing'])] for name in sorted(_pins()['files'])]
    except (OSError,ValueError) as exc:
        pinned = [['invalid_pin_manifest',type(exc).__name__,str(exc)]]
    present = not truncated and all(
        name in rows and type(rows[name][0]) is int and stat.S_ISREG(rows[name][5])
        for name in ('bin/lean','bin/leanchecker'))
    payload = {'selection':selection,'selected_path':str(path),'selected':selected_row,
               'resolved_path':str(resolved),'root':root_row,'pinned':pinned,
               'search_closure':sorted(rows.items()),'errors':errors,'truncated':truncated}
    return {'signature':digest(payload),'present':present,'selected_path':str(path),
            'resolved_path':str(resolved),'metadata_entries':len(rows)}


def _text(value, name, maximum=128):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError('invalid ' + name)
    return value


def _integer(value, name, minimum=1, maximum=1000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('invalid ' + name)
    return value


def _number(value, name, minimum=0, maximum=3600):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum < value <= maximum:
        raise ValueError('invalid ' + name)
    return value


def _targets(value, ir):
    if type(value) is not list or not 1 <= len(value) <= 32:
        raise ValueError('targets must be a bounded nonempty list')
    for target in value:
        _text(target, 'target')
    if len(set(value)) != len(value):
        raise ValueError('duplicate target')
    valid = {n['id'] for n in ir['nodes']} | set(MILESTONE_MAP) | {'scalar-variance-bounds'}
    if set(value) - valid:
        raise ValueError('unknown target')
    return sorted(value)


def _build_blueprint(ir, target_ids, budget, record_fingerprints=None, pilot_contract=None, identity_version=2):
    """Expand required AND closure; missing atomic obligations are the frontier.

    With no certified records, there are no optional OR paths or proven atoms to
    prune. Deduplicate the full required closure, retain every side condition,
    and rank shared blockers by how many downstream record obligations use them.
    """
    targets = _targets(target_ids, ir)
    if type(identity_version) is not int or identity_version not in (1,2):
        raise ValueError('unsupported blueprint identity version')
    _integer(budget, 'budget')
    nodes = {n['id']: n for n in ir['nodes']}
    record_fingerprints = record_fingerprints or {rid:n['input_fingerprint'] for rid,n in nodes.items()}
    edges = {e['conclusion']: e for e in ir['hyperedges']}
    roots = set()
    for target in targets:
        if target == 'scalar-variance-bounds':
            continue
        roots.update(MILESTONE_MAP[target]['bound_record_ids'] if target in MILESTONE_MAP else [target])
    required = set()
    def include(rid):
        if rid in required:
            return
        required.add(rid)
        for parent in edges[rid]['premises']:
            include(parent)
    for rid in roots:
        include(rid)
    users = {rid: set() for rid in nodes}
    for rid in nodes:
        for parent in edges[rid]['premises']:
            users[parent].add(rid)
    def downstream(rid):
        seen, todo = {rid}, [rid]
        while todo:
            for child in users[todo.pop()]:
                if child not in seen:
                    seen.add(child)
                    todo.append(child)
        return seen
    impacts = {rid: downstream(rid) for rid in nodes}
    items, atom_users = {}, {}
    def add(kind, subject, question, scope, bound_records):
        binding = {'records': [[rid, record_fingerprints[rid]] for rid in sorted(bound_records)]}
        fingerprint = digest({'question': question, 'scope': scope, 'inputs': binding})
        key = 'ft-' + digest({'kind':kind,'subject':subject,'input':fingerprint})[:24]
        task = {'task_id': key, 'kind': kind, 'subject_id': subject,
                'intended_question': question, 'scope': copy.deepcopy(scope),
                'question_sha256': digest(question), 'scope_sha256': digest(scope),
                'input_fingerprint': fingerprint, 'origin_ir_sha256': ir['ir_sha256'],
                'bound_record_ids': sorted(bound_records), 'dependencies': [],
                'effects': list(EFFECTS), 'completed': False}
        if key in items and items[key] != task:
            raise ValueError('task identity collision')
        items[key] = task
        return key
    record_ids = {}
    for rid in sorted(required):
        n = nodes[rid]
        record_ids[rid] = add('record', rid, 'Audit the exact proof obligation for ' + rid + ': ' + n['statement'],
            {'kind':'canonical_record','record_id':rid,'text':n['scope']}, [rid])
    imports = {i['id']:i for i in ir['imports']}
    corrections = {c['id']:c for c in ir['corrections']}
    for rid in sorted(required):
        edge, n = edges[rid], nodes[rid]
        deps = [record_ids[p] for p in edge['premises']]
        atoms = []
        for condition in edge['side_conditions']:
            atoms.append(add('side_condition', condition['id'], 'Audit this undischarged necessary side condition: ' + condition['text'],
                {'kind':'side_condition','record_id':rid,'record_scope':n['scope'],'condition_id':condition['id']}, [rid]))
        # Imports/corrections share exact descriptor identity across target records.
        for iid in edge['imports']:
            descriptor = imports[iid]
            if descriptor['import_status'] != 'independently_certified':
                applicable = sorted(descriptor['applies_to'])
                atoms.append(add('import', iid, 'Authenticate and check applicability of imported theorem: ' + descriptor['statement_or_required_evidence'],
                    {'kind':'external_import','id':iid,'scope':descriptor['scope_description'], 'applies_to':descriptor['applies_to']}, applicable))
        for cid in edge['corrections']:
            descriptor = corrections[cid]
            if not descriptor['validated']:
                applicable = sorted(r for r in nodes if cid in edges[r]['corrections'])
                atoms.append(add('correction', cid, 'Authenticate and review mandatory source correction: ' + descriptor['role'],
                    {'kind':'source_correction','id':cid,'source':descriptor['source_url']}, applicable))
        atoms.append(add('source', rid, 'Authenticate exact proof source bytes for ' + rid + ': ' + n['source'],
            {'kind':'source_custody','record_id':rid,'source':n['source'],'record_scope':n['scope']}, [rid]))
        for atom in atoms:
            atom_users.setdefault(atom,set()).add(rid)
        items[record_ids[rid]]['dependencies'] = sorted(set(deps + atoms))
    for target in targets:
        if target not in MILESTONE_MAP:
            continue
        milestone = MILESTONE_MAP[target]
        records = milestone['bound_record_ids']
        interface = add('research_interface', target + ':missing-interface',
            'Investigate the missing proof interface: ' + INTERFACES[target] + ' Exact target question: ' + milestone['intended_question'],
            {'kind':'research_interface','milestone':target,'target_scope':milestone['scope']}, records)
        atom_users[interface] = set(records)
        task = add('milestone', target, milestone['intended_question'], milestone['scope'], records)
        items[task]['dependencies'] = sorted([interface] + [record_ids[rid] for rid in records])
    if 'scalar-variance-bounds' in targets:
        from polynomial_prover import profile_contract
        contract = profile_contract() if pilot_contract is None else bounded_json(pilot_contract,'historical scalar contract',65536)
        if type(contract) is not dict or contract.get('schema') != 'ul-v19-scalar-contract-1' or contract.get('actual_field_applicable') is not False or contract.get('full_theorem_certificates') != []:
            raise ValueError('invalid historical scalar contract')
        if type(contract.get('statements')) is not list or len(contract['statements']) != 2 or any(type(statement) is not dict or type(statement.get('formal_statement')) is not str for statement in contract['statements']):
            raise ValueError('invalid historical scalar statement contract')
        scope = {'kind':'scalar_pilot', 'formal_domain':'Nat',
                 'interpretation':'Scalar polynomial bounds for rational variance parameters after separate denominator conditions; actual field applicability remains open.',
                 'profile_scope':contract['scope'], 'profile_contract_sha256':digest(contract),
                 'actual_field_applicable':False}
        question = 'Prove exactly these two universal scalar inequalities: ' + '; '.join(statement['formal_statement'] for statement in contract['statements'])
        pilot = add('scalar_pilot','scalar-variance-bounds',question,scope,[])
        items[pilot]['profile_contract'] = contract
        items[pilot]['effects'] = list(EFFECTS) + [PILOT_EFFECT]
    for task in items.values():
        impacted = set()
        for rid in task['bound_record_ids']:
            impacted.update(impacts[rid])
        task['downstream_impact'] = len(impacted)
        task['priority'] = 100 * len(impacted) + {'research_interface':80,'import':60,'correction':50,'source':40,'side_condition':30,'record':0,'milestone':0,'scalar_pilot':70}[task['kind']]
    tasks = sorted(items.values(), key=lambda t: t['task_id'])
    identity = {'ir':ir['ir_sha256'],'targets':targets}
    if identity_version == 2:
        identity.update(identity_version=2,task_contracts=[[task['task_id'],task['input_fingerprint']] for task in tasks])
    blueprint = {'blueprint_id':'bp-' + digest(identity)[:24],
                 'ir_sha256':ir['ir_sha256'], 'branch_sha256':ir['branch_sha256'],
                 'target_ids':targets, 'budget':budget,
                  'target_contracts':[{'target_id':target,
                     'intended_question': items[pilot]['intended_question'] if target == 'scalar-variance-bounds' else MILESTONE_MAP[target]['intended_question'] if target in MILESTONE_MAP else nodes[target]['statement'],
                     'scope': copy.deepcopy(items[pilot]['scope']) if target == 'scalar-variance-bounds' else copy.deepcopy(MILESTONE_MAP[target]['scope']) if target in MILESTONE_MAP else nodes[target]['scope']}
                     for target in targets],
                 'task_ids':[t['task_id'] for t in tasks],
                 'blocking_frontier':sorted((t['task_id'] for t in tasks if not t['dependencies']),
                     key=lambda tid:(-items[tid]['priority'],tid)),
                 'necessary_not_sufficient':True,
                 'frontier_semantics':'All listed blockers are jointly necessary; no OR alternative or assumption-completeness claim.'}
    if identity_version == 2:
        blueprint['identity_version'] = 2
    return blueprint, tasks


class ProofConductor:
    def __init__(self, workdir, coordinator=None, toolchain=None):
        self.coordinator = coordinator
        self.toolchain = toolchain
        self.base = load()
        self.journal = Journal(workdir)
        self._cache = {}
        self._proof_cache = {}
        self._live_cache = {}
        self.snapshot()

    @contextmanager
    def _transaction(self):
        # Lock ordering is engine mutex -> engine OS journal -> this journal.
        with ExitStack() as stack:
            if self.coordinator is not None:
                stack.enter_context(self.coordinator._mutex)
                stack.enter_context(self.coordinator.journal.transaction())
            stack.enter_context(self.journal.transaction())
            yield

    def _current_ir(self):
        if self.coordinator is None:
            key, data = 'baseline', self.base
            branch = None
            attachments = {}
        else:
            data, state = self.coordinator._replay()
            key = branch = state['branch_sha256']
            attachments = state['certificate_attachments']
        if key not in self._live_cache:
            source = copy.deepcopy(data)
            # Only an existing validated coordinator may provide a working
            # branch whose stored fingerprints still describe its baseline.
            # Offline source fingerprints must validate without normalization.
            if self.coordinator is not None:
                fps = fingerprints(source)
                for obligation in source['obligations']:
                    obligation['fingerprint'] = fps[obligation['conclusion']]
            self._live_cache = {key: compile_ir(source, branch)}
        self._attachments = attachments
        self._record_fingerprints = mathematical_fingerprints(data,attachments)
        return self._live_cache[key]

    def _stale(self, task):
        if task['kind'] == 'scalar_pilot':
            from polynomial_prover import profile_contract
            if task['profile_contract'] != profile_contract():
                return True
        records = [[rid,self._record_fingerprints[rid]] for rid in task['bound_record_ids']]
        actual = digest({'question':task['intended_question'],'scope':task['scope'],
                         'inputs':{'records':records}})
        return actual != task['input_fingerprint']

    @staticmethod
    def _used(s, task_id):
        return [c['strategy'] for c in s['claims'].values() if c['task_id'] == task_id]

    @staticmethod
    def _technical_retry(s, task_id, at):
        previous = next((c for c in reversed(list(s['claims'].values())) if c['task_id']==task_id),None)
        if previous is None:
            return False
        report = next((a for a in s['attempts'] if a['claim_id']==previous['claim_id']),None)
        if report is not None:
            return report['diagnosis'] in {'checker_failure','timeout'}
        return previous['expires_at'] <= at

    @classmethod
    def _strategy(cls, s, task_id, requested=None, at=None):
        used = cls._used(s, task_id)
        pilot = s['tasks'][task_id]['kind'] == 'scalar_pilot'
        if requested == 'native_scalar_profile' and not pilot:
            raise ValueError('native scalar strategy is restricted to the scalar pilot')
        if len(used) >= MAX_ATTEMPTS:
            raise ValueError('task attempt cap exhausted')
        technical_retry = pilot and cls._technical_retry(s,task_id,time.time() if at is None else at)
        if pilot and (not used or technical_retry) and requested in (None,'native_scalar_profile'):
            return 'native_scalar_profile'
        if requested is not None:
            if type(requested) is not str or requested not in STRATEGIES or requested in used:
                raise ValueError('unknown or previously attempted strategy')
            return requested
        previous = next((d for d in reversed(s['decisions']) if d['task_id'] == task_id), None)
        if previous and previous['strategy'] not in used:
            return previous['strategy']
        return next(strategy for strategy in STRATEGIES if strategy not in used and strategy != 'native_scalar_profile')

    @classmethod
    def _decision(cls, s, task, claim, diagnosis):
        action, preferred = DIAGNOSES[diagnosis]
        used = cls._used(s, task['task_id'])
        choices = [preferred] + [strategy for strategy in STRATEGIES if strategy != 'native_scalar_profile']
        strategy = next((p for p in choices if p not in used), None) if len(used) < MAX_ATTEMPTS else None
        technical_retry = task['kind']=='scalar_pilot' and diagnosis in {'checker_failure','timeout'} and len(used)<MAX_ATTEMPTS
        if technical_retry:
            strategy = 'native_scalar_profile'
        disposition = {'false_lemma':'investigate_untrusted_false_lemma_report',
                       'invalid_decomposition':'propose_complete_and_hyperedge_reconstruction',
                       'statement_drift':'restore_exact_question_contract'}.get(diagnosis,'investigate_report')
        return {'decision_id':'decision-' + claim['claim_id'], 'claim_id':claim['claim_id'],
                'task_id':task['task_id'], 'diagnosis':diagnosis, 'action':action,
                'previous_strategy':claim['strategy'], 'strategy':strategy,
                'disposition':disposition, 'question_sha256':task['question_sha256'],
                'scope_sha256':task['scope_sha256'], 'input_fingerprint':task['input_fingerprint'],
                'guidance': (STRATEGIES[strategy] if strategy else 'Attempt cap reached; explicit owner review required for further effort.'),
                'preserve_dependencies':task['dependencies'], 'proposed_only':True,
                'mathematical_verdict':'not_established_by_agent_report',
                'technical_retry_same_fixed_worker':technical_retry}

    def _read(self):
        if self.journal.path.exists() and self.journal.path.stat().st_size > MAX_JOURNAL_BYTES:
            raise ValueError('conductor journal byte bound exceeded')
        self.journal._load()
        if len(self.journal.events) > MAX_EVENTS:
            raise ValueError('conductor journal event bound exceeded')
        s = {'irs':{}, 'blueprints':{}, 'tasks':{}, 'claims':{}, 'attempts':[], 'decisions':[], 'completions':[], 'completion_checks':{}, 'budget_limit':None}
        for event in self.journal.events:
            kind, payload = event['type'], event['payload']
            if kind == 'IR_REGISTERED':
                key = digest(payload)
                if key not in self._cache:
                    validate_ir(payload)
                    self._cache[key] = True
                ir_hash = payload['ir_sha256']
                if ir_hash in s['irs']:
                    raise ValueError('duplicate IR registration')
                s['irs'][ir_hash] = payload
            elif kind == 'BLUEPRINT_CREATED':
                if set(payload) != {'blueprint','tasks','input_context'}:
                    raise ValueError('invalid blueprint event')
                bp = payload['blueprint']
                if bp['ir_sha256'] not in s['irs']:
                    raise ValueError('blueprint input is unregistered')
                context = payload['input_context']
                if type(context) is not dict or set(context) != {'attachments'}:
                    raise ValueError('invalid mathematical input context')
                bounded_json(context,'mathematical input context')
                historical_ir = s['irs'][bp['ir_sha256']]
                context_fps = mathematical_fingerprints(historical_ir['source_dataset'],context['attachments'])
                frozen_contract = None
                if 'scalar-variance-bounds' in bp['target_ids']:
                    pilots = [task for task in payload['tasks'] if type(task) is dict and task.get('kind')=='scalar_pilot']
                    if len(pilots) != 1 or type(pilots[0].get('profile_contract')) is not dict:
                        raise ValueError('missing frozen historical pilot contract')
                    frozen_contract = pilots[0]['profile_contract']
                expected, tasks = _build_blueprint(historical_ir,bp['target_ids'],bp['budget'],context_fps,pilot_contract=frozen_contract,identity_version=bp.get('identity_version',1))
                if payload != {'blueprint':expected,'tasks':tasks,'input_context':context}:
                    raise ValueError('forged blueprint or task contract')
                if bp['blueprint_id'] in s['blueprints']:
                    raise ValueError('duplicate blueprint event')
                if s['budget_limit'] is None:
                    s['budget_limit'] = bp['budget']
                if bp['budget'] > s['budget_limit']:
                    raise ValueError('global budget cannot be increased by a blueprint')
                s['blueprints'][bp['blueprint_id']] = bp
                for task in tasks:
                    old = s['tasks'].get(task['task_id'])
                    if old is not None:
                        stable_keys = set(task) - {'origin_ir_sha256','priority','downstream_impact'}
                        if {k:old[k] for k in stable_keys} != {k:task[k] for k in stable_keys}:
                            raise ValueError('shared task contract mismatch')
                    else:
                        s['tasks'][task['task_id']] = task
            elif kind == 'TASK_CLAIMED':
                self._validate_claim(s,payload)
                s['claims'][payload['claim_id']] = payload
            elif kind == 'FAILURE_REPORTED':
                self._validate_failure(s,payload)
                s['attempts'].append(payload['attempt'])
                s['decisions'].append(payload['decision'])
            elif kind == 'SCALAR_PILOT_COMPLETED':
                s['completion_checks'][payload['claim_id']] = self._validate_completion(s,payload)
                s['completions'].append(payload)
            else:
                raise ValueError('unknown conductor event; no mathematical authority: ' + kind)
        return s

    def _validate_claim(self, s, claim):
        fields = {'claim_id','task_id','agent_id','strategy','claimed_at','lease_seconds','expires_at',
                  'input_fingerprint','question_sha256','scope_sha256','artifact_directory'}
        if set(claim) != fields or claim.get('task_id') not in s['tasks']:
            raise ValueError('invalid or orphan claim')
        _text(claim['claim_id'],'claim id')
        if re.fullmatch(r'claim-[0-9a-f]{32}',claim['claim_id']) is None:
            raise ValueError('invalid claim identifier')
        if claim['claim_id'] in s['claims']:
            raise ValueError('duplicate claim')
        _text(claim['agent_id'],'agent id')
        expected_directory = Path('artifacts') / claim['claim_id']
        if claim['artifact_directory'] != expected_directory.as_posix():
            raise ValueError('claim artifact directory does not match assigned path')
        _number(claim['claimed_at'],'claim time',maximum=2**53)
        _number(claim['lease_seconds'],'lease')
        if claim['expires_at'] != claim['claimed_at'] + claim['lease_seconds']:
            raise ValueError('invalid claim expiry')
        task = s['tasks'][claim['task_id']]
        if task['dependencies']:
            raise ValueError('AND dependencies remain open')
        if any(c['task_id']==task['task_id'] for c in s['completions']):
            raise ValueError('scalar pilot already completed')
        for field in ('input_fingerprint','question_sha256','scope_sha256'):
            if claim[field] != task[field]:
                raise ValueError('claim contract mismatch')
        if len(s['claims']) >= s['budget_limit']:
            raise ValueError('global budget exhausted')
        reported = {a['claim_id'] for a in s['attempts']} | {c['claim_id'] for c in s['completions']}
        if any(c['task_id']==claim['task_id'] and c['claim_id'] not in reported and c['expires_at']>claim['claimed_at'] for c in s['claims'].values()):
            raise ValueError('task already leased')
        self._strategy(s,claim['task_id'],claim['strategy'],at=claim['claimed_at'])

    def _validate_failure(self, s, payload):
        if set(payload) != {'attempt','decision'}:
            raise ValueError('invalid failure event')
        attempt = payload['attempt']
        fields = {'attempt_id','claim_id','task_id','diagnosis','evidence','submitted_at','input_fingerprint','kind'}
        if set(attempt) != fields or attempt.get('claim_id') not in s['claims']:
            raise ValueError('invalid or orphan failure report')
        claim = s['claims'][attempt['claim_id']]
        if any(c['claim_id']==claim['claim_id'] for c in s['completions']):
            raise ValueError('claim completed; no pending failure report')
        if any(a['claim_id']==claim['claim_id'] for a in s['attempts']):
            raise ValueError('duplicate failure report')
        if type(attempt['diagnosis']) is not str or attempt['diagnosis'] not in DIAGNOSES:
            raise ValueError('invalid diagnosis')
        bounded_json(attempt['evidence'],'failure evidence',32768)
        _number(attempt['submitted_at'],'submission time',maximum=2**53)
        if not claim['claimed_at'] <= attempt['submitted_at'] < claim['expires_at']:
            raise ValueError('expired claim lease')
        if (attempt['attempt_id'] != 'attempt-' + claim['claim_id'] or attempt['task_id'] != claim['task_id'] or
            attempt['input_fingerprint'] != claim['input_fingerprint'] or attempt['kind'] != 'agent_failure_report'):
            raise ValueError('failure contract mismatch')
        expected = self._decision(s,s['tasks'][claim['task_id']],claim,attempt['diagnosis'])
        if payload['decision'] != expected:
            raise ValueError('forged failure decision or authority')

    def _append(self, kind, payload):
        if len(self.journal.events) >= MAX_EVENTS:
            raise ValueError('conductor journal event bound exceeded')
        # The journal sizes the exact canonical record under its own transaction.
        return self.journal.append(kind,payload,max_bytes=MAX_JOURNAL_BYTES)

    def blueprint(self, target_ids, budget=20):
        with self._transaction():
            ir = self._current_ir()
            bp, tasks = _build_blueprint(ir,target_ids,budget,self._record_fingerprints)
            s = self._read()
            old = s['blueprints'].get(bp['blueprint_id'])
            if old is not None:
                if old != bp:
                    raise ValueError('existing blueprint budget differs; cannot replenish budget')
                return copy.deepcopy(old)
            if s['budget_limit'] is not None and budget > s['budget_limit']:
                raise ValueError('global budget is fixed by the first authorized blueprint')
            if ir['ir_sha256'] not in s['irs']:
                self._append('IR_REGISTERED',ir)
            self._append('BLUEPRINT_CREATED',{'blueprint':bp,'tasks':tasks,'input_context':{'attachments':copy.deepcopy(self._attachments)}})
            return copy.deepcopy(bp)

    def claim(self, task_id, agent_id, strategy=None, lease_seconds=60):
        _text(task_id,'task id')
        _text(agent_id,'agent id')
        _number(lease_seconds,'lease')
        if strategy is not None and (type(strategy) is not str or strategy not in STRATEGIES):
            raise ValueError('unknown registered strategy')
        with self._transaction():
            ir, s = self._current_ir(), self._read()
            task = s['tasks'].get(task_id)
            if task is None:
                raise ValueError('unknown task')
            if self._stale(task):
                raise ValueError('stale task inputs; create a current blueprint')
            now = time.time()
            reported = {a['claim_id'] for a in s['attempts']} | {c['claim_id'] for c in s['completions']}
            active = next((c for c in s['claims'].values() if c['task_id']==task_id and c['claim_id'] not in reported and c['expires_at']>now),None)
            if active:
                if active['agent_id'] == agent_id and strategy in (None,active['strategy']):
                    return self._public_claim(active)
                raise ValueError('task already leased to an active claim')
            selected = self._strategy(s,task_id,strategy,at=now)
            claim_id = 'claim-' + uuid.uuid4().hex
            artifact_directory = self.journal.workdir.resolve() / 'artifacts' / claim_id
            claim = {'claim_id':claim_id, 'artifact_directory':(Path('artifacts')/claim_id).as_posix(), 'task_id':task_id, 'agent_id':agent_id,
                     'strategy':selected, 'claimed_at':now, 'lease_seconds':lease_seconds,
                     'expires_at':now + lease_seconds,
                     **{k:task[k] for k in ('input_fingerprint','question_sha256','scope_sha256')}}
            self._validate_claim(s,claim)
            if artifact_directory.parent.is_symlink():
                raise ValueError('artifact root is a symlink')
            artifact_directory.mkdir(parents=True,exist_ok=False)
            self._append('TASK_CLAIMED',claim)
            return self._public_claim(claim)

    def submit_failure(self, claim_id, diagnosis, evidence=None):
        _text(claim_id,'claim id')
        if type(diagnosis) is not str or diagnosis not in DIAGNOSES:
            raise ValueError('invalid typed failure diagnosis')
        evidence = bounded_json(evidence,'failure evidence',32768)
        with self._transaction():
            ir, s = self._current_ir(), self._read()
            claim = s['claims'].get(claim_id)
            if claim is None:
                raise ValueError('unknown claim')
            task = s['tasks'][claim['task_id']]
            if self._stale(task):
                raise ValueError('stale claim input fingerprint')
            old = next((a for a in s['attempts'] if a['claim_id']==claim_id),None)
            if old:
                if (old['diagnosis'],old['evidence']) != (diagnosis,evidence):
                    raise ValueError('claim already reported a different failure')
                return copy.deepcopy({'attempt':old,'decision':next(d for d in s['decisions'] if d['claim_id']==claim_id)})
            attempt = {'attempt_id':'attempt-' + claim_id,'claim_id':claim_id,'task_id':task['task_id'],
                       'diagnosis':diagnosis,'evidence':evidence,'submitted_at':time.time(),
                       'input_fingerprint':claim['input_fingerprint'],'kind':'agent_failure_report'}
            result = {'attempt':attempt,'decision':self._decision(s,task,claim,diagnosis)}
            self._validate_failure(s,result)
            self._append('FAILURE_REPORTED',result)
            return copy.deepcopy(result)

    def _public_claim(self,claim):
        result = copy.deepcopy(claim)
        result['artifact_relative_directory'] = claim['artifact_directory']
        result['artifact_directory'] = str(self.journal.workdir.resolve()/claim['artifact_directory'])
        return result

    def _public_completion(self,completion,verification=None):
        result = copy.deepcopy(completion)
        if verification is not None:
            result['recorded_verification'] = copy.deepcopy(completion['verification'])
            result['verification'] = copy.deepcopy(verification)
        result['receipt_relative_path'] = completion['receipt_path']
        result['receipt_path'] = str(self.journal.workdir.resolve()/completion['receipt_path'])
        return result

    def _artifact_receipt(self, claim, receipt_path):
        if not isinstance(receipt_path,(str,Path)):
            raise ValueError('invalid receipt path')
        assigned = self.journal.workdir.resolve() / 'artifacts' / claim['claim_id']
        path = Path(receipt_path)
        if '..' in path.parts:
            raise ValueError('receipt path traversal is forbidden')
        if not path.is_absolute():
            path = self.journal.workdir.resolve() / path
        path = path.absolute()
        if not path.is_relative_to(assigned) or not path.resolve().is_relative_to(assigned):
            raise ValueError('receipt path outside assigned artifact directory')
        cursor = path
        while cursor != assigned.parent:
            if cursor.is_symlink():
                raise ValueError('symlink in receipt path or assigned directory')
            cursor = cursor.parent
        try:
            if not path.is_file() or path.stat().st_size > 2*1024*1024:
                raise ValueError('receipt missing or exceeds bound')
            raw = path.read_bytes()
            receipt = bounded_json(json.loads(raw),'proof receipt',2*1024*1024)
            if type(receipt) is not dict:
                raise ValueError('receipt must be an object')
            project = path.parent / 'project'
            files = sorted(project.rglob('*'))
            if not project.is_dir() or project.is_symlink() or len(files)>32 or any(p.is_symlink() for p in files):
                raise ValueError('invalid receipt artifact tree')
            inventory = []
            for artifact in files:
                if artifact.is_file():
                    if artifact.stat().st_size > 2*1024*1024:
                        raise ValueError('artifact byte bound exceeded')
                    inventory.append([artifact.relative_to(project).as_posix(),hashlib.sha256(artifact.read_bytes()).hexdigest()])
            return path,receipt,hashlib.sha256(raw).hexdigest(),digest(inventory)
        except (OSError,TypeError,json.JSONDecodeError) as exc:
            raise ValueError('invalid proof receipt or artifacts') from exc

    def _completion(self, s, claim, receipt_path, submitted_at, historical=False):
        task = s['tasks'][claim['task_id']]
        if task['kind'] != 'scalar_pilot':
            raise ValueError('only the exact scalar pilot can be completed; canonical and D5 tasks remain open')
        if claim['strategy'] != 'native_scalar_profile':
            raise ValueError('claim strategy does not authorize the fixed scalar profile worker')
        if any(c['claim_id']==claim['claim_id'] for c in s['completions']) or any(a['claim_id']==claim['claim_id'] for a in s['attempts']):
            raise ValueError('claim is no longer pending or is already completed')
        _number(submitted_at,'completion time',maximum=2**53)
        if not claim['claimed_at'] <= submitted_at < claim['expires_at']:
            raise ValueError('expired claim lease')
        from polynomial_prover import profile_contract, verify_proof_receipt
        current_contract = profile_contract()
        contract = task['profile_contract']
        stale_contract = contract != current_contract
        if task['scope']['profile_contract_sha256'] != digest(contract) or (stale_contract and not historical):
            raise ValueError('stale scalar pilot profile contract')
        path,receipt,receipt_hash,artifact_hash = self._artifact_receipt(claim,receipt_path)
        for key in ('profile_id','profile_sha256','scope','alignment_status','statements','source_hashes','actual_field_applicable','full_theorem_certificates'):
            if receipt.get(key) != contract[key]:
                raise ValueError('receipt profile contract mismatch: '+key)
        if stale_contract:
            # Custody and the frozen question still validate, but a new worker
            # manifest cannot authenticate old proof authority. Never execute
            # old commands, helpers or profile implementations.
            verification = {'valid':False,'state':'stale_profile_contract','fresh_replay':False,
                'reasons':['Historical scalar worker/profile contract differs from the current registry.'],
                'full_theorem_certificates':[]}
        else:
            backend = native_backend_identity(self.toolchain)
            if getattr(self,'_proof_backend_signature',None) != backend['signature']:
                self._proof_cache.clear()
                self._proof_backend_signature = backend['signature']
            cache_key = digest({'receipt':receipt_hash,'artifacts':artifact_hash,'contract':contract,'path':str(path),'backend':backend['signature']})
            verification = self._proof_cache.get(cache_key)
            if verification is None:
                verification = verify_proof_receipt(path,toolchain=self.toolchain)
                if verification.get('valid') is True and native_backend_identity(self.toolchain)['signature'] != backend['signature']:
                    verification = {'valid':False,'state':'blocked','fresh_replay':False,
                        'reasons':['Native backend metadata changed during independent replay.'],
                        'full_theorem_certificates':[]}
                if verification.get('valid') is not True or verification.get('fresh_replay') is not True or verification.get('actual_field_applicable') is not False or verification.get('full_theorem_certificates') != []:
                    if not historical or verification.get('state') != 'blocked':
                        raise ValueError('independent native receipt verification failed: '+str(verification.get('reasons')))
                after = self._artifact_receipt(claim,path)
                if after[2:] != (receipt_hash,artifact_hash):
                    raise ValueError('proof receipt changed during independent replay')
                if verification.get('valid') is True:
                    self._proof_cache[cache_key] = verification
        current_verification = copy.deepcopy(verification)
        if historical and verification.get('state') in {'blocked','stale_profile_contract'}:
            verification = {'valid':True,'state':'scoped_kernel_checked','reasons':[],
                'fresh_replay':True,'artifact_integrity':True,'axiom_free_count':2,
                'actual_field_applicable':False,'full_theorem_certificates':[]}
        result = {'kind':'verified_scalar_pilot','claim_id':claim['claim_id'],'task_id':task['task_id'],
                'input_fingerprint':task['input_fingerprint'],'profile_contract_sha256':digest(contract),
                'receipt_path':path.relative_to(self.journal.workdir.resolve()).as_posix(),'receipt_file_sha256':receipt_hash,'artifact_tree_sha256':artifact_hash,
                'submitted_at':submitted_at,'verification':copy.deepcopy(verification),
                'worker_attempts':copy.deepcopy(receipt['attempts']),
                'actual_field_applicable':False,'full_theorem_certificates':[]}
        if historical:
            result['_current_verification'] = current_verification
        return result

    def _validate_completion(self,s,payload):
        if payload.get('claim_id') not in s['claims']:
            raise ValueError('orphan scalar completion')
        claim = s['claims'][payload['claim_id']]
        expected = self._completion(s,claim,payload.get('receipt_path'),payload.get('submitted_at'),historical=True)
        current_verification = expected.pop('_current_verification')
        if expected != payload:
            raise ValueError('forged scalar completion or scope authority')
        return current_verification

    def complete_claim(self, claim_id, receipt_path):
        """Close only the registered scalar pilot after independent native replay."""
        _text(claim_id,'claim id')
        with self._transaction():
            ir,s = self._current_ir(),self._read()
            claim = s['claims'].get(claim_id)
            if claim is None:
                raise ValueError('unknown claim')
            task = s['tasks'][claim['task_id']]
            if self._stale(task):
                raise ValueError('stale claim input fingerprint')
            result = self._completion(s,claim,receipt_path,time.time())
            # A checker run must still finish inside its reserved lease, with
            # stable source/profile inputs protected by the engine-first locks.
            if time.time() >= claim['expires_at']:
                raise ValueError('claim lease expired during native replay')
            self._current_ir()
            if self._stale(task):
                raise ValueError('stale inputs during native replay')
            self._append('SCALAR_PILOT_COMPLETED',result)
            return self._public_completion(result)

    def snapshot(self):
        with self._transaction():
            ir, s = self._current_ir(), self._read()
            now = time.time()
            reported = {a['claim_id'] for a in s['attempts']}
            completed_claims = {c['claim_id'] for c in s['completions'] if s['completion_checks'][c['claim_id']].get('valid') is True}
            completed_tasks = {c['task_id'] for c in s['completions'] if c['claim_id'] in completed_claims}
            historical_claims = {c['claim_id'] for c in s['completions']} - completed_claims
            historical_tasks = {c['task_id'] for c in s['completions'] if c['claim_id'] in historical_claims}
            historical_states = {c['claim_id']:('historical_scalar_pilot_profile_stale' if s['completion_checks'][c['claim_id']].get('state')=='stale_profile_contract' else 'historical_scalar_pilot_replay_blocked') for c in s['completions'] if c['claim_id'] in historical_claims}
            historical_task_states = {c['task_id']:historical_states[c['claim_id']] for c in s['completions'] if c['claim_id'] in historical_claims}
            claims = []
            for original in s['claims'].values():
                claim = self._public_claim(original)
                claim['state'] = ('completed_scalar_pilot' if claim['claim_id'] in completed_claims else historical_states[claim['claim_id']] if claim['claim_id'] in historical_claims else 'reported' if claim['claim_id'] in reported else
                                  'expired' if claim['expires_at'] <= now else
                                  'stale' if self._stale(s['tasks'][claim['task_id']]) else 'active')
                claims.append(claim)
            tasks = []
            for original in s['tasks'].values():
                task = copy.deepcopy(original)
                task['completed'] = task['task_id'] in completed_tasks
                task['stale'] = self._stale(task)
                task['blueprint_ids'] = sorted(b['blueprint_id'] for b in s['blueprints'].values() if task['task_id'] in b['task_ids'])
                task['attempt_count'] = len(self._used(s,task['task_id']))
                task['active_claim_id'] = next((c['claim_id'] for c in claims if c['task_id']==task['task_id'] and c['state']=='active'),None)
                task['claimable'] = task['task_id'] not in historical_tasks and not task['completed'] and not task['stale'] and not task['dependencies'] and not task['active_claim_id'] and task['attempt_count'] < MAX_ATTEMPTS
                latest = next((d for d in reversed(s['decisions']) if d['task_id']==task['task_id']),None)
                task['recommended_strategy'] = latest['strategy'] if latest else 'native_scalar_profile' if task['kind']=='scalar_pilot' else 'dependency_decomposition'
                task['state'] = 'completed_scalar_pilot' if task['completed'] else historical_task_states[task['task_id']] if task['task_id'] in historical_tasks else 'stale' if task['stale'] else 'leased' if task['active_claim_id'] else 'exhausted' if task['attempt_count']>=MAX_ATTEMPTS else 'blocked' if task['dependencies'] else 'open'
                tasks.append(task)
            reserved = sum(c['state']=='active' for c in claims)
            blueprints = [dict(copy.deepcopy(b),stale=any(self._stale(s['tasks'][tid]) for tid in b['task_ids'])) for b in s['blueprints'].values()]
            return {'version':VERSION, 'ir':copy.deepcopy(ir),
                    'blueprints':blueprints, 'tasks':sorted(tasks,key=lambda t:(-t['priority'],t['task_id'])),
                    'claims':claims, 'attempts':copy.deepcopy(s['attempts']), 'decisions':copy.deepcopy(s['decisions']),
                    'completions':[self._public_completion(c,s['completion_checks'][c['claim_id']]) for c in s['completions']],
                    'budget':{'limit':s['budget_limit'] or 0,'reserved':reserved,'spent':len(claims)-reserved,
                              'remaining':(s['budget_limit'] or 0)-len(claims),'attempt_cap':MAX_ATTEMPTS},
                    'capabilities':{'effects':list(EFFECTS)+[PILOT_EFFECT],'strategies':copy.deepcopy(STRATEGIES),
                                    'registered_checkers':{'native_scalar_profile':{'effect':PILOT_EFFECT,'target_kind':'scalar_pilot','checker':'polynomial_prover.verify_proof_receipt'}},
                                    'mathematical_approval':False,'submitted_code_execution':False},
                    'last_sequence':len(self.journal.events),
                    'journal_head_sha256':self.journal.events[-1]['event_hash'] if self.journal.events else '0'*64,
                    'full_theorem_certificates':0,'status':'PLANNING_AND_AGENT_REPORTS_NOT_MATH_CERTIFIED'}
