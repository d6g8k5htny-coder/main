#!/usr/bin/env python3
"""Offline event-driven proof-obligation conductor (not a theorem prover).

Original v15 dataset and its hashes stay frozen. Edits are proposals in an
append-only, hash-chained journal; approvals create a working branch, invalidate
transitive mathematical dependents, and schedule bounded *local* diagnostics.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, queue, secrets, subprocess, sys, tempfile, threading, time, uuid
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from model import load, validate, fingerprints, digest, GateError

ROOT=Path(__file__).resolve().parent
ALLOWED={'statement', 'premise', 'side_condition', 'import', 'correction', 'local_evidence', 'certificate', 'research_evidence'}
MAX_VALUE=20000
ZERO='0'*64

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf8')

def change_key(kind, id):return kind+':'+id

def find_item(data,kind,key):
    groups={'research_evidence':('records','id'), 'statement':('records','id'), 'premise':('obligations','conclusion'), 'certificate':('records','id'), 'side_condition':('obligations','conclusion'),
            'import':('imports','id'),'correction':('corrections','id'),
            'local_evidence':('local_evidence','id')}
    if kind not in groups:raise ValueError('unknown change type')
    group,field=groups[kind]
    found=[item for item in data[group] if item[field]==key]
    if len(found)!=1:raise ValueError('target does not exist: '+key)
    return found[0]

def apply_change(data,kind,key,value):
    obj=find_item(data,kind,key)
    if kind=='statement':obj['statement']=value
    elif kind=='premise':obj['premises']=sorted(json.loads(value) if isinstance(value,str) else value)
    elif kind=='side_condition':obj['side_conditions'][0]['text']=value
    elif kind=='import':obj['scope_description']=value
    elif kind=='correction':obj['role']=value
    elif kind=='local_evidence':obj['limits']=value


def validate_edit(data,kind,key,value):
    if kind not in ALLOWED:raise ValueError('unsupported field')
    if not isinstance(key,str) or not 1<=len(key)<=128:raise ValueError('invalid record id')
    if kind in {'premise','certificate','research_evidence'} and not isinstance(value,str):
        value=json.dumps(value,sort_keys=True)
    if not isinstance(value,str) or not 1<=len(value)<=MAX_VALUE:raise ValueError('invalid replacement text')
    test=copy.deepcopy(data)
    if kind=='research_evidence':
        from research_evidence import validate_attachment
        find_item(test,kind,key)
        validate_attachment(json.loads(value),test,key)
        return test
    if kind=='certificate':
        find_item(test,kind,key)
        descriptor=json.loads(value)
        if not isinstance(descriptor,dict):raise ValueError('certificate metadata must be an object')
        if descriptor.get('record_id')!=key or descriptor.get('obligation_id','OBL:'+key)!='OBL:'+key:
            raise ValueError('certificate-to-statement target mismatch')
        return test
    if kind=='premise':
        proposed=json.loads(value)
        if not isinstance(proposed,list) or not all(isinstance(p,str) for p in proposed):raise ValueError('premises must be a JSON list of record identifiers')
    apply_change(test,kind,key,value)
    if digest(test)==digest(data):raise ValueError('replacement is unchanged; no event required')
    validate(test,require_stored_fingerprints=False)
    return test

def impact_from(base,updated):
    before=fingerprints(base);after=fingerprints(updated)
    if set(before)!=set(after):raise ValueError('record membership changed')
    affected=sorted(k for k in before if before[k]!=after[k])
    primary={r['id'] for r in base['records'] if r['project_acceptance']=='primary'}
    return {'affected':affected,'primary':sorted(primary.intersection(affected)),
            'count':len(affected),'fingerprints':{k:after[k] for k in affected}}

def diagnostic(data, affected):
    obs={o['conclusion']:o for o in data['obligations']}
    imp={i['id']:i for i in data['imports']}
    corr={c['id']:c for c in data['corrections']}
    result=[]
    for rid in affected:
        o=obs[rid]
        unvalidated=[i for i in o['external_theorem_imports'] if imp[i]['import_status']!='independently_certified']
        corrections=[i for i in o['source_corrections'] if not corr[i]['validated']]
        result.append({'id':rid,'proof_state':'needs_mathematical_reverification',
                       'import_gaps':unvalidated,'source_correction_gaps':corrections,
                       'side_condition_count':len(o['side_conditions']),
                       'next_action': 'Check source corrections and imported theorem scopes; attach a scope-aligned external proof certificate.'})
    return result

def mermaid_source(snapshot, record_id):
    """Deterministic text projection of current proof requirements (not a proof)."""
    records={r['id']:r for r in snapshot['records']}
    obligations={o['id']:o for o in snapshot['obligations']}
    if record_id not in records:raise ValueError('unknown theorem')
    r=records[record_id];o=obligations[record_id]
    def label(s):
        return str(s).replace('"','\\"').replace('\n',' ').replace('|','/')[:115]
    lines=['flowchart TD', '  CONCL["'+label(record_id+' · '+r['title'])+'"]',
           '  JOINT{{"Joint proof obligation"}}','  JOINT --> CONCL']
    for i,p in enumerate(o['premises']):
        lines.append('  P%d["%s"] --> JOINT'%(i,label(p)))
    for i,c in enumerate(o['side_conditions']):
        lines.append('  SC%d{{"%s"}} --> JOINT'%(i,label(c['text'])))
    for i,c in enumerate(o['external_imports']):
        lines.append('  IMP%d["Import: %s"] --> JOINT'%(i,label(c)))
    for i,c in enumerate(o['corrections']):
        lines.append('  COR%d["Correction: %s"] --> JOINT'%(i,label(c)))
    lines += ['  classDef stale fill:#624b29,stroke:#edc277,color:#fff',
              '  classDef uncertified fill:#27354e,stroke:#859ab5,color:#fff',
              '  class CONCL '+('stale' if record_id in snapshot['changed_ids'] else 'uncertified')]
    return '\n'.join(lines)+'\n'

class Journal:
    """Fsync'd hash-chained events; explicit delivery IDs make retries idempotent.

    A truncated or malformed journal fails closed. The service owns one runtime;
    an OS lock serializes appends across local processes and threads.
    """
    def __init__(self,workdir):
        self.workdir=Path(workdir)
        self.workdir.mkdir(parents=True,exist_ok=True)
        self.path=self.workdir/'events.jsonl'
        self._mutex=threading.RLock()
        self._cond=threading.Condition(self._mutex)
        self._lock_depth=threading.local()
        self.events=[]
        with self.transaction():self._load()

    @contextmanager
    def transaction(self):
        with self._mutex:
            depth=getattr(self._lock_depth,'value',0)
            if depth:
                self._lock_depth.value=depth+1
                try:yield
                finally:self._lock_depth.value=depth
                return
            lock=(self.workdir/'.journal.lock').open('a+b')
            try:
                try:
                    import fcntl
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                except ImportError:pass  # Windows supports the documented single writer.
                self._lock_depth.value=1
                yield
            finally:
                self._lock_depth.value=0
                lock.close()

    def _load(self):
        events=[];prev=ZERO;ids={}
        if not self.path.exists():self.events=[];return
        raw=self.path.read_bytes()
        if raw and not raw.endswith(b'\n'):raise ValueError('incomplete journal tail; preserve and repair explicitly before resuming')
        for n,line in enumerate(raw.decode('utf8').splitlines(),1):
            if not line.strip():raise ValueError('empty journal line '+str(n))
            event=json.loads(line)
            if not isinstance(event,dict) or not isinstance(event.get('payload'),dict) or not isinstance(event.get('type'),str):
                raise ValueError('malformed journal event at line '+str(n))
            if event['seq']!=len(events)+1 or event['prev_hash']!=prev:
                raise ValueError('event sequence/previous hash corrupted at line '+str(n))
            expected=hashlib.sha256(canonical({k:v for k,v in event.items() if k!='event_hash'})).hexdigest()
            if event.get('event_hash')!=expected:raise ValueError('tampered event at line '+str(n))
            eid=event.get('event_id')
            if eid:
                if eid in ids:raise ValueError('duplicate event ID in stored journal')
                ids[eid]=event
            events.append(event);prev=expected
        self.events=events

    def append(self,etype,payload,event_id=None):
        if event_id is not None and (not isinstance(event_id,str) or not 1<=len(event_id)<=180):raise ValueError('invalid event ID')
        with self.transaction():
            self._load()
            if event_id:
                for old in self.events:
                    if old.get('event_id')==event_id:
                        if old['type']!=etype or old['payload']!=payload:raise ValueError('event ID reused with different content')
                        return copy.deepcopy(old)
            d={'seq':len(self.events)+1,'when_utc':datetime.now(timezone.utc).isoformat(),
               'type':etype,'payload':copy.deepcopy(payload),'prev_hash':self.events[-1]['event_hash'] if self.events else ZERO}
            if event_id:d['event_id']=event_id
            d['event_hash']=hashlib.sha256(canonical(d)).hexdigest()
            with self.path.open('ab') as f:
                f.write(canonical(d)+b'\n');f.flush();os.fsync(f.fileno())
            self.events.append(d);self._cond.notify_all();return copy.deepcopy(d)

    def since(self,seq):
        with self.transaction():
            self._load()
            return [copy.deepcopy(e) for e in self.events if e['seq']>seq]


def downstream_closure(data,seeds):
    children=defaultdict(set)
    for o in data['obligations']:
        for p in o['premises']:children[p].add(o['conclusion'])
    closure=set(seeds);todo=list(seeds)
    while todo:
        for child in children[todo.pop()]:
            if child not in closure:closure.add(child);todo.append(child)
    return sorted(closure)


def mathematical_fingerprints(data,attachments=None):
    """Canonical fingerprints plus quarantined/approved evidence inputs.

    Approved attachments never modify the immutable source dataset or theorem
    truth. Their change still invalidates every dependent inference.
    """
    base=fingerprints(data);attachments=attachments or {}
    obs={o['conclusion']:o for o in data['obligations']};result={}
    def go(rid):
        if rid in result:return result[rid]
        parents=[[p,go(p)] for p in obs[rid]['premises']]
        if attachments.get(rid) or any(fp!=base[p] for p,fp in parents):
            result[rid]=digest({'canonical':base[rid],'certificate_inputs':attachments.get(rid,[]),'premises':parents})
        else:result[rid]=base[rid]
        return result[rid]
    for rid in sorted(base):go(rid)
    return result


class Coordinator:
    def __init__(self,workdir,max_workers=2,max_attempts=3,lease_seconds=30,retry_backoff=.05):
        if not isinstance(max_workers,int) or not 1<=max_workers<=8:raise ValueError('worker concurrency must be between 1 and 8')
        if not isinstance(max_attempts,int) or not 1<=max_attempts<=10:raise ValueError('attempt cap must be between 1 and 10')
        if not 0<lease_seconds<=3600 or not 0<=retry_backoff<=60:raise ValueError('invalid scheduler bounds')
        self.base=load();validate(self.base)
        self.journal=Journal(workdir)
        self._mutex=threading.RLock()
        self.max_workers=max_workers;self.max_attempts=max_attempts
        self.lease_seconds=float(lease_seconds);self.retry_backoff=float(retry_backoff)
        self.tasks=queue.Queue();self.thread=None;self.threads=[];self.started=False
        self.inbox=Path(workdir)/'inbox';self.inbox.mkdir(exist_ok=True)
        self._stop=threading.Event();self._service_lock=None
        self.snapshot()  # Validate all historical events before offering any API.

    def _replay(self,until=None):
        with self.journal.transaction():
            self.journal._load()
            events=[copy.deepcopy(e) for e in self.journal.events if until is None or e['seq']<=until]
        effective=copy.deepcopy(self.base)
        proposals={};runs=[];jobs={};attachments={};certificates=[];changes=[];required={};reinspection=set()
        research={};latest_research={};research_failures={}
        for e in events:
            p=e['payload'];t=e['type']
            if t=='RESEARCH_CAPTURED':
                from research_evidence import validate_capture
                validate_capture(p,effective)
                research[p['revision_id']]=p
                latest_research[p['source_key']]=p['revision_id']
                research_failures.pop(p['source_key'],None)
            elif t=='RESEARCH_CAPTURE_FAILED':
                if not isinstance(p.get('source_key'),str) or not isinstance(p.get('reason'),str):
                    raise ValueError('invalid research failure event')
                research_failures[p['source_key']]=p['reason']
            elif t=='RESEARCH_CAPTURE_RECOVERED':
                if not isinstance(p.get('source_key'),str) or p['source_key'] not in research_failures:
                    raise ValueError('orphan research recovery')
                research_failures.pop(p['source_key'])
            elif t=='PROPOSAL_CREATED':
                pid=p['proposal_id']
                if pid in proposals:raise ValueError('duplicate proposal')
                validate_edit(effective,p['kind'],p['target'],p['replacement'])
                if p['kind']=='research_evidence':
                    capture=json.loads(p['replacement'])['capture']
                    if research.get(capture['revision_id'])!=capture:raise ValueError('unregistered research capture')
                proposals[pid]={**p,'status':'awaiting_approval'}
            elif t=='PROPOSAL_REJECTED':
                if p['proposal_id'] not in proposals or proposals[p['proposal_id']]['status']!='awaiting_approval':raise ValueError('orphan rejection')
                proposals[p['proposal_id']]['status']='rejected'
            elif t=='PROPOSAL_APPROVED':
                pid=p['proposal_id']
                if pid not in proposals or proposals[pid]['status']!='awaiting_approval':raise ValueError('orphan or repeated approval')
                proposal=proposals[pid]
                if p.get('baseline_dataset_sha256',digest(self.base))!=digest(self.base):raise ValueError('approval belongs to a different baseline')
                if proposal['base_effective_dataset_sha256']!=digest(effective):raise ValueError('stale approval in journal')
                branch_hash=digest({'data':digest(effective),'attachments':attachments})
                if proposal.get('base_branch_sha256',branch_hash)!=branch_hash:raise ValueError('stale certificate branch approval')
                before=mathematical_fingerprints(effective,attachments)
                if proposal['kind']=='research_evidence':
                    from research_evidence import attach_research
                    descriptor=json.loads(proposal['replacement'])
                    capture=descriptor['capture']
                    if latest_research.get(capture['source_key'])!=capture['revision_id'] or capture['source_key'] in research_failures:
                        raise ValueError('stale research evidence approval')
                    attach_research(attachments,proposal['target'],descriptor)
                elif proposal['kind']=='certificate':
                    descriptor=json.loads(proposal['replacement'])
                    attachments.setdefault(proposal['target'],[]).append(descriptor)
                else:effective=validate_edit(effective,proposal['kind'],proposal['target'],proposal['replacement'])
                after=mathematical_fingerprints(effective,attachments)
                affected=sorted(k for k in before if before[k]!=after[k])
                if 'affected' in p and p['affected']!=affected:raise ValueError('approval impact mismatch')
                proposals[pid]['status']='approved';changes.append(e['seq']);reinspection.update(affected)
                # The approval is the durable transaction/outbox: queue survives a
                # crash before any JOB_QUEUED notification is appended.
                approved_jobs=p.get('jobs')
                if approved_jobs is None:
                    from workers import make_job
                    approved_jobs=[make_job(effective,rid,{'event_id':'approval:'+pid,'sequence':e['seq'],'proposal_id':pid}) for rid in affected]
                if not isinstance(approved_jobs,list) or sorted(spec.get('record_id','') for spec in approved_jobs)!=affected:
                    raise ValueError('approval job coverage does not match affected mathematical obligations')
                attempt_cap=p.get('max_attempts',3)
                if isinstance(attempt_cap,bool) or not isinstance(attempt_cap,int) or not 1<=attempt_cap<=10:raise ValueError('invalid durable retry cap')
                canonical_fp=fingerprints(effective)
                for spec in approved_jobs:
                    from research_evidence import relevant_captures
                    expected_research=relevant_captures(effective,attachments,spec['record_id'])
                    if spec.get('research_inputs',[])!=expected_research:raise ValueError('queued research input binding mismatch')
                    rid=spec['record_id'];jid=spec['job_id']
                    if not isinstance(jid,str) or len(jid)!=34 or not jid.startswith('j-') or any(c not in '0123456789abcdef' for c in jid[2:]):raise ValueError('invalid durable job identifier')
                    if rid not in affected or spec.get('mathematical_fingerprint',after[rid])!=after[rid]:raise ValueError('queued job does not match approved mathematical input')
                    if spec.get('obligation_id')!='OBL:'+rid or spec.get('inputs',{}).get('obligation_fingerprint')!=canonical_fp[rid]:raise ValueError('queued obligation binding mismatch')
                    if spec.get('inputs',{}).get('baseline_dataset_sha256')!=digest(self.base):raise ValueError('queued job baseline mismatch')
                    if jid in jobs:raise ValueError('duplicate approved job')
                    old_id=required.get(rid)
                    if old_id and jobs[old_id]['state'] in {'queued','retry_wait'}:jobs[old_id]['state']='stale'
                    jobs[jid]={**spec,'mathematical_fingerprint':after[rid],'state':'queued','attempts':0,
                               'max_attempts':attempt_cap,'lease_token':None,'lease_expires_at':None,
                               'approval_sequence':e['seq'],'receipt':None,'escalated':False,'next_attempt_at':0}
                    required[rid]=jid
            elif t=='JOB_QUEUED':
                if p['job_id'] not in jobs:raise ValueError('orphan job queue notification')
            elif t=='JOB_LEASED':
                if p['job_id'] not in jobs:raise ValueError('orphan job lease')
                job=jobs[p['job_id']]
                if job['state'] not in {'queued','retry_wait'}:raise ValueError('invalid job lease transition')
                if p['attempt']!=job['attempts']+1 or p['attempt']>job['max_attempts']:raise ValueError('invalid job attempt')
                if not isinstance(p.get('lease_token'),str) or len(p['lease_token'])!=32 or any(c not in '0123456789abcdef' for c in p['lease_token']):raise ValueError('invalid durable lease token')
                job.update(state='running',attempts=p['attempt'],lease_token=p['lease_token'],
                           lease_expires_at=p['lease_expires_at'],worker_id=p['worker_id'])
            elif t in {'JOB_FINISHED','JOB_RETRY_SCHEDULED','JOB_ESCALATED','JOB_STALE'}:
                if p['job_id'] not in jobs:raise ValueError('orphan job outcome')
                job=jobs[p['job_id']]
                if t!='JOB_STALE' and (job['state']!='running' or job['lease_token']!=p.get('lease_token')):raise ValueError('invalid job outcome lease')
                if t=='JOB_FINISHED':
                    if p['result'] not in {'diagnostic_completed','blocked','failed'}:raise ValueError('invalid worker result')
                    job.update(state=p['result'],receipt=p.get('receipt'),artifact_directory=p.get('artifact_directory'))
                elif t=='JOB_RETRY_SCHEDULED':job.update(state='retry_wait',next_attempt_at=p['next_attempt_at'],last_error=p['reason'],receipt=p.get('receipt'))
                elif t=='JOB_ESCALATED':job.update(state='failed',escalated=True,last_error=p['reason'],receipt=p.get('receipt'))
                else:job.update(state='stale',last_error=p['reason'])
                job['lease_token']=None;job['lease_expires_at']=None
            elif t=='JOB_RESULT_REJECTED':
                if p['job_id'] not in jobs:raise ValueError('orphan rejected result')
            elif t=='CERTIFICATE_EVALUATED':
                if p['proposal_id'] not in proposals or proposals[p['proposal_id']]['status']!='approved':raise ValueError('orphan certificate evaluation')
                if p['verdict'].get('formal_status')!='not_certified' or p['verdict'].get('theorem_certificates_awarded',0)!=0:raise ValueError('unearned mathematical certification')
                certificates.append(p)
            elif t=='CHECK_RUN':runs.append(e)
            else:raise ValueError('unrecognized event type: '+str(t))
        delta=impact_from(self.base,effective);fps=mathematical_fingerprints(effective,attachments)
        last_approval=max(changes,default=0);expected_dataset_hash=digest(effective)
        for proposal in proposals.values():
            proposal['ui_status']=('stale' if proposal['status']=='awaiting_approval' and
                (proposal['base_effective_dataset_sha256']!=expected_dataset_hash or proposal.get('base_branch_sha256') not in (None,digest({'data':expected_dataset_hash,'attachments':attachments}))) else proposal['status'])
            if proposal['kind']=='research_evidence' and proposal['status']=='awaiting_approval':
                capture=json.loads(proposal['replacement'])['capture']
                if latest_research.get(capture['source_key'])!=capture['revision_id'] or capture['source_key'] in research_failures:
                    proposal['ui_status']='stale'
        successful_checks=[e for e in runs if e['seq']>last_approval and e['payload'].get('result')=='RECALCULATED_STILL_NEEDS_MATH_VERIFICATION'
            and e['payload'].get('effective_dataset_sha256')==expected_dataset_hash and e['payload'].get('last_approved_sequence')==last_approval]
        current_jobs=[jobs[jid] for jid in required.values()]
        pending=bool(last_approval and (not successful_checks or any(j['state']!='diagnostic_completed' for j in current_jobs)))
        by_check=[e['payload'] for e in runs if e['seq']>last_approval]
        records=[]
        for r in effective['records']:
            rid=r['id'];latest=jobs.get(required.get(rid))
            records.append({'id':rid,'title':r['title'],'topic':r['topic'],'acceptance':r['project_acceptance'],
                            'project_acceptance':r['project_acceptance'],'statement':r['statement'],'scope':r['scope'],
                            'status':'needs_mathematical_reverification' if rid in reinspection else 'uncertified_baseline',
                            'proof_state':'needs_mathematical_reverification' if rid in reinspection else 'side_conditions_open',
                            'worker_state':latest['state'] if latest else 'not_queued','formal_status':'not_certified',
                            'certificate_status':'not_certified','fingerprint':fps[rid],'source':r['source'],'equations':r['equations']})
        state={'status':'WORKING_BRANCH_NOT_MATH_CERTIFIED','baseline':'UL-obligation-certificate-v15.0',
            'event_protocol_version':'research-flow-events-1','worker_protocol_version':'research-flow-worker-1',
            'baseline_dataset_sha256':digest(self.base),'effective_dataset_sha256':expected_dataset_hash,
            'branch_sha256':digest({'data':expected_dataset_hash,'attachments':attachments}),
            'last_sequence':events[-1]['seq'] if events else 0,'journal_head_sha256':events[-1]['event_hash'] if events else ZERO,
            'primary_count':sum(r['project_acceptance']=='primary' for r in effective['records']),'record_count':len(effective['records']),
            'changed_count':len(reinspection),'changed_ids':sorted(reinspection),'net_changed_ids':delta['affected'],
            'changed_primary':sorted(r['id'] for r in effective['records'] if r['project_acceptance']=='primary' and r['id'] in reinspection),
            'proposals':sorted(proposals.values(),key=lambda x:x['proposal_id']),'approvals_pending_diagnostics':pending,
            'checks':by_check[-10:],'events':events[-120:],'records':records,
            'obligations':[{'id':o['conclusion'],'premises':o['premises'],'side_conditions':o['side_conditions'],
                            'external_imports':o['external_theorem_imports'],'corrections':o['source_corrections'],
                            'certificate_count':0,'proof_state':'needs_mathematical_reverification' if o['conclusion'] in reinspection else o['status']} for o in effective['obligations']],
            'imports':effective['imports'],'corrections':effective['corrections'],'audit_history':effective['audit_history'],
            'open_import_classes':sum(i['import_status']!='independently_certified' for i in effective['imports']),
            'full_theorem_certificates':0,'latest_checks_not_certificates':True,'certificate_attachments':attachments,
            'certificate_evaluations':certificates,'jobs':list(jobs.values()),
            'research_discoveries':[{**research[revision],'capture_state':('source_unavailable' if source in research_failures else 'captured')} for source,revision in sorted(latest_research.items())],
            'research_revision_count':len(research),'research_failures':research_failures,
            'research_evidence_attachments':{rid:[v for v in values if v.get('kind')=='research_evidence'] for rid,values in attachments.items()},'required_job_ids':required,
            'pending_job_count':sum(j['state'] in {'queued','retry_wait'} for j in current_jobs),
            'running_job_count':sum(j['state']=='running' for j in current_jobs),
            'failed_job_count':sum(j['state']=='failed' for j in current_jobs),
            'blocked_job_count':sum(j['state']=='blocked' for j in current_jobs),
            'diagnostic_completed_count':sum(j['state']=='diagnostic_completed' for j in current_jobs)}
        return effective,state

    def snapshot(self,at=None):
        if at is not None and (not isinstance(at,int) or at<0):raise ValueError('invalid historical sequence')
        with self._mutex:return self._replay(at)[1]

    def publish_snapshot(self):
        with self._mutex,self.journal.transaction():
            snapshot=self.snapshot();target=self.journal.workdir/'current_state.json'
            temp=self.journal.workdir/('current_state.'+uuid.uuid4().hex+'.tmp')
            with temp.open('wb') as stream:
                stream.write(canonical(snapshot));stream.flush();os.fsync(stream.fileno())
            os.replace(temp,target);return snapshot

    def propose(self,kind,target,replacement,actor='human',event_id=None):
        if not isinstance(actor,str) or not actor or len(actor)>120:raise ValueError('invalid actor')
        if kind in {'premise','certificate','research_evidence'} and not isinstance(replacement,str):replacement=json.dumps(replacement,sort_keys=True)
        with self._mutex,self.journal.transaction():
            self.journal._load()
            if event_id:
                old=next((e for e in self.journal.events if e.get('event_id')==event_id),None)
                if old:
                    p=old['payload']
                    if old['type']!='PROPOSAL_CREATED' or (p['kind'],p['target'],p['replacement'],p['actor'])!=(kind,target,replacement,actor):raise ValueError('event ID reused with different proposal')
                    return copy.deepcopy(p)
            effective,s=self._replay();candidate=validate_edit(effective,kind,target,replacement)
            if kind=='research_evidence':
                from research_evidence import verify_capture
                capture=json.loads(replacement)['capture']
                if not any(c['revision_id']==capture['revision_id'] for c in s['research_discoveries']):raise ValueError('research capture is not current or registered')
                verify_capture(self.journal.workdir,capture)
            affected=downstream_closure(effective,[target]) if kind in {'certificate','research_evidence'} else impact_from(effective,candidate)['affected']
            primary={r['id'] for r in effective['records'] if r['project_acceptance']=='primary'}
            payload={'proposal_id':'p-'+uuid.uuid4().hex[:12],'base_effective_dataset_sha256':digest(effective),
                     'base_branch_sha256':s['branch_sha256'],'kind':kind,'target':target,'replacement':replacement,
                     'actor':actor,'preview_affected':affected,'preview_primary':sorted(primary.intersection(affected)),
                     'preview_count':len(affected),'source':'staged_untrusted_change','math_claim_status':'not_evaluated'}
            self.journal.append('PROPOSAL_CREATED',payload,event_id=event_id)
            self.publish_snapshot();return payload

    def decide(self,pid,approve,actor='human:local',authorized=True,event_id=None):
        if not authorized or not isinstance(actor,str) or not actor.startswith(('human','owner')):raise ValueError('unauthorized approval authority')
        with self._mutex,self.journal.transaction():
            effective,state=self._replay()
            if event_id:
                old=next((e for e in self.journal.events if e.get('event_id')==event_id),None)
                if old:
                    if old['type']!=('PROPOSAL_APPROVED' if approve else 'PROPOSAL_REJECTED') or old['payload']['proposal_id']!=pid:raise ValueError('event ID reused with different decision')
                    if approve:self.handle_event(old)
                    return self.publish_snapshot()
            props={p['proposal_id']:p for p in state['proposals']}
            if pid not in props or props[pid]['status']!='awaiting_approval':raise ValueError('proposal missing or already decided')
            proposal=props[pid]
            if approve:
                if proposal['base_effective_dataset_sha256']!=digest(effective) or proposal.get('base_branch_sha256')!=state['branch_sha256']:raise ValueError('stale proposal: working branch changed; restage against current state')
                candidate=validate_edit(effective,proposal['kind'],proposal['target'],proposal['replacement'])
                attachments=copy.deepcopy(state['certificate_attachments'])
                if proposal['kind']=='research_evidence':
                    from research_evidence import attach_research,verify_capture
                    value=json.loads(proposal['replacement']);capture=value['capture']
                    if proposal.get('ui_status')=='stale':raise ValueError('stale research evidence')
                    verify_capture(self.journal.workdir,capture)
                    attach_research(attachments,proposal['target'],value)
                elif proposal['kind']=='certificate':attachments.setdefault(proposal['target'],[]).append(json.loads(proposal['replacement']))
                before=mathematical_fingerprints(effective,state['certificate_attachments']);after=mathematical_fingerprints(candidate,attachments)
                affected=sorted(k for k in before if before[k]!=after[k])
                from workers import make_job
                trigger={'event_id':'approval:'+pid,'proposal_id':pid,'sequence':state['last_sequence']+1}
                jobs=[]
                for rid in affected:
                    job=make_job(candidate,rid,trigger)
                    job['mathematical_fingerprint']=after[rid];job['baseline_dataset_sha256']=digest(self.base)
                    job['certificate_input_sha256']=digest(attachments)
                    from research_evidence import relevant_captures
                    job['research_inputs']=relevant_captures(candidate,attachments,rid)
                    jobs.append(job)
                payload={'proposal_id':pid,'approved_by':actor,'baseline_dataset_sha256':digest(self.base),
                         'changed_input_sha256':digest({'kind':proposal['kind'],'target':proposal['target'],'replacement':proposal['replacement']}),
                         'affected':affected,'affected_hyperedges':['OBL:'+rid for rid in affected],
                         'fingerprints':{k:after[k] for k in affected},'jobs':jobs,'max_attempts':self.max_attempts}
                event=self.journal.append('PROPOSAL_APPROVED',payload,event_id=event_id)
                self.handle_event(event)
                self.tasks.put(event['seq'])
            else:self.journal.append('PROPOSAL_REJECTED',{'proposal_id':pid,'rejected_by':actor},event_id=event_id)
            return self.publish_snapshot()

    def handle_event(self,event):
        """Idempotent local delivery of an event already committed to this journal."""
        with self._mutex,self.journal.transaction():
            self.journal._load()
            existing=next((e for e in self.journal.events if e['seq']==event.get('seq')),None)
            if existing!=event:raise ValueError('event is not a verified member of this journal')
            if event['type']=='PROPOSAL_APPROVED':
                effective,s=self._replay(event['seq'])
                for job in s['jobs']:
                    if job['approval_sequence']==event['seq']:
                        self.journal.append('JOB_QUEUED',{'job_id':job['job_id'],'triggering_sequence':event['seq']},event_id='queue:'+job['job_id'])
                pid=event['payload']['proposal_id']
                proposal=next(p for p in s['proposals'] if p['proposal_id']==pid)
                if proposal['kind']=='certificate':
                    evaluation_id='certificate-evaluation:'+pid
                    if not any(e.get('event_id')==evaluation_id for e in self.journal.events):
                        from certificates import validate_certificate
                        try:verdict=validate_certificate(json.loads(proposal['replacement']),effective,self.journal.workdir)
                        except Exception as err:
                            verdict={'status':'blocked','formal_status':'not_certified','theorem_certificates_awarded':0,
                                     'kernel_execution_performed':False,'certificate_sha256':digest(json.loads(proposal['replacement'])),
                                     'reasons':[{'code':'GATE_EXECUTION_FAILED','message':type(err).__name__+': '+str(err)}]}
                        self.journal.append('CERTIFICATE_EVALUATED',{'proposal_id':pid,'target':proposal['target'],'verdict':verdict},event_id=evaluation_id)
            return self.snapshot()

    def recover_outbox(self):
        """Deliver unfinished approval effects once, including certificate gates."""
        with self._mutex,self.journal.transaction():
            _,state=self._replay()
            seen={e.get('event_id') for e in self.journal.events}
            props={p['proposal_id']:p for p in state['proposals']}
            approvals=[e for e in self.journal.events if e['type']=='PROPOSAL_APPROVED']
            recovered=0
            for event in approvals:
                pid=event['payload']['proposal_id']
                queue_ids=['queue:'+j['job_id'] for j in state['jobs'] if j['approval_sequence']==event['seq']]
                missing=any(eid not in seen for eid in queue_ids)
                if props[pid]['kind']=='certificate' and 'certificate-evaluation:'+pid not in seen:missing=True
                if missing:self.handle_event(event);recovered+=1
            return recovered

    def recover_jobs(self,now=None):
        now=time.time() if now is None else now;count=0
        with self._mutex,self.journal.transaction():
            _,s=self._replay()
            for job in s['jobs']:
                if job['state']=='running' and job['lease_expires_at']<=now:
                    self._retry_or_escalate(job,'worker lease expired after crash/restart',now=now)
                    count+=1
            if count:self.publish_snapshot()
        return count

    def _retry_or_escalate(self,job,reason,receipt=None,now=None):
        now=time.time() if now is None else now
        payload={'job_id':job['job_id'],'lease_token':job['lease_token'],'reason':str(reason),'receipt':receipt}
        if job['attempts']>=job['max_attempts']:
            self.journal.append('JOB_ESCALATED',payload)
        else:
            payload['next_attempt_at']=now+min(60,self.retry_backoff*(2**(job['attempts']-1)))
            self.journal.append('JOB_RETRY_SCHEDULED',payload)

    def claim_job(self,worker_id='manual',now=None):
        if not isinstance(worker_id,str) or not worker_id or len(worker_id)>120:raise ValueError('invalid worker identity')
        now=time.time() if now is None else now
        with self._mutex,self.journal.transaction():
            self.recover_jobs(now);effective,s=self._replay()
            if sum(j['state']=='running' for j in s['jobs'])>=self.max_workers:return None
            fps=mathematical_fingerprints(effective,s['certificate_attachments'])
            for job in s['jobs']:
                if job['state'] not in {'queued','retry_wait'} or job['next_attempt_at']>now:continue
                if job['mathematical_fingerprint']!=fps[job['record_id']]:
                    self.journal.append('JOB_STALE',{'job_id':job['job_id'],'reason':'mathematical inputs changed before dispatch'});continue
                token=uuid.uuid4().hex
                payload={'job_id':job['job_id'],'attempt':job['attempts']+1,'lease_token':token,
                         'lease_expires_at':now+self.lease_seconds,'worker_id':worker_id}
                self.journal.append('JOB_LEASED',payload)
                self.publish_snapshot()
                return next(j for j in self.snapshot()['jobs'] if j['job_id']==job['job_id'])
            return None

    def complete_job(self,job_id,lease_token,receipt):
        with self._mutex,self.journal.transaction():
            effective,s=self._replay();job=next((j for j in s['jobs'] if j['job_id']==job_id),None)
            if job is None:raise ValueError('unknown job')
            reason=None
            if job['state']!='running' or job['lease_token']!=lease_token:reason='late result or mismatched worker lease'
            elif job['lease_expires_at']<=time.time():reason='worker lease expired before completion'
            elif job['mathematical_fingerprint']!=mathematical_fingerprints(effective,s['certificate_attachments'])[job['record_id']]:reason='stale mathematical input fingerprint'
            if reason:
                self.journal.append('JOB_RESULT_REJECTED',{'job_id':job_id,'reason':reason,'lease_token':lease_token})
                if job['state']=='running' and job['lease_token']==lease_token:
                    if reason=='worker lease expired before completion':self._retry_or_escalate(job,reason)
                    else:self.journal.append('JOB_STALE',{'job_id':job_id,'reason':reason})
                self.publish_snapshot();raise ValueError(reason)
            from workers import verify_receipt
            approved_data,_=self._replay(job['approval_sequence'])
            expected_dir=self.journal.workdir/'artifacts'/job_id/lease_token
            try:
                artifact_dir=receipt.get('artifact_directory')
                if not artifact_dir or Path(artifact_dir).resolve()!=expected_dir.resolve():raise ValueError('receipt artifact directory mismatch')
                verify_receipt(receipt,job=job,data=approved_data,workdir=expected_dir)
            except (ValueError,KeyError,TypeError) as err:
                self.journal.append('JOB_RESULT_REJECTED',{'job_id':job_id,'lease_token':lease_token,'reason':'invalid receipt: '+str(err)})
                self._retry_or_escalate(job,'invalid receipt: '+str(err))
                self.publish_snapshot();raise ValueError('invalid worker receipt: '+str(err))
            result=receipt['result']
            if result=='failed' and receipt.get('retryable'):
                self._retry_or_escalate(job,'transient worker failure',receipt)
            elif result=='failed':
                self.journal.append('JOB_ESCALATED',{'job_id':job_id,'lease_token':lease_token,
                                    'reason':'persistent or nonretryable worker failure','receipt':receipt})
            else:
                self.journal.append('JOB_FINISHED',{'job_id':job_id,'lease_token':lease_token,'result':result,
                                    'receipt':receipt,'artifact_directory':str(expected_dir)})
            return self.publish_snapshot()

    def run_one_job(self,worker_id='manual'):
        job=self.claim_job(worker_id)
        if job is None:return None
        approved_data,_=self._replay(job['approval_sequence'])
        artifact_dir=self.journal.workdir/'artifacts'/job['job_id']/job['lease_token']
        from workers import execute_job
        try:
            receipt=execute_job(job,approved_data,artifact_dir,research_root=self.journal.workdir)
            self.complete_job(job['job_id'],job['lease_token'],receipt)
            return receipt
        except Exception as err:
            with self._mutex,self.journal.transaction():
                _,s=self._replay();latest=next(j for j in s['jobs'] if j['job_id']==job['job_id'])
                if latest['state']=='running' and latest['lease_token']==job['lease_token']:
                    self._retry_or_escalate(latest,type(err).__name__+': '+str(err))
                    self.publish_snapshot()
            return {'job_id':job['job_id'],'result':'failed','retryable':True,'error':str(err),'formal_certification':'not_certified'}

    def run_jobs(self):
        receipts=[]
        # Finite drain: retries have bounded attempts and short capped backoff.
        while True:
            if self._stop.is_set():break
            receipt=self.run_one_job('worker:'+threading.current_thread().name)
            if receipt is not None:receipts.append(receipt);continue
            s=self.snapshot();waiting=[j for j in s['jobs'] if j['state']=='retry_wait']
            if waiting:
                wait=max(0,min(j['next_attempt_at'] for j in waiting)-time.time())
                if wait>60:break
                self._stop.wait(min(wait,.1));continue
            break
        return receipts

    def run_checks(self):
        """Run actual bounded local workers; diagnostics cannot discharge proofs."""
        self.recover_outbox()
        self.run_jobs()
        with self._mutex,self.journal.transaction():
            effective,s=self._replay();validate(effective,require_stored_fingerprints=False)
            current=[j for j in s['jobs'] if j['job_id'] in set(s['required_job_ids'].values())]
            good=all(j['state']=='diagnostic_completed' for j in current)
            note={'schema_agent':'passed_working_branch_structural_checks',
                  'dependency_agent':{'affected_count':len(s['changed_ids']),'primary_affected':s['changed_primary']},
                  'proof_diagnostic_agent':diagnostic(effective,s['changed_ids']),
                  'formal_proof_checker':'not_installed_not_executed','theorem_certificates_awarded':0,
                  'result':'RECALCULATED_STILL_NEEDS_MATH_VERIFICATION' if good else 'CHECK_ERROR',
                  'worker_results':{j['job_id']:j['state'] for j in current},
                  'effective_dataset_sha256':s['effective_dataset_sha256'],
                  'last_approved_sequence':max((j['approval_sequence'] for j in s['jobs']),default=0)}
            self.journal.append('CHECK_RUN',note);self.publish_snapshot();return note

    def _worker(self):
        # An OSError raised before any journal bytes land (EMFILE on a lock, a failed read or snapshot write)
        # leaves the journal and leases authoritative: back off, resume, and let recover_jobs() expire any lease
        # that was left. A torn append still fails closed, because _load raises ValueError for it. Integrity
        # ValueErrors end the worker, as before.
        delay=.2
        while not self._stop.is_set():
            try:
                try:msg=self.tasks.get(timeout=.2)
                except queue.Empty:
                    self.recover_jobs()
                    if self.snapshot()['pending_job_count']:self.tasks.put('durable-recovery')
                    delay=.2;continue
                try:self.run_checks()
                except Exception as err:
                    self.journal.append('CHECK_RUN',{'result':'CHECK_ERROR','message':str(err),'theorem_certificates_awarded':0})
                finally:self.tasks.task_done()
                delay=.2
            except OSError as err:
                if delay==.2:print('UL worker: transient',repr(err),'- backing off',file=sys.stderr)
                self._stop.wait(delay);delay=min(delay*2,5.0)

    def _watch_once(self):
        for f in sorted(self.inbox.glob('*.json')):
            try:
                if f.stat().st_size>20000:raise ValueError('inbox proposal too large')
                d=json.loads(f.read_text(encoding='utf8'))
                self.propose(d['kind'],d['target'],d['replacement'],d.get('actor','agent:inbox'),event_id='inbox:'+hashlib.sha256(f.read_bytes()).hexdigest())
                dest=f.with_suffix('.staged')
            except Exception as e:
                dest=f.with_suffix('.rejected');dest.with_suffix('.error.txt').write_text(str(e),encoding='utf8')
            os.replace(f,dest)

    def _watch(self):
        while not self._stop.wait(.5):self._watch_once()

    def start(self):
        if self.started:return
        service_lock=(self.journal.workdir/'.service.lock').open('a+b')
        try:
            import fcntl
            fcntl.flock(service_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            service_lock.close();raise ValueError('another coordinator service owns this work directory')
        except ImportError:pass
        self._service_lock=service_lock;self.started=True;self._stop.clear()
        try:
            self.recover_jobs();self.recover_outbox()
        except Exception:
            self.started=False;self._service_lock.close();self._service_lock=None;raise
        self.threads=[threading.Thread(target=self._worker,daemon=True,name='UL-'+str(i)) for i in range(self.max_workers)]
        for thread in self.threads:thread.start()
        self.thread=self.threads[0]
        threading.Thread(target=self._watch,daemon=True).start()
        s=self.publish_snapshot()
        if s['approvals_pending_diagnostics']:self.tasks.put('replay')

    def stop(self):
        self._stop.set()
        deadline=time.monotonic()+20
        for thread in self.threads:thread.join(timeout=max(0,deadline-time.monotonic()))
        alive=any(thread.is_alive() for thread in self.threads)
        self.started=alive
        if self._service_lock and not alive:self._service_lock.close();self._service_lock=None

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,server_address,handler,coord):
        self.coord=coord;self.token=secrets.token_urlsafe(28)
        super().__init__(server_address,handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def _loopback_request(self):
        host=self.headers.get('Host','').lower()
        valid_hosts={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if host not in valid_hosts:return False
        origin=self.headers.get('Origin')
        if origin and origin.lower() not in {f'http://{h}' for h in valid_hosts}:return False
        return True
    def write_json(self,obj,status=200):
        raw=json.dumps(obj,ensure_ascii=False,sort_keys=True).encode('utf8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)))
        try:self.end_headers();self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if not self._loopback_request():
            self.write_json({'error':'loopback origin required'},403);return
        path=urlsplit(self.path)
        if path.path in ('/','/conductor.html'):
            html=(ROOT/'conductor_template.html').read_text(encoding='utf8')
            bootstrap=json.dumps(self.server.coord.snapshot(),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
            html=html.replace('%%BOOTSTRAP%%',bootstrap).replace('%%LOCAL_TOKEN%%',self.server.token)
            raw=html.encode('utf8')
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw);return
        if path.path=='/api/research/artifact':
            from research_evidence import SHA,verify_capture
            sha=parse_qs(path.query).get('sha',[''])[0]
            if not SHA.fullmatch(sha):self.write_json({'error':'invalid research hash'},400);return
            captures=[e['payload'] for e in self.server.coord.journal.since(0) if e['type']=='RESEARCH_CAPTURED' and e['payload']['sha256']==sha]
            if not captures:self.write_json({'error':'unknown research object'},404);return
            try:raw=verify_capture(self.server.coord.journal.workdir,captures[-1])
            except (ValueError,OSError) as err:self.write_json({'error':str(err)},409);return
            self.send_response(200);self.send_header('Content-Type','application/octet-stream')
            self.send_header('Content-Disposition','attachment; filename="'+sha+'.bin"')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        if path.path=='/api/mermaid':
            rid=parse_qs(path.query).get('id',['D1-A'])[0]
            try:text=mermaid_source(self.server.coord.snapshot(),rid)
            except ValueError as e:self.write_json({'error':str(e)},404);return
            raw=text.encode('utf8')
            self.send_response(200);self.send_header('Content-Type','text/plain; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw);return
        if path.path=='/api/state':
            try:
                qs=parse_qs(path.query);at=int(qs['at'][0]) if 'at' in qs else None
                self.write_json(self.server.coord.snapshot(at))
            except ValueError as err:self.write_json({'error':str(err)},400)
            return
        if path.path=='/api/events':
            try:
                since=int(parse_qs(path.query).get('since',[self.headers.get('Last-Event-ID','0')])[0])
                if since<0:raise ValueError('negative event cursor')
            except ValueError:self.write_json({'error':'invalid event cursor'},400);return
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Cache-Control','no-cache');self.send_header('X-Accel-Buffering','no');self.end_headers()
            try:
                while True:
                    with self.server.coord.journal._cond:
                        new=self.server.coord.journal.since(since)
                        if not new:self.server.coord.journal._cond.wait(timeout=10)
                        new=self.server.coord.journal.since(since)
                    if not new:
                        self.wfile.write(b': keep-alive\n\n');self.wfile.flush();continue
                    for e in new:
                        self.wfile.write(('id: %s\nevent: update\ndata: %s\n\n'%(e['seq'],json.dumps(e,ensure_ascii=False))).encode('utf8'))
                        since=e['seq']
                    self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,TimeoutError):return
        else:self.write_json({'error':'not found'},404)
    def do_POST(self):
        if not self._loopback_request():
            self.write_json({'error':'loopback origin required'},403);return
        if self.headers.get('X-UL-Token')!=self.server.token:
            self.write_json({'error':'missing/invalid local write token'},403);return
        try:n=int(self.headers.get('Content-Length','0'))
        except ValueError:self.write_json({'error':'invalid body size'},400);return
        if n<1 or n>30000:self.write_json({'error':'invalid body size'},400);return
        try:
            payload=json.loads(self.rfile.read(n))
            if not isinstance(payload,dict):raise ValueError('request must be a JSON object')
            path=urlsplit(self.path).path
            if path=='/api/propose':
                result=self.server.coord.propose(payload['kind'],payload['target'],payload['replacement'],payload.get('actor','human:ui'),event_id=payload.get('event_id'))
            elif path in ('/api/approve','/api/reject'):
                result=self.server.coord.decide(payload['proposal_id'],path.endswith('approve'),actor=payload.get('actor','human:ui'),event_id=payload.get('event_id'))
            elif path=='/api/check':
                result=self.server.coord.run_checks()
            else:self.write_json({'error':'not found'},404);return
            self.write_json(result)
        except (ValueError,KeyError,GateError,json.JSONDecodeError) as e:self.write_json({'error':str(e)},400)


def main():
    p=argparse.ArgumentParser(description='Offline reactive UL proof-conductor; no theorem certificate claims')
    p.add_argument('--workdir',default=str(ROOT/'runtime'),help='local mutable event journal directory')
    sub=p.add_subparsers(dest='cmd',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8765)
    prop=sub.add_parser('propose');prop.add_argument('--kind',required=True,choices=sorted(ALLOWED));prop.add_argument('--id',required=True);prop.add_argument('--value',required=True);prop.add_argument('--actor',default='human:cli')
    approve=sub.add_parser('approve');approve.add_argument('proposal_id')
    reject=sub.add_parser('reject');reject.add_argument('proposal_id')
    snap=sub.add_parser('state');snap.add_argument('--at',type=int)
    sub.add_parser('check')
    demo=sub.add_parser('demo');demo.add_argument('--kind',default='statement',choices=sorted(ALLOWED));demo.add_argument('--id',default='D1-A')
    args=p.parse_args()
    if args.cmd=='demo':
        with tempfile.TemporaryDirectory() as t:
            c=Coordinator(t)
            obj=find_item(c.base,args.kind,args.id)
            if args.kind=='certificate':
                value={'record_id':args.id,'kind':'unvalidated_demo_attachment','explanation':'An illustrative attachment is not a theorem certificate.'}
            elif args.kind=='premise':
                if not obj['premises']:raise ValueError('choose a record with an existing premise for this demo')
                value=None
                for record in c.base['records']:
                    if record['id'] in obj['premises'] or record['id']==args.id:continue
                    candidate=[record['id'],*obj['premises'][1:]]
                    try:validate_edit(c.base,'premise',args.id,candidate)
                    except (ValueError,GateError):continue
                    value=candidate;break
                if value is None:raise ValueError('no admissible illustrative premise replacement')
            else:
                field={'statement':'statement','side_condition':'side_conditions','import':'scope_description','correction':'role','local_evidence':'limits'}[args.kind]
                old=obj[field][0]['text'] if args.kind=='side_condition' else obj[field]
                value=old+' [DEMO PROPOSED CHANGE]'
            proposal=c.propose(args.kind,args.id,value)
            print('staged:',proposal['proposal_id'],'preview:',proposal['preview_affected'])
            c.decide(proposal['proposal_id'],True)
            print(json.dumps(c.run_checks(),indent=2)[:8000])
            print('baseline unchanged; runtime temporary');return
    c=Coordinator(args.workdir)
    if args.cmd=='propose':print(json.dumps(c.propose(args.kind,args.id,args.value,args.actor),indent=2))
    elif args.cmd in ('approve','reject'):
        c.decide(args.proposal_id,args.cmd=='approve');
        if args.cmd=='approve':c.run_checks()
        print(json.dumps(c.snapshot(),indent=2)[:3000])
    elif args.cmd=='state':print(json.dumps(c.snapshot(args.at),indent=2))
    elif args.cmd=='check':print(json.dumps(c.run_checks(),indent=2))
    elif args.cmd=='serve':
        c.start();http=Server(('127.0.0.1',args.port),Handler,c)
        print('UL Reactive Conductor: http://127.0.0.1:%d/'%args.port,flush=True)
        print('Local-only, no external AI provider, no autonomous theorem certificates',flush=True)
        try:http.serve_forever()
        except KeyboardInterrupt:pass
        finally:c.stop();http.server_close()

if __name__=='__main__':
    try:main()
    except (GateError,ValueError,KeyError) as e:
        print('ERROR:',e,file=sys.stderr);sys.exit(1)
