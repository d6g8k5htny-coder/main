import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import * as m from '../docs/site/current-model.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const recordsDir = fileURLToPath(new URL('../records/node_status/', import.meta.url));
const FORBIDDEN = ['lemma_closed', 'prizes_solved', 'discharges_OBL_H5_JETMOD', 'certified_C_H', 'freeze', 'inventable_attempt_accepted'];
const PATTERN = String.raw`^https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/[1-9]\d*#(issuecomment-[1-9]\d*|pullrequestreview-[1-9]\d*)$`;
const DAY = 24 * 60 * 60 * 1000;
const PICKUP = 'https://github.com/d6g8k5htny-coder/Math-/pull/463#issuecomment-6198606495';
const RELEASED = 'https://github.com/d6g8k5htny-coder/Math-/pull/463#pullrequestreview-5537612171';
const COMMENT = 'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346301123';
const commit = 'a'.repeat(40), main = 'c'.repeat(40), digest = 'b'.repeat(64);
const snapshot = nodes => ({schema_version: 1, observed_at: '2026-10-13T12:00:00Z', scientific_effect: 'NONE', repositories: [{repository: 'd6g8k5htny-coder/main', commit: main, tree: main}, {repository: 'd6g8k5htny-coder/Math-', commit, tree: commit}], sources: [], graph: {identity: {repository: 'd6g8k5htny-coder/Math-', commit, path: 'frontiers/x/GRAPH.json', blob: commit, bytes: 20, sha256: digest}, data: {nodes, edges: []}}, equations: [], references: []});

function python(args, options = {}) {
  try {
    const stdout = execFileSync('python3', ['-B', '-S', ...args], {encoding: 'utf8', cwd: root, stdio: ['pipe', 'pipe', 'pipe'], ...options});
    return {status: 0, stdout, stderr: ''};
  } catch (error) {
    return {status: error.status ?? 1, stdout: String(error.stdout ?? ''), stderr: String(error.stderr ?? '')};
  }
}

function pythonJson(args, options) {
  const result = python(args, options);
  if (!result.stdout) throw new Error(`python ${args[0]} failed (${result.status}): ${result.stderr}`);
  return JSON.parse(result.stdout);
}

async function records() {
  const names = (await readdir(recordsDir)).filter(name => name.endsWith('.json')).sort();
  const entries = await Promise.all(names.map(async name => [name.slice(0, -5), JSON.parse(await readFile(join(recordsDir, name), 'utf8'))]));
  return Object.fromEntries(entries);
}

function flags(value, found = []) {
  if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) {
    if (FORBIDDEN.includes(key)) found.push(key);
    flags(item, found);
  }
  return found;
}

async function validate(record) {
  const directory = await mkdtemp(join(tmpdir(), 'node-status-'));
  const name = `${record.node_id}.json`;
  const path = join(directory, name);
  await writeFile(path, JSON.stringify(record));
  const result = python(['tools/validate_node_status.py', directory]);
  const saved = JSON.parse(await readFile(path, 'utf8'));
  await rm(directory, {recursive: true, force: true});
  return {...result, saved};
}

function ownerRow(view) {
  return view.operational.find(row => row.field === 'owner_evidence');
}

function noteOf(view) {
  return view.operational.find(row => row.field === 'next_action').note;
}

test('published node-status records are real graph nodes with no owner and no mathematical flags', async () => {
  const view = await records();
  const ids = Object.keys(view);
  assert.ok(ids.length >= 2 && ids.length <= 3, ids.join(','));
  const schema = JSON.parse(await readFile(new URL('../architecture/node-status.schema.json', import.meta.url), 'utf8'));
  assert.deepEqual(schema.required, ['schema_version', 'node_id', 'owner', 'next_action', 'missing_input', 'stops', 'sources', 'source_rev', 'observed_at', 'status', 'scientific_effect']);
  for (const name of ['predecessor', 'successor', 'carried_forward', 'refresh_error']) {
    assert.equal(schema.required.includes(name), false, name);
    assert.ok(schema.properties[name], name);
  }
  assert.equal(schema.additionalProperties, false);
  assert.deepEqual(schema.properties.status.enum, ['fresh', 'stale', 'unknown']);
  assert.equal(schema.properties.scientific_effect.const, 'NONE');
  const ownerObject = schema.properties.owner.oneOf.find(item => item.type === 'object');
  assert.equal(ownerObject.type, 'object');
  assert.equal(ownerObject.additionalProperties, false);
  assert.deepEqual(ownerObject.required, ['id', 'evidence_url']);
  assert.equal(ownerObject.properties.evidence_url.pattern, PATTERN);
  assert.equal(ownerObject.properties.released_url.pattern, PATTERN);
  assert.equal(schema.properties.carried_forward.items.properties.resolution_url.pattern, PATTERN);
  assert.equal(m.NODE_STATUS_EVIDENCE_URL_PATTERN, PATTERN);
  const printed = python(['-c', 'import sys; sys.path.insert(0, "tools"); from validate_node_status import EVIDENCE_URL_PATTERN; from current_site_history import EVIDENCE_URL_PATTERN as shared; assert shared == EVIDENCE_URL_PATTERN; print(EVIDENCE_URL_PATTERN)']);
  assert.equal(printed.status, 0, printed.stderr);
  assert.equal(printed.stdout.trim(), PATTERN);
  const graph = JSON.parse(await readFile(new URL('../docs/site/current.json', import.meta.url), 'utf8')).graph.data.nodes;
  const bundle = JSON.parse(await readFile(new URL('../docs/site/node-status.json', import.meta.url), 'utf8'));
  assert.deepEqual(bundle, view);
  assert.deepEqual(Object.keys(bundle), [...ids].sort());
  for (const [id, record] of Object.entries(view)) {
    assert.equal(record.node_id, id);
    assert.equal(record.schema_version, 0);
    assert.equal(record.owner, null);
    assert.equal(record.scientific_effect, 'NONE');
    assert.ok(['fresh', 'stale', 'unknown'].includes(record.status));
    assert.equal(record.source_rev, record.sources[0].rev);
    assert.equal(flags(record).length, 0);
    assert.ok(Object.hasOwn(graph, id), id);
  }
  const checked = python(['tools/validate_node_status.py', 'records/node_status']);
  assert.equal(checked.status, 0, checked.stderr);
  assert.match(checked.stdout, /node-status records valid/);
  assert.doesNotMatch(checked.stdout + checked.stderr, /skip/i);
});

test('a record with an owner but no evidence_url fails validation', async () => {
  const base = (await records())['eng.hard-gate'];
  const failed = await validate({...base, owner: {id: 'agent 9'}});
  assert.notEqual(failed.status, 0);
  assert.match(failed.stderr, /eng\.hard-gate\.json: owner requires evidence_url/);
  assert.doesNotMatch(failed.stderr, /Traceback/);
  const shaped = {...base, owner: {id: 'shape-only', evidence_url: COMMENT}};
  const passed = await validate(shaped);
  assert.equal(passed.status, 0, passed.stderr);
  const flagged = {...base, lemma_closed: true};
  const refused = await validate(flagged);
  assert.notEqual(refused.status, 0);
  assert.match(refused.stderr, /forbidden mathematical status flag: lemma_closed/);
  const nested = {...base, carried_forward: [{id: 'x', text: 'y', source_url: COMMENT, status: 'open', lemma_closed: true}]};
  const nestedRefused = await validate(nested);
  assert.match(nestedRefused.stderr, /forbidden mathematical status flag: lemma_closed/);
  const notDict = await validate({...base, owner: 'agent 9'});
  assert.match(notDict.stderr, /owner must be null or an object with evidence_url/);
  assert.doesNotMatch(notDict.stderr, /Traceback/);
  const widened = await validate({...base, status: 'current'});
  assert.match(widened.stderr, /status must be fresh, stale, or unknown/);
  const looseSha = {...base, sources: [{url: base.sources[0].url, rev: 'not-a-timestamp'}], source_rev: 'not-a-timestamp'};
  const sha = await validate(looseSha);
  assert.match(sha.stderr, /sources\[0\]\.rev must be a 40-character commit/);
});

test('closed schema, evidence pattern, and primary source_rev are enforced', async () => {
  const base = (await records())['eng.hard-gate'];
  const extra = await validate({...base, not_in_schema: 1});
  assert.notEqual(extra.status, 0);
  assert.match(extra.stderr, /unexpected fields: not_in_schema/);
  const script = await validate({...base, owner: {id: 'agent 9', evidence_url: 'javascript:alert(1)'}});
  assert.notEqual(script.status, 0);
  assert.match(script.stderr, /owner evidence_url does not match the comment pattern/);
  assert.doesNotMatch(script.stderr, /Traceback/);
  const zeros = [
    'https://github.com/d6g8k5htny-coder/main/issues/0#issuecomment-1',
    'https://github.com/d6g8k5htny-coder/main/issues/01#issuecomment-12',
    'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-0',
    'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-01',
    'https://github.com/d6g8k5htny-coder/main/pull/463#pullrequestreview-0',
    'https://github.com/d6g8k5htny-coder/Math-/pull/0463#issuecomment-6198606495',
    'https://github.com/d6g8k5htny-coder/main/issues/475#pullrequestreview-01',
  ];
  for (const url of zeros) {
    assert.equal(m.NODE_STATUS_EVIDENCE_URL.test(url), false, url);
    const rejected = await validate({...base, owner: {id: 'agent 9', evidence_url: url}});
    assert.notEqual(rejected.status, 0, url);
    assert.match(rejected.stderr, /owner evidence_url does not match the comment pattern/);
    const released = await validate({...base, owner: {id: 'agent 9', evidence_url: COMMENT, released_url: url}});
    assert.notEqual(released.status, 0, url);
    assert.match(released.stderr, /owner released_url does not match the comment pattern/);
  }
  const ten = 'https://github.com/d6g8k5htny-coder/main/issues/10#issuecomment-10';
  assert.equal(m.NODE_STATUS_EVIDENCE_URL.test(ten), true);
  assert.equal((await validate({...base, owner: {id: 'agent 9', evidence_url: ten}})).status, 0);
  const older = '2020-01-01T00:00:00Z';
  const newer = '2026-10-22T17:00:35Z';
  const pin = base.sources[0];
  const comment = {url: COMMENT, rev: newer};
  const other = {url: 'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346276856', rev: older};
  const primary = await validate({...base, sources: [{...other, rev: older}, comment], source_rev: older});
  assert.equal(primary.status, 0, primary.stderr);
  const notPrimary = await validate({...base, sources: [other, comment], source_rev: newer});
  assert.notEqual(notPrimary.status, 0);
  assert.match(notPrimary.stderr, /source_rev must equal sources\[0\]\.rev/);
  const pinPrimary = await validate({...base, sources: [pin, comment], source_rev: pin.rev});
  assert.equal(pinPrimary.status, 0, pinPrimary.stderr);
  const joined = await validate({...base, sources: [other, comment], source_rev: `${older},${newer}`});
  assert.notEqual(joined.status, 0);
  assert.match(joined.stderr, /source_rev must be a single rev/);
  assert.doesNotMatch(joined.stderr, /source_rev must equal sources\[0\]\.rev/);
});

test('carried_forward resolution_url uses the shared comment pattern', async () => {
  const base = (await records())['eng.hard-gate'];
  const item = (resolution_url, status = 'resolved') => ({
    ...base,
    carried_forward: [{
      id: 'pickup',
      text: 'A carried comment.',
      source_url: COMMENT,
      status,
      resolution_url,
    }],
  });
  const rejected = [
    'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-0',
    'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-01',
    'https://github.com/d6g8k5htny-coder/main/pull/463#pullrequestreview-0',
    'https://github.com/d6g8k5htny-coder/Math-/pull/0463#issuecomment-6198606495',
    'https://github.com/d6g8k5htny-coder/meta-framework/issues/475#issuecomment-12',
    'https://github.com/other-owner/main/issues/475#issuecomment-12',
  ];
  for (const url of rejected) {
    const failed = await validate(item(url));
    assert.notEqual(failed.status, 0, url);
    assert.match(failed.stderr, /carried_forward\[0\] resolution_url does not match the comment pattern/);
    assert.doesNotMatch(failed.stderr, /Traceback/);
  }
  for (const url of [COMMENT, RELEASED]) {
    const passed = await validate(item(url));
    assert.equal(passed.status, 0, `${url}\n${passed.stderr}`);
    assert.equal(passed.saved.carried_forward[0].resolution_url, url);
  }
  const open = await validate({
    ...base,
    carried_forward: [{id: 'pickup', text: 'Still open.', source_url: 'https://github.com/d6g8k5htny-coder/meta-framework/issues/475#issuecomment-0', status: 'open'}],
  });
  assert.equal(open.status, 0, open.stderr);
});

test('released_url keeps the owner object and the atlas says released', async () => {
  const fixture = JSON.parse(await readFile(new URL('./fixtures/node_status/math-463-released.json', import.meta.url), 'utf8'));
  assert.equal(fixture.owner.evidence_url, PICKUP);
  assert.equal(fixture.owner.released_url, RELEASED);
  assert.notEqual(fixture.owner, null);
  const checked = await validate(fixture);
  assert.equal(checked.status, 0, checked.stderr);
  assert.equal(typeof checked.saved.owner, 'object');
  assert.notEqual(checked.saved.owner, null);
  assert.equal(checked.saved.owner.id, 'agent-old');
  assert.equal(checked.saved.owner.evidence_url, PICKUP);
  assert.equal(checked.saved.owner.released_url, RELEASED);
  assert.equal(checked.saved.predecessor.node_id, 'board-old');
  assert.equal(checked.saved.successor, null);
  assert.equal(checked.saved.carried_forward[0].status, 'resolved');
  assert.equal(checked.saved.refresh_error, null);
  const data = snapshot({[fixture.node_id]: {kind: 'candidate', classification: 'OPEN_ACTIVE'}});
  const before = fixture.owner;
  const shown = m.resumeProjection(data, fixture.node_id, Date.parse(fixture.observed_at), {[fixture.node_id]: fixture});
  assert.equal(fixture.owner, before);
  assert.notEqual(fixture.owner, null);
  const row = ownerRow(shown);
  assert.equal(row.text, 'released');
  assert.equal(row.url, RELEASED);
  assert.equal(m.resumeFieldHref(row), RELEASED);
  assert.doesNotMatch(JSON.stringify(row), /current owner/);
  assert.doesNotMatch(JSON.stringify(row), /agent-old/);
  assert.equal(row.label, 'Owner evidence');
  assert.doesNotMatch(row.text, /No owner recorded/);
  const script = 'import json,sys\nsys.path.insert(0,sys.argv[1])\nfrom current_site_history import resume_projection\npayload=json.load(sys.stdin)\nprint(json.dumps(resume_projection(payload["data"], payload["id"], payload["records"], payload["now"])["operational"]))';
  const py = pythonJson(['-c', script, fileURLToPath(new URL('../tools/', import.meta.url))], {input: JSON.stringify({data, id: fixture.node_id, records: {[fixture.node_id]: fixture}, now: Date.parse(fixture.observed_at)})});
  assert.deepEqual(py, shown.operational);
});

test('resume projection shows a node-status view and refuses an owner without evidence', async () => {
  const view = await records();
  const nodes = Object.fromEntries(Object.keys(view).map(id => [id, {kind: 'candidate', classification: 'OPEN_ACTIVE', next_action: 'merge now', owner: 'not an owner'}]));
  nodes.other = {kind: 'candidate', classification: 'OPEN_ACTIVE'};
  const data = snapshot(nodes);
  for (const id of Object.keys(view)) {
    const shown = m.resumeProjection(data, id, undefined, view);
    assert.equal(shown.operational.find(row => row.field === 'next_action').text, view[id].next_action);
    assert.match(noteOf(shown), /Node-status view fresh as of 2026-10-22T17:00:35Z\. Recorded status fresh\./);
    assert.match(noteOf(shown), /Freshness is not scientific validity/);
    assert.equal(ownerRow(shown).text, 'No owner recorded in this node-status view.');
    assert.equal(ownerRow(shown).url, undefined);
    assert.equal(shown.scientific_effect, undefined);
    assert.doesNotMatch(JSON.stringify(shown.operational), /merge now|not an owner/);
    assert.deepEqual(flags(shown), []);
  }
  const bare = m.resumeProjection(data, 'other', undefined, view);
  assert.ok(bare.operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  const omitted = m.resumeProjection(data, Object.keys(view)[0]);
  assert.ok(omitted.operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  const id = Object.keys(view)[0];
  const ownerless = {[id]: {...view[id], owner: {id: 'agent 9'}}};
  const hidden = m.resumeProjection(data, id, undefined, ownerless);
  assert.ok(hidden.operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  assert.doesNotMatch(JSON.stringify(hidden), /agent 9/);
  const flagged = {[id]: {...view[id], lemma_closed: true}};
  assert.ok(m.resumeProjection(data, id, undefined, flagged).operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  const effect = {[id]: {...view[id], scientific_effect: 'SOME'}};
  assert.ok(m.resumeProjection(data, id, undefined, effect).operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  const hostile = {[id]: {...view[id], owner: {id: 'agent 9', evidence_url: 'javascript:alert(1)'}}};
  const plain = m.resumeProjection(data, id, undefined, hostile);
  assert.equal(ownerRow(plain).url, undefined);
  assert.equal(ownerRow(plain).text, 'javascript:alert(1)');
  assert.equal(m.resumeFieldHref({field: 'owner_evidence', text: 'javascript:alert(1)', url: 'javascript:alert(1)'}), null);
  assert.equal(m.nodeStatusEvidenceURL('javascript:alert(1)'), null);
  const observed = view[id].observed_at;
  const instant = Date.parse(observed);
  const boundary = m.resumeProjection(data, id, instant + DAY, view);
  assert.match(noteOf(boundary), /Node-status view fresh as of /);
  assert.doesNotMatch(noteOf(boundary), /Node-status view stale/);
  const past = m.resumeProjection(data, id, instant + DAY + 1, view);
  assert.match(noteOf(past), /Node-status view stale as of /);
  assert.match(noteOf(past), /Recorded status fresh/);
  const unknown = {[id]: {...view[id], status: 'unknown'}};
  const overridden = m.resumeProjection(data, id, instant + DAY + 1, unknown);
  assert.match(noteOf(overridden), /Node-status view stale as of /);
  assert.match(noteOf(overridden), /Recorded status unknown/);
  const tools = fileURLToPath(new URL('../tools/', import.meta.url));
  const script = 'import json,sys\nsys.path.insert(0,sys.argv[1])\nfrom current_site_history import resume_projection\npayload=json.load(sys.stdin)\nprint(json.dumps([resume_projection(payload["data"], i, payload["records"], payload.get("now")).get("operational") for i in payload["ids"]]))';
  const ids = [...Object.keys(view), 'other'];
  const py = pythonJson(['-c', script, tools], {input: JSON.stringify({data, ids, records: view})});
  assert.deepEqual(py, ids.map(node => m.resumeProjection(data, node, undefined, view).operational));
  const pyPast = pythonJson(['-c', script, tools], {input: JSON.stringify({data, ids: [id], records: view, now: instant + DAY + 1})});
  assert.deepEqual(pyPast, [m.resumeProjection(data, id, instant + DAY + 1, view).operational]);
  const pyEdge = pythonJson(['-c', script, tools], {input: JSON.stringify({data, ids: [id], records: view, now: instant + DAY})});
  assert.deepEqual(pyEdge, [m.resumeProjection(data, id, instant + DAY, view).operational]);
  const pyScript = pythonJson(['-c', script, tools], {input: JSON.stringify({data, ids: [id], records: hostile})});
  assert.deepEqual(pyScript, [plain.operational]);
  const pyOwner = pythonJson(['-c', script, tools], {input: JSON.stringify({data, ids: [id], records: ownerless})});
  assert.deepEqual(pyOwner, [hidden.operational]);
  const pyFlag = pythonJson(['-c', script, tools], {input: JSON.stringify({data, ids: [id], records: flagged})});
  assert.deepEqual(pyFlag, [m.resumeProjection(data, id, undefined, flagged).operational]);
});

test('the atlas resume section passes the published node-status view into resumeProjection', async () => {
  const source = await readFile(new URL('../docs/site/atlas.mjs', import.meta.url), 'utf8');
  assert.match(source, /fetch\('node-status\.json'/);
  assert.match(source, /resumeProjection\(data,selected,Date\.now\(\),nodeStatus\)/);
  assert.match(source, /function resumeSection\(view\)/);
  assert.match(source, /resumeFieldHref\(item\)/);
  assert.match(source, /import \{[^}]*resumeFieldHref\} from '\.\/current-model\.mjs/);
});
