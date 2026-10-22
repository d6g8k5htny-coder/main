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

test('published node-status records are real graph nodes with no owner and no mathematical flags', async () => {
  const view = await records();
  const ids = Object.keys(view);
  assert.ok(ids.length >= 2 && ids.length <= 3, ids.join(','));
  const schema = JSON.parse(await readFile(new URL('../architecture/node-status.schema.json', import.meta.url), 'utf8'));
  assert.deepEqual(schema.required, ['schema_version', 'node_id', 'owner', 'next_action', 'missing_input', 'stops', 'sources', 'source_rev', 'observed_at', 'status', 'scientific_effect']);
  assert.equal(schema.additionalProperties, false);
  assert.deepEqual(schema.properties.status.enum, ['fresh', 'stale', 'unknown']);
  assert.equal(schema.properties.scientific_effect.const, 'NONE');
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
  const directory = await mkdtemp(join(tmpdir(), 'node-status-'));
  const bad = {...base, owner: {id: 'agent 9'}};
  await writeFile(join(directory, 'eng.hard-gate.json'), JSON.stringify(bad));
  const failed = python(['tools/validate_node_status.py', directory]);
  assert.notEqual(failed.status, 0);
  assert.match(failed.stderr, /evidence_url/);
  const shaped = {...base, owner: {id: 'shape-only', evidence_url: 'https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346301123'}};
  const okDir = await mkdtemp(join(tmpdir(), 'node-status-ok-'));
  await writeFile(join(okDir, 'eng.hard-gate.json'), JSON.stringify(shaped));
  const passed = python(['tools/validate_node_status.py', okDir]);
  assert.equal(passed.status, 0, passed.stderr);
  const flagged = {...base, lemma_closed: true};
  const flagDir = await mkdtemp(join(tmpdir(), 'node-status-flag-'));
  await writeFile(join(flagDir, 'eng.hard-gate.json'), JSON.stringify(flagged));
  const refused = python(['tools/validate_node_status.py', flagDir]);
  assert.notEqual(refused.status, 0);
  assert.match(refused.stderr, /lemma_closed/);
  await rm(directory, {recursive: true});
  await rm(okDir, {recursive: true});
  await rm(flagDir, {recursive: true});
});

test('resume projection shows a node-status view and refuses an owner without evidence', async () => {
  const view = await records();
  const nodes = Object.fromEntries(Object.keys(view).map(id => [id, {kind: 'candidate', classification: 'OPEN_ACTIVE', next_action: 'merge now', owner: 'not an owner'}]));
  nodes.other = {kind: 'candidate', classification: 'OPEN_ACTIVE'};
  const data = snapshot(nodes);
  for (const id of Object.keys(view)) {
    const shown = m.resumeProjection(data, id, undefined, view);
    assert.equal(shown.operational.find(row => row.field === 'next_action').text, view[id].next_action);
    assert.match(shown.operational.find(row => row.field === 'next_action').note, /Freshness is not scientific validity/);
    assert.equal(shown.operational.find(row => row.field === 'owner_evidence').text, 'No owner recorded in this node-status view.');
    assert.equal(shown.operational.find(row => row.field === 'owner_evidence').url, undefined);
    assert.equal(shown.scientific_effect, undefined);
    assert.doesNotMatch(JSON.stringify(shown.operational), /merge now|not an owner/);
    assert.deepEqual(flags(shown), []);
  }
  const bare = m.resumeProjection(data, 'other', undefined, view);
  assert.ok(bare.operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  const omitted = m.resumeProjection(data, Object.keys(view)[0]);
  assert.ok(omitted.operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  const ownerless = {[Object.keys(view)[0]]: {...view[Object.keys(view)[0]], owner: {id: 'agent 9'}}};
  const hidden = m.resumeProjection(data, Object.keys(view)[0], undefined, ownerless);
  assert.ok(hidden.operational.every(row => row.text === m.RESUME_NOT_RECORDED));
  assert.doesNotMatch(JSON.stringify(hidden), /agent 9/);
  const tools = fileURLToPath(new URL('../tools/', import.meta.url));
  const script = 'import json,sys\nsys.path.insert(0,sys.argv[1])\nfrom current_site_history import resume_projection\npayload=json.load(sys.stdin)\nprint(json.dumps([resume_projection(payload["data"], i, payload["records"]).get("operational") for i in payload["ids"]]))';
  const ids = [...Object.keys(view), 'other'];
  const py = JSON.parse(python(['-c', script, tools], {input: JSON.stringify({data, ids, records: view})}).stdout);
  assert.deepEqual(py, ids.map(id => m.resumeProjection(data, id, undefined, view).operational));
});

test('the atlas resume section passes the published node-status view into resumeProjection', async () => {
  const source = await readFile(new URL('../docs/site/atlas.mjs', import.meta.url), 'utf8');
  assert.match(source, /fetch\('node-status\.json'/);
  assert.match(source, /resumeProjection\(data,selected,Date\.now\(\),nodeStatus\)/);
  assert.match(source, /function resumeSection\(view\)/);
});
