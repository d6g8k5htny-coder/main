// Source-only external controls. All NOT_RUN at author delivery.
// No production reducer, fake DOM, Canvas double, wire adapter or hidden input.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

const REPOSITORY = 'd6g8k5htny-coder/main';
const COMMIT = '77d35feff69c4df1b55a7449f3deaa68ed75de7d';
const PINS = Object.freeze([
  {path:'interface/src/scripts/cubic-model.mjs', bytes:1897,
    blob:'0cfd5acf6c031cfaf54320d292c4f2a8a07eca08',
    sha256:'fbff097a93429652ce5077056c400343b1472375a9cb1cbc388894c7e7bf2051'},
  {path:'interface/src/scripts/elder-model.mjs', bytes:5335,
    blob:'aab4a06a5ef0fdde609192d37169b578fcbcef19',
    sha256:'d92b8f6139d42360a35e933b40a287414b95811f14f0dc19c520d1d2558ec767'},
  {path:'docs/plans/2026-10-09-canvas-model-contract.md', bytes:13354,
    blob:'ab5315ab47014bc5ce2f89f432239ebff30998d5',
    sha256:'3473a664be9126b67006527f5257a7eeb4c2082c836d7ff1b38e21f0d40eecbc'},
]);
export const CONTROL_INVENTORY = Object.freeze([
  'D00 setup source pins and exports',
  'O01 independent partial-edge oracle literal calibration',
  'D01 signed zero and complete cubic outputs',
  'D02 unbounded native BigInt exact intervals',
  'D03 owned immutable input result and selection',
  'D04 native admission without coercion or replacement',
  'D05 insertion prefixes and complete tied level',
  'D06 independent periodic BFS selected state',
  'D07 strict navigation and whole-sweep preservation',
  'D08 staged atomic commit and stale tickets',
  'D09 complete escaped static HTML text',
  'D10 declared native provenance without false joins',
]);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}
function gitBlob(bytes) {
  return createHash('sha1').update(Buffer.from('blob ' + bytes.length + '\0'))
    .update(bytes).digest('hex');
}
function binding(label = 'Inline public native controls') {
  return {repository:REPOSITORY, modelCommit:COMMIT, cubicSha256:PINS[0].sha256,
    elderSha256:PINS[1].sha256, contractSha256:PINS[2].sha256, label,
    composition:null};
}
const urls = Object.fromEntries(PINS.map((p, i) =>
  [['cubic','elder','contract'][i],
    'https://github.com/' + REPOSITORY + '/blob/' + COMMIT + '/' + p.path]));

// Setup is deliberately separate: missing/broken exports are not behavior RED.
async function preflight() {
  const identities = [];
  try {
    for (const pin of PINS) {
      const bytes = await readFile(new URL('../../' + pin.path, import.meta.url));
      if (bytes.length !== pin.bytes || sha256(bytes) !== pin.sha256
          || gitBlob(bytes) !== pin.blob) {
        throw new Error('SOURCE_PIN_MISMATCH:' + pin.path);
      }
      identities.push({path:pin.path, bytes:bytes.length, sha256:sha256(bytes)});
    }
    const specifications = [
      ['../src/scripts/cubic-model.mjs', ['cubicModel']],
      ['../src/scripts/elder-model.mjs', ['finiteElderTrace']],
      ['../src/scripts/canvas-reader-state.mjs', ['createReaderController']],
      ['../src/scripts/canvas-reader-view.mjs', ['renderStaticReader','mountReader']],
    ];
    const modules = [];
    for (const [relative, names] of specifications) {
      const url = new URL(relative, import.meta.url);
      const bytes = await readFile(url);
      const module = await import(url.href);
      if (names.some(name => typeof module[name] !== 'function')) {
        throw new Error('WRONG_EXPORTS:' + relative);
      }
      identities.push({path:relative, bytes:bytes.length, sha256:sha256(bytes)});
      modules.push(module);
    }
    return {blocked:null, identities, state:modules[2], view:modules[3]};
  } catch (error) {
    const category = error instanceof SyntaxError ? 'SYNTAX'
      : error?.code === 'ENOENT' || error?.code === 'ERR_MODULE_NOT_FOUND' ? 'MISSING'
      : String(error?.message).startsWith('WRONG_EXPORTS:') ? 'WRONG_EXPORTS'
      : String(error?.message).startsWith('SOURCE_PIN_MISMATCH:') ? 'SOURCE_PIN_MISMATCH'
      : 'IMPORT_FAILED';
    return {blocked:category, identities};
  }
}
const setup = await preflight();
test(CONTROL_INVENTORY[0], t => {
  t.diagnostic(JSON.stringify({kind:'SOURCE_SETUP', identities:setup.identities,
    setup:setup.blocked ? 'SETUP_BLOCKED' : 'READY',
    behavior:setup.blocked ? 'NOT_RUN' : 'UNDETERMINED',
    inventory:CONTROL_INVENTORY, inventoryMeaning:'DECLARED_NOT_OBSERVED'}));
  assert.equal(setup.blocked, null,
    'SETUP_BLOCKED:' + setup.blocked + '; behavior NOT_RUN');
});
function behavior(index, fn) {
  test(CONTROL_INVENTORY[index], {
    skip:setup.blocked ? 'SETUP_BLOCKED:' + setup.blocked + '; behavior NOT_RUN' : false,
  }, fn);
}
function controller(label) {
  return setup.state.createReaderController(binding(label));
}
function exactKeys(value, names) {
  assert.deepEqual(Object.keys(value).sort(), [...names].sort());
}
function deeplyFrozen(value) {
  if (value === null || typeof value !== 'object') return;
  assert.equal(Object.isFrozen(value), true);
  for (const child of Object.values(value)) deeplyFrozen(child);
}
function refusal(value, code) {
  exactKeys(value, ['kind','code']);
  assert.deepEqual(value, {kind:'Refusal',code});
  deeplyFrozen(value);
}
function ready(value) {
  exactKeys(value, ['kind','ticket','snapshot']);
  assert.equal(value.kind, 'Ready');
  assert.equal(typeof value.ticket, 'object');
  assert.notEqual(value.ticket, null);
  assert.equal(Object.isFrozen(value.ticket), true);
  assert.equal(Object.isFrozen(value), true);
  const s = value.snapshot;
  exactKeys(s, ['kind','revision','documentRevision','model','result','selection',
    'selected','navigation','source']);
  assert.equal(s.kind, 'Snapshot');
  assert.equal(typeof s.revision, 'bigint');
  assert.equal(typeof s.documentRevision, 'bigint');
  deeplyFrozen(s);
  return value;
}
function install(c, document) {
  const old = c.current();
  const r = ready(c.prepareDocument(document));
  assert.strictEqual(c.current(), old, 'prepare must not publish');
  assert.strictEqual(c.commit(r.ticket), r.snapshot);
  assert.strictEqual(c.current(), r.snapshot);
  return r.snapshot;
}
function select(c, selection) {
  const old = c.current();
  const r = ready(c.prepareSelection(selection));
  assert.strictEqual(c.current(), old);
  assert.strictEqual(r.snapshot.result, old.result);
  assert.strictEqual(c.commit(r.ticket), r.snapshot);
  return r.snapshot;
}
function columns(base = 0n, scale = 7n) {
  const row = [5n,1n,4n,2n,3n,0n];
  return {n:6, scale, values:Array.from({length:36}, (_, i) => base + row[i % 6])};
}
function flat3() {
  return {n:3,scale:2n,values:Array(9).fill(-7n)};
}

// Independent oracle: explicit graph/BFS, no union-find or production imports.
function components(active, adjacency, values) {
  const seen = new Set(), result = [];
  for (const first of [...active].sort((a,b) => a-b)) {
    if (seen.has(first)) continue;
    const vertices = [], queue = [first];
    seen.add(first);
    for (let offset = 0; offset < queue.length; offset++) {
      const vertex = queue[offset];
      vertices.push(vertex);
      for (const next of adjacency[vertex]) {
        if (!seen.has(next)) {seen.add(next); queue.push(next);}
      }
    }
    vertices.sort((a,b) => a-b);
    let elder = vertices[0];
    for (const vertex of vertices) {
      if (values[vertex] > values[elder]
          || (values[vertex] === values[elder] && vertex < elder)) elder = vertex;
    }
    result.push({elder,vertices});
  }
  return result.sort((a,b) => a.elder-b.elder);
}
function oracleTrace(input) {
  const {n,scale} = input, values = [...input.values], N = n*n;
  const active = new Set(), adjacency = Array.from({length:N}, () => new Set());
  const order = Array.from({length:N}, (_, i) => i).sort((a,b) =>
    values[a] > values[b] ? -1 : values[a] < values[b] ? 1 : a-b);
  const events = [], levels = [], finite = [];
  let zero = 0;
  for (let position = 0; position < order.length; position++) {
    const vertex = order[position], level = values[vertex];
    active.add(vertex);
    events.push({kind:'activate',vertex,level});
    const x = Math.floor(vertex/n), y = vertex%n;
    const neighbors = [((x+n-1)%n)*n+y, ((x+1)%n)*n+y,
      x*n+(y+n-1)%n, x*n+(y+1)%n];
    for (const neighbor of neighbors) {
      if (!active.has(neighbor)) continue;
      const before = components(active,adjacency,values);
      const a = before.find(c => c.vertices.includes(vertex));
      const b = before.find(c => c.vertices.includes(neighbor));
      if (a === b) {
        events.push({kind:'neutral',edge:[vertex,neighbor],level,elder:a.elder});
      } else {
        const ae = a.elder, be = b.elder;
        const survivor = values[ae] > values[be]
          || (values[ae] === values[be] && ae < be) ? ae : be;
        const dying = survivor === ae ? be : ae;
        const birth = values[dying], death = level;
        events.push({kind:'merge',edge:[vertex,neighbor],level,survivor,dying,
          birth,death,zero:birth === death});
        if (birth === death) zero++;
        else finite.push([birth,death,dying]);
      }
      adjacency[vertex].add(neighbor);
      adjacency[neighbor].add(vertex);
    }
    if (position + 1 === order.length || values[order[position+1]] !== level) {
      levels.push({level,eventEnd:events.length,activeCount:active.size,
        componentCount:components(active,adjacency,values).length,
        positiveCount:finite.length,zeroCount:zero});
    }
  }
  finite.sort((a,b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1
    : a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : a[2]-b[2]);
  const elder = components(active,adjacency,values)[0].elder;
  return {n,scale,values,events,levels,barcode:{
    intervals:finite.map(r => [r[0],r[1]]),
    birth_vertices:finite.map(r => r[2]),
    essential:[values[elder]],essential_vertices:[elder],zero_count:zero}};
}
function oracleSelected(trace, eventEnd) {
  const active = new Set(), edges = [];
  const adjacency = Array.from({length:trace.values.length}, () => new Set());
  let positiveCount = 0, zeroCount = 0, neutralCount = 0;
  for (const event of trace.events.slice(0,eventEnd)) {
    if (event.kind === 'activate') active.add(event.vertex);
    else {
      edges.push([...event.edge]);
      const [a,b] = event.edge;
      adjacency[a].add(b); adjacency[b].add(a);
      if (event.kind === 'neutral') neutralCount++;
      else if (event.zero) zeroCount++;
      else positiveCount++;
    }
  }
  return {activeVertices:[...active].sort((a,b) => a-b),edges,
    components:components(active,adjacency,trace.values),positiveCount,zeroCount,
    neutralCount,lastEvent:eventEnd === 0 ? null : trace.events[eventEnd-1]};
}
function expectedNavigation(trace, E) {
  const earlier = trace.levels.map((l,i) => [l.eventEnd,i]).filter(([e]) => e<E);
  const later = trace.levels.findIndex(l => l.eventEnd>E);
  return {previousEvent:E>0 ? E-1 : null,
    nextEvent:E<trace.events.length ? E+1 : null,
    previousLevel:earlier.length ? earlier[earlier.length-1][1] : null,
    nextLevel:later<0 ? null : later,reset:E>0 ? 0 : null};
}
test(CONTROL_INVENTORY[1], () => {
  const trace = oracleTrace(flat3());
  assert.equal(trace.events.length,27);
  assert.deepEqual(trace.events.slice(0,6), [
    {kind:'activate',vertex:0,level:-7n},
    {kind:'activate',vertex:1,level:-7n},
    {kind:'merge',edge:[1,0],level:-7n,survivor:0,dying:1,birth:-7n,death:-7n,zero:true},
    {kind:'activate',vertex:2,level:-7n},
    {kind:'merge',edge:[2,1],level:-7n,survivor:0,dying:2,birth:-7n,death:-7n,zero:true},
    {kind:'neutral',edge:[2,0],level:-7n,elder:0},
  ]);
  assert.deepEqual([2,3,6].map(E => {
    const s=oracleSelected(trace,E);
    return [s.activeVertices.length,s.components.length,s.zeroCount,s.neutralCount];
  }), [[2,2,0,0],[2,1,1,0],[3,1,2,1]]);
  assert.deepEqual(trace.levels,[{level:-7n,eventEnd:27,activeCount:9,
    componentCount:1,positiveCount:0,zeroCount:8}]);
  const six = oracleTrace(columns());
  assert.deepEqual(six.barcode,{intervals:[[3n,2n],[4n,1n]],
    birth_vertices:[4,2],essential:[5n],essential_vertices:[0],zero_count:33});
  assert.deepEqual(six.levels.map(l => [l.level,l.eventEnd,l.activeCount,
    l.componentCount,l.positiveCount,l.zeroCount]), [
    [5n,12,6,1,0,5],[4n,24,12,2,0,10],[3n,36,18,3,0,15],
    [2n,60,24,2,1,21],[1n,84,30,1,2,27],[0n,108,36,1,2,33],
  ]);
});

behavior(2, () => {
  // Catches JSON cloning, lost -0, omitted normalizers and wrong r^3 scaling.
  const c = controller();
  const s = install(c,{model:'cubic',input:{r:1,k:1,b:-0}});
  assert.deepEqual(s.result,{r:1,k:1,b:-0,rCubed:1,pins:[-0.5,0.5],gap:1,
    heightWindow:[-1,-0],normalized:{xDivisor:1,yOrigin:-0,yDivisor:1,
      pins:[-0.5,0.5],heightWindow:[-1,+0]}});
  for (const v of [s.result.b,s.result.heightWindow[1],s.result.normalized.yOrigin])
    assert.equal(Object.is(v,-0),true);
  assert.equal(Object.is(s.result.normalized.heightWindow[1],+0),true);
  const half = install(c,{model:'cubic',input:{r:0.5,k:1,b:0}});
  assert.deepEqual(half.result,{r:0.5,k:1,b:0,rCubed:0.125,pins:[-0.25,0.25],
    gap:0.125,heightWindow:[-0.125,0],normalized:{xDivisor:0.5,yOrigin:0,
      yDivisor:0.125,pins:[-0.5,0.5],heightWindow:[-1,0]}});
});
behavior(3, () => {
  // Catches Number conversion, fixed decimal caps, essential/longest-bar loss.
  const base = 10n**400n;
  const s = install(controller(),{model:'elder',input:columns(base,7n)});
  assert.equal(s.result.values[0],base+5n);
  assert.deepEqual(s.result.barcode,{
    intervals:[[base+3n,base+2n],[base+4n,base+1n]],birth_vertices:[4,2],
    essential:[base+5n],essential_vertices:[0],zero_count:33});
  assert.equal(s.result.scale,7n);
  assert.deepEqual(s.result.barcode.intervals.map(([b,d]) => b-d),[1n,3n]);
  assert.equal(s.result.levels[5].level,base);
  const hugeScale=base+7n;
  const scaled=install(controller(),{model:'elder',input:{...flat3(),scale:hugeScale}});
  assert.equal(scaled.result.scale,hugeScale);
  assert.deepEqual(scaled.result.barcode,{intervals:[],birth_vertices:[],
    essential:[-7n],essential_vertices:[0],zero_count:8});
});
behavior(4, () => {
  // Catches aliases, shallow freezing, mutation/sorting, selection recomputation.
  const c = controller(), input = columns(), before = {...input,values:[...input.values]};
  const r = ready(c.prepareDocument({model:'elder',input}));
  assert.deepEqual(input,before);
  input.values[0] = -999n; input.scale = 999n;
  assert.deepEqual(r.snapshot.result.values,before.values);
  assert.equal(r.snapshot.result.scale,7n);
  const s = c.commit(r.ticket);
  assert.throws(() => s.result.values.push(9n),TypeError);
  assert.throws(() => {s.result.events[0].level=999n;},TypeError);
  assert.throws(() => s.result.barcode.intervals[0].reverse(),TypeError);
  assert.throws(() => {s.source.binding.label='changed';},TypeError);
  for (const E of [3,6,36,108,0]) {
    const next = select(c,{mode:'event_prefix',eventEnd:E});
    assert.strictEqual(next.result,s.result);
    assert.deepEqual(next.result.values,before.values);
    deeplyFrozen(next);
  }
  const independent = install(controller(),{model:'elder',input:before});
  assert.notStrictEqual(independent.result,s.result);
  assert.notStrictEqual(independent.result.events,s.result.events);
});
behavior(5, () => {
  // Catches coerced/inherited/sparse inputs and invalid partial publication.
  const c = controller(), old = install(c,{model:'cubic',input:{r:1,k:1,b:0}});
  const inherited = Object.create({r:1}); inherited.k=1; inherited.b=0;
  const sparse = Array(9).fill(0n); delete sparse[4];
  const inheritedSlots = Array(9).fill(0n); delete inheritedSlots[4];
  const proto = Object.create(Array.prototype); proto[4]=0n;
  Object.setPrototypeOf(inheritedSlots,proto);
  let getterCalls = 0;
  const accessor = {k:1,b:0};
  Object.defineProperty(accessor,'r',{get(){getterCalls++; return 1;},enumerable:true});
  const invalid = [
    null, [], {}, {model:'unknown',input:{}},
    Object.create({model:'cubic',input:{r:1,k:1,b:0}}),
    {model:'cubic',input:{r:1,k:1,b:0},result:{r:999}},
    {model:'cubic',input:null}, {model:'cubic',input:inherited},
    {model:'cubic',input:accessor},
    ...['1',true,new Number(1),NaN,Infinity,-Infinity,0,-1,Number.MIN_VALUE]
      .map(r => ({model:'cubic',input:{r,k:1,b:0}})),
    ...['k','b'].flatMap(field => ['1',true,new Number(1),NaN,Infinity,-Infinity]
      .map(value => ({model:'cubic',input:{r:1,k:1,b:0,[field]:value}}))),
    {model:'cubic',input:{r:1,k:0,b:0}},
    {model:'cubic',input:{r:1,k:1,b:1e308}},
    {model:'cubic',input:{r:1e200,k:1,b:0}},
    ...[2,33,3.5,'3',true].map(n =>
      ({model:'elder',input:{n,scale:1n,values:Array(9).fill(0n)}})),
    ...[0n,-1n,1,'1',true].map(scale =>
      ({model:'elder',input:{n:3,scale,values:Array(9).fill(0n)}})),
    ...[sparse,inheritedSlots,Array(8).fill(0n),Array(9).fill(0),
      new BigInt64Array(9)].map(values =>
      ({model:'elder',input:{n:3,scale:1n,values}})),
  ];
  for (const document of invalid) {
    refusal(c.prepareDocument(document),'INVALID_DOCUMENT');
    assert.strictEqual(c.current(),old);
  }
  assert.equal(getterCalls,0);
  const extra = {r:1,k:1,b:0,ignored:'native extras remain ignored'};
  assert.equal(install(c,{model:'cubic',input:extra}).result.r,1);
  const ownNullPrototype=Object.assign(Object.create(null),{r:1,k:1,b:-0});
  assert.equal(Object.is(install(c,{model:'cubic',input:ownNullPrototype}).result.b,-0),true);
  const maximum=install(c,{model:'elder',
    input:{n:32,scale:1n,values:Array(1024).fill(0n)}});
  assert.equal(maximum.result.n,32);
  assert.equal(maximum.result.events.length,3072);
  assert.equal(maximum.result.barcode.zero_count,1023);
});
behavior(6, () => {
  // Catches activation-only connectivity and off-by-one complete-block labels.
  const c = controller(), trace = oracleTrace(flat3());
  const initial = install(c,{model:'elder',input:flat3()});
  assert.deepEqual(initial.selection,{mode:'event_prefix',eventEnd:0});
  assert.deepEqual(initial.selected,oracleSelected(trace,0));
  for (const E of [2,3,6]) {
    const s = select(c,{mode:'event_prefix',eventEnd:E});
    assert.deepEqual(s.selected,oracleSelected(trace,E));
    assert.strictEqual(s.selected.lastEvent,s.result.events[E-1]);
    assert.deepEqual(s.selection,{mode:'event_prefix',eventEnd:E});
  }
  const full = select(c,{mode:'complete_level',levelIndex:0});
  assert.deepEqual(full.selection,{mode:'complete_level',levelIndex:0,eventEnd:27});
  assert.equal(full.selected.activeVertices.length,9);
  assert.equal(full.selected.components.length,1);
  assert.equal(full.selected.zeroCount,8);
  assert.equal(full.selected.neutralCount,10);
});
behavior(7, () => {
  // Catches nonperiodic/diagonal connectivity, storage-root elders and fake counts.
  const cases = [
    columns(),
    {n:3,scale:3n,values:[9n,0n,8n,0n,7n,0n,6n,0n,5n]},
    {n:3,scale:2n,values:[4n,1n,4n,1n,4n,1n,4n,1n,4n]},
    {n:4,scale:1n,values:[5n,-2n,-2n,-2n,-2n,5n,-2n,-2n,
      -2n,-2n,5n,-2n,-2n,-2n,-2n,5n]},
  ];
  for (const input of cases) {
    const expected = oracleTrace(input), c = controller();
    const start = install(c,{model:'elder',input});
    assert.deepEqual(start.result,expected);
    for (let E=0; E<=expected.events.length; E++) {
      const s = select(c,{mode:'event_prefix',eventEnd:E});
      assert.deepEqual(s.selected,oracleSelected(expected,E));
      assert.deepEqual(s.navigation,expectedNavigation(expected,E));
      assert.strictEqual(s.result,start.result);
      if (E>0) assert.strictEqual(s.selected.lastEvent,s.result.events[E-1]);
    }
  }
});
behavior(8, () => {
  // Catches wrapping/rounded level jumps, conflated modes and lost finite bars.
  const c = controller(), expected = oracleTrace(columns());
  const first = install(c,{model:'elder',input:columns()});
  for (const E of [0,11,12,13,24,60,108]) {
    const s = select(c,{mode:'event_prefix',eventEnd:E});
    assert.deepEqual(s.navigation,expectedNavigation(expected,E));
    assert.equal(s.selection.mode,'event_prefix');
    assert.strictEqual(s.result.barcode,first.result.barcode);
    assert.equal(s.result.barcode.zero_count,33);
    assert.equal(s.result.barcode.intervals.length,2);
  }
  for (let i=0;i<expected.levels.length;i++) {
    const s = select(c,{mode:'complete_level',levelIndex:i});
    const E = expected.levels[i].eventEnd;
    assert.deepEqual(s.selection,{mode:'complete_level',levelIndex:i,eventEnd:E});
    assert.deepEqual(s.navigation,expectedNavigation(expected,E));
    assert.deepEqual(s.selected,oracleSelected(expected,E));
  }
  const reset = select(c,{mode:'reset'});
  assert.deepEqual(reset.selection,{mode:'event_prefix',eventEnd:0});
  assert.deepEqual(reset.selected,oracleSelected(expected,0));
  const old = c.current();
  const inherited = Object.create({mode:'event_prefix',eventEnd:0});
  for (const choice of [null,{},inherited,{mode:'next'},
    ...[-1,109,0.5,'0',0n,NaN,Infinity].map(eventEnd =>
      ({mode:'event_prefix',eventEnd})),
    {mode:'complete_level',levelIndex:-1},{mode:'complete_level',levelIndex:6},
    {mode:'complete_level',levelIndex:'0'},
    {mode:'event_prefix',eventEnd:0,extra:true},{mode:'reset',eventEnd:0}]) {
    refusal(c.prepareSelection(choice),'INVALID_SELECTION');
    assert.strictEqual(c.current(),old);
  }
  const empty = controller();
  refusal(empty.prepareSelection({mode:'reset'}),'NO_DOCUMENT');
  const cubic = controller();
  install(cubic,{model:'cubic',input:{r:1,k:1,b:0}});
  for (const action of [{mode:'event_prefix',eventEnd:0},
    {mode:'complete_level',levelIndex:0},{mode:'reset'}])
    refusal(cubic.prepareSelection(action),'SELECTION_UNAVAILABLE');
});
behavior(9, () => {
  // Catches premature publication, failure invalidation and accepted stale tokens.
  const c = controller();
  assert.equal(c.current(),null);
  const first = install(c,{model:'elder',input:flat3()});
  assert.equal(first.revision,1n); assert.equal(first.documentRevision,1n);
  const paused = select(c,{mode:'event_prefix',eventEnd:6});
  assert.equal(paused.revision,2n); assert.equal(paused.documentRevision,1n);
  const a = ready(c.prepareDocument({model:'cubic',input:{r:1,k:1,b:0}}));
  assert.strictEqual(c.current(),paused);
  const b = ready(c.prepareDocument({model:'cubic',input:{r:0.5,k:1,b:-0}}));
  refusal(c.prepareDocument({model:'elder',input:{}}),'INVALID_DOCUMENT');
  assert.strictEqual(c.current(),paused);
  refusal(c.commit(a.ticket),'STALE_TICKET');
  assert.strictEqual(c.current(),paused,'older pending ticket must not publish');
  const latest = c.commit(b.ticket);
  assert.strictEqual(latest,b.snapshot);
  assert.equal(latest.revision,3n); assert.equal(latest.documentRevision,2n);
  refusal(c.commit(a.ticket),'STALE_TICKET');
  refusal(c.commit(b.ticket),'STALE_TICKET');
  refusal(c.commit(Object.freeze({})),'STALE_TICKET');
  const other = controller();
  const foreign = ready(other.prepareDocument({model:'cubic',input:{r:1,k:1,b:0}}));
  refusal(c.commit(foreign.ticket),'STALE_TICKET');
  assert.strictEqual(c.current(),latest);
  assert.strictEqual(other.current(),null);

  // One latest outstanding ticket spans both successful prepare methods.
  for (const route of ['selection->selection','selection->document','document->selection']) {
    const cross=controller();
    const origin=install(cross,{model:'elder',input:flat3()});
    const oldTicket=route.startsWith('selection')
      ? ready(cross.prepareSelection({mode:'event_prefix',eventEnd:2}))
      : ready(cross.prepareDocument({model:'cubic',input:{r:1,k:1,b:0}}));
    const newerIsDocument=route.endsWith('document');
    const newer=newerIsDocument
      ? ready(cross.prepareDocument({model:'cubic',input:{r:0.5,k:1,b:-0}}))
      : ready(cross.prepareSelection({mode:'event_prefix',eventEnd:6}));
    assert.strictEqual(cross.current(),origin,route + ': preparation is private');
    refusal(cross.commit(oldTicket.ticket),'STALE_TICKET');
    assert.strictEqual(cross.current(),origin,route + ': reject before newer commit');
    refusal(cross.prepareSelection({mode:'event_prefix',eventEnd:28}),'INVALID_SELECTION');
    refusal(cross.prepareDocument({model:'elder',input:{}}),'INVALID_DOCUMENT');
    assert.strictEqual(cross.current(),origin,route + ': failed preparation is private');
    const committed=cross.commit(newer.ticket);
    assert.strictEqual(committed,newer.snapshot,route + ': latest exact snapshot');
    assert.strictEqual(cross.current(),committed);
    assert.strictEqual(committed.result,newer.snapshot.result);
    assert.equal(committed.revision,2n,route + ': one commit, no refusal increment');
    assert.equal(committed.documentRevision,newerIsDocument ? 2n : 1n);
    if (newerIsDocument) {
      assert.equal(committed.model,'cubic');
      assert.notStrictEqual(committed.result,origin.result);
      assert.deepEqual(committed.result,{r:0.5,k:1,b:-0,rCubed:0.125,
        pins:[-0.25,0.25],gap:0.125,heightWindow:[-0.125,-0],
        normalized:{xDivisor:0.5,yOrigin:-0,yDivisor:0.125,
          pins:[-0.5,0.5],heightWindow:[-1,+0]}});
      assert.deepEqual(committed.selection,{mode:'static'});
      assert.equal(committed.selected,null);
      assert.equal(committed.navigation,null);
    } else {
      assert.equal(committed.model,'elder');
      assert.strictEqual(committed.result,origin.result,route + ': owned result retained');
      assert.deepEqual(committed.selection,{mode:'event_prefix',eventEnd:6});
      assert.deepEqual(committed.selected,oracleSelected(oracleTrace(flat3()),6));
      assert.strictEqual(committed.selected.lastEvent,committed.result.events[5]);
      assert.deepEqual(committed.navigation,{
        previousEvent:5,nextEvent:7,previousLevel:null,nextLevel:0,reset:0});
    }
    deeplyFrozen(committed);
    refusal(cross.commit(oldTicket.ticket),'STALE_TICKET');
    refusal(cross.commit(newer.ticket),'STALE_TICKET');
    assert.strictEqual(cross.current(),committed,route + ': stale/reuse retains latest');
  }
});

function numberText(v) {
  return Object.is(v,-0) ? '-0' : Object.is(v,+0) ? '+0' : String(v);
}
function decodedText(html) {
  const entities = {amp:'&',lt:'<',gt:'>',quot:'"', '#39':"'"};
  return html.replace(/<[^>]*>/g,'').replace(/&(amp|lt|gt|quot|#39);/g,
    (_,name) => entities[name]).trim();
}
function tableRows(html, name) {
  const pattern = new RegExp('<table\\b[^>]*data-reader-table="' + name
    + '"[^>]*>([\\s\\S]*?)<\\/table>','g');
  const matches = [...html.matchAll(pattern)];
  assert.equal(matches.length,1,'one complete ' + name + ' table');
  const table = matches[0][1];
  assert.match(table,/<caption\b[^>]*>[\s\S]+?<\/caption>/);
  assert.match(table,/<thead\b[^>]*>[\s\S]*?<th\b[^>]*scope="col"/);
  const body = table.match(/<tbody\b[^>]*>([\s\S]*?)<\/tbody>/);
  assert.ok(body,'real tbody for ' + name);
  return [...body[1].matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)].map(row => {
    assert.match(row[1],/<th\b[^>]*scope="row"/);
    return [...row[1].matchAll(/<(td|th)\b[^>]*>([\s\S]*?)<\/\1>/g)]
      .map(cell => decodedText(cell[2]));
  });
}
function sourceRows(b) {
  return [
    ['repository',b.repository],['modelCommit',b.modelCommit],
    ['cubicSha256',b.cubicSha256],['elderSha256',b.elderSha256],
    ['contractSha256',b.contractSha256],['label',b.label],
    ['composition.commit',b.composition ? b.composition.commit : 'UNKNOWN'],
    ['composition.tree',b.composition ? b.composition.tree : 'UNKNOWN'],
    ['qualification','DECLARED_PINS_ONLY'],['inputProducer','DECLARED_LABEL_ONLY'],
    ['consumer','UNQUALIFIED'],['build','UNQUALIFIED'],['assets','UNQUALIFIED'],
    ['graph','UNMAPPED'],['review','UNKNOWN'],['kernel','UNMAPPED'],
    ['scientificEffect','NONE'],['scientificStatusAuthority','false'],
  ];
}
function section(html,s) {
  assert.equal(typeof html,'string');
  assert.match(html,/<section\b[^>]*id="canvas-reader"/);
  assert.match(html,/<section\b[^>]*aria-labelledby="reader-title"/);
  assert.match(html,/<h2\b[^>]*id="reader-title"/);
  assert.ok(html.includes('data-reader-model="' + s.model + '"'));
  assert.ok(html.includes('data-reader-revision="' + s.revision + '"'));
  assert.ok(html.includes('data-document-revision="' + s.documentRevision + '"'));
  assert.ok(html.includes('data-selection-mode="' + s.selection.mode + '"'));
  assert.deepEqual(tableRows(html,'provenance'),sourceRows(s.source.binding));
  for (const url of Object.values(urls)) assert.ok(html.includes('href="' + url + '"'));
  assert.doesNotMatch(html,/<script\b|<img\b|href="javascript:/i);
}
function checkElderHTML(html,s,expected,E) {
  section(html,s);
  assert.ok(html.includes('data-event-end="' + E + '"'));
  const p=oracleSelected(expected,E);
  assert.deepEqual(tableRows(html,'grid'),expected.values.map((v,i) =>
    [String(i),String(Math.floor(i/expected.n)),String(i%expected.n),String(v),
      String(p.activeVertices.includes(i))]));
  const eventRows=expected.events.map((e,i) => [
    String(i),e.kind,e.vertex === undefined ? '' : String(e.vertex),
    e.edge ? String(e.edge[0]) : '',e.edge ? String(e.edge[1]) : '',String(e.level),
    e.survivor === undefined ? '' : String(e.survivor),
    e.dying === undefined ? '' : String(e.dying),
    e.birth === undefined ? '' : String(e.birth),
    e.death === undefined ? '' : String(e.death),
    e.zero === undefined ? '' : String(e.zero),
    e.elder === undefined ? '' : String(e.elder),String(i<E)]);
  assert.deepEqual(tableRows(html,'events'),eventRows);
  assert.deepEqual(tableRows(html,'levels'),expected.levels.map((l,i) =>
    [String(i),String(l.level),String(l.eventEnd),String(l.activeCount),
      String(l.componentCount),String(l.positiveCount),String(l.zeroCount),
      'Complete equal-value block']));
  const scale=String(expected.scale);
  assert.deepEqual(tableRows(html,'intervals'),expected.barcode.intervals.map(([b,d],i) =>
    [String(i),String(b),String(d),String(expected.barcode.birth_vertices[i]),
      '(' + d + '/' + scale + ', ' + b + '/' + scale + ']',
      (b-d) + '/' + scale]));
  assert.deepEqual(tableRows(html,'essential'),[
    [String(expected.barcode.essential[0]),String(expected.barcode.essential_vertices[0]),
      '(-infinity, ' + expected.barcode.essential[0] + '/' + scale + ']']]);
  assert.deepEqual(tableRows(html,'components'),p.components.map(c =>
    [String(c.elder),c.vertices.join(',')]));
  assert.deepEqual(tableRows(html,'summary'),[
    ['n',String(expected.n)],['scale',scale],
    ['zero_count',String(expected.barcode.zero_count)],
    ['selection.mode',s.selection.mode],['selection.eventEnd',String(E)],
    ['selection.levelIndex',s.selection.mode === 'complete_level'
      ? String(s.selection.levelIndex) : 'NONE'],
    ['selected.activeCount',String(p.activeVertices.length)],
    ['selected.componentCount',String(p.components.length)],
    ['selected.positiveCount',String(p.positiveCount)],
    ['selected.zeroCount',String(p.zeroCount)],
    ['selected.neutralCount',String(p.neutralCount)],
    ['selected.lastEventIndex',E === 0 ? 'NONE' : String(E-1)],
    ['interpretation',s.selection.mode === 'complete_level'
      ? 'Complete equal-value block' : 'Insertion-order prefix'],
  ]);
  assert.ok(html.includes('id="event-prefix-0"'));
  expected.events.forEach((_,i) =>
    assert.ok(html.includes('id="event-prefix-' + (i+1) + '"')));
  const levelsTable=html.match(/<table\b[^>]*data-reader-table="levels"[^>]*>([\s\S]*?)<\/table>/)[1];
  const levelBodies=[...levelsTable.matchAll(/<tbody\b[^>]*>([\s\S]*?)<\/tbody>/g)];
  const levelRows=[...levelBodies[0][1].matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/g)];
  expected.levels.forEach((l,i) => {
    assert.ok(levelRows[i][1].includes('href="#event-prefix-' + l.eventEnd + '"'));
  });
}
behavior(10, () => {
  // Catches omitted rows/variant fields, bad endpoint math and source-label injection.
  const label="</td><script>alert('source')</script>&\"";
  const c=controller(label), base=10n**400n, input=columns(base,7n);
  const expected=oracleTrace(input);
  install(c,{model:'elder',input});
  for (const selection of [{mode:'event_prefix',eventEnd:0},
    {mode:'event_prefix',eventEnd:6},{mode:'complete_level',levelIndex:3},
    {mode:'event_prefix',eventEnd:108}]) {
    const s=select(c,selection), html=setup.view.renderStaticReader(s);
    checkElderHTML(html,s,expected,s.selection.eventEnd);
    assert.ok(html.includes('&lt;/td&gt;&lt;script&gt;alert(&#39;source&#39;)&lt;/script&gt;&amp;&quot;'));
    assert.strictEqual(c.current(),s,'rendering must not select or publish');
  }
  const flat=install(controller(),{model:'elder',input:flat3()});
  checkElderHTML(setup.view.renderStaticReader(flat),flat,oracleTrace(flat3()),0);
  const cubic=install(controller(),{model:'cubic',input:{r:1,k:1,b:-0}});
  const html=setup.view.renderStaticReader(cubic);
  section(html,cubic);
  const expectedFields=[
    ['r',1],['k',1],['b',-0],['rCubed',1],['pins[0]',-0.5],['pins[1]',0.5],
    ['gap',1],['heightWindow[0]',-1],['heightWindow[1]',-0],
    ['normalized.xDivisor',1],['normalized.yOrigin',-0],['normalized.yDivisor',1],
    ['normalized.pins[0]',-0.5],['normalized.pins[1]',0.5],
    ['normalized.heightWindow[0]',-1],['normalized.heightWindow[1]',+0],
  ];
  assert.deepEqual(tableRows(html,'cubic-fields'),
    expectedFields.map(([name,value]) => [name,numberText(value)]));
  assert.deepEqual(cubic.selection,{mode:'static'});
  assert.equal(cubic.selected,null); assert.equal(cubic.navigation,null);
  const half=install(controller(),{model:'cubic',input:{r:0.5,k:1,b:2}});
  const halfHTML=setup.view.renderStaticReader(half);
  section(halfHTML,half);
  assert.deepEqual(tableRows(halfHTML,'cubic-fields'),[
    ['r','0.5'],['k','1'],['b','2'],['rCubed','0.125'],
    ['pins[0]','-0.25'],['pins[1]','0.25'],['gap','0.125'],
    ['heightWindow[0]','1.875'],['heightWindow[1]','2'],
    ['normalized.xDivisor','0.5'],['normalized.yOrigin','2'],
    ['normalized.yDivisor','0.125'],['normalized.pins[0]','-0.5'],
    ['normalized.pins[1]','0.5'],['normalized.heightWindow[0]','-1'],
    ['normalized.heightWindow[1]','+0'],
  ]);
});
behavior(11, () => {
  // Catches caller-chosen links/pins and false authenticated or scientific joins.
  const original=binding('Source label & provenance');
  const c=setup.state.createReaderController(original);
  original.label='mutated after creation';
  const s=install(c,{model:'cubic',input:{r:1,k:1,b:0}});
  const expected={binding:binding('Source label & provenance'),
    qualification:'DECLARED_PINS_ONLY',inputProducer:'DECLARED_LABEL_ONLY',
    consumer:'UNQUALIFIED',build:'UNQUALIFIED',assets:'UNQUALIFIED',
    graph:'UNMAPPED',review:'UNKNOWN',kernel:'UNMAPPED',scientificEffect:'NONE',
    scientificStatusAuthority:false,urls};
  assert.deepEqual(s.source,expected);
  for (const field of ['repository','modelCommit','cubicSha256','elderSha256','contractSha256']) {
    const bad=binding(); bad[field]='wrong';
    assert.throws(() => setup.state.createReaderController(bad),TypeError);
  }
  for (const composition of [{commit:'bad',tree:'bad'},
    {commit:COMMIT},{commit:COMMIT,tree:COMMIT,extra:true}]) {
    assert.throws(() => setup.state.createReaderController({...binding(),composition}),TypeError);
  }
  assert.throws(() => setup.state.createReaderController({...binding(),
    url:'javascript:alert(1)'}),TypeError);
  // Actual historical model composition, not an authored consumer/build identity.
  const declared={commit:COMMIT,tree:'e102ee82ff9c1723be408f5ac4c84b415ad659ef'};
  const labeled=setup.state.createReaderController({...binding(),composition:declared});
  declared.commit='mutated after creation';
  const next=install(labeled,{model:'cubic',input:{r:1,k:1,b:0}});
  assert.deepEqual(next.source.binding.composition,
    {commit:COMMIT,tree:'e102ee82ff9c1723be408f5ac4c84b415ad659ef'});
  assert.equal(next.source.qualification,'DECLARED_PINS_ONLY');
  assert.equal(next.source.consumer,'UNQUALIFIED');
  assert.equal(next.source.build,'UNQUALIFIED');
  assert.equal(next.source.graph,'UNMAPPED');
  assert.equal(next.source.kernel,'UNMAPPED');
});
