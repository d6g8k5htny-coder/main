// Direct native reader state; NON-CERTIFYING, scientific effect NONE.
// Author: OpenAI/Codex /root/architecture_memory_regression_inventory.
// Source/test exposed, same provider/principal, organizational independence 0.
// Private source candidate only; no runtime or behavioral acceptance asserted.
import {cubicModel} from './cubic-model.mjs';
import {finiteElderTrace} from './elder-model.mjs';

const PINS = Object.freeze({
  repository: 'd6g8k5htny-coder/main',
  modelCommit: '77d35feff69c4df1b55a7449f3deaa68ed75de7d',
  cubicSha256: 'fbff097a93429652ce5077056c400343b1472375a9cb1cbc388894c7e7bf2051',
  elderSha256: 'd92b8f6139d42360a35e933b40a287414b95811f14f0dc19c520d1d2558ec767',
  contractSha256: '3473a664be9126b67006527f5257a7eeb4c2082c836d7ff1b38e21f0d40eecbc',
});
const SOURCE_ROOT = 'https://github.com/' + PINS.repository + '/blob/'
  + PINS.modelCommit + '/';
const SOURCE_URLS = Object.freeze({
  cubic: SOURCE_ROOT + 'interface/src/scripts/cubic-model.mjs',
  elder: SOURCE_ROOT + 'interface/src/scripts/elder-model.mjs',
  contract: SOURCE_ROOT + 'docs/plans/2026-10-09-canvas-model-contract.md',
});
const REFUSALS = Object.freeze(Object.fromEntries([
  'INVALID_DOCUMENT', 'INVALID_SELECTION', 'NO_DOCUMENT',
  'SELECTION_UNAVAILABLE', 'STALE_TICKET',
].map(code => [code, Object.freeze({kind: 'Refusal', code})])));

// Descriptor inspection never reads an accessor's value. This is a native
// data boundary, not a sandbox for Proxies or mutated JavaScript intrinsics.
function ownData(value, record = true) {
  if (value === null || typeof value !== 'object'
      || (record && Array.isArray(value))) {
    throw new TypeError('Plain data is required');
  }
  if (record) {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError('Plain record is required');
    }
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const key of Reflect.ownKeys(descriptors)) {
    if (!Object.hasOwn(descriptors[key], 'value')) {
      throw new TypeError('Accessors are not reader data');
    }
  }
  return descriptors;
}

function closedData(value, keys) {
  const descriptors = ownData(value);
  const names = Reflect.ownKeys(descriptors);
  if (names.length !== keys.length
      || names.some(name => typeof name !== 'string' || !keys.includes(name))
      || keys.some(key => !Object.hasOwn(descriptors, key))) {
    throw new TypeError('Unexpected record fields');
  }
  return descriptors;
}

function required(descriptors, key) {
  if (!Object.hasOwn(descriptors, key)) {
    throw new TypeError('Own input field is required');
  }
  return descriptors[key].value;
}

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const key of Reflect.ownKeys(value)) {
      deepFreeze(Object.getOwnPropertyDescriptor(value, key).value);
    }
    Object.freeze(value);
  }
  return value;
}

// The pinned models return plain native data. Copy their complete result,
// retaining Number bits (including -0), BigInt and every nested result field.
function copyResult(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean'
      || typeof value === 'bigint') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) {
    const descriptors = ownData(value, false);
    const length = required(descriptors, 'length');
    const result = [];
    for (let index = 0; index < length; index++) {
      result.push(copyResult(required(descriptors, String(index))));
    }
    return result;
  }
  const descriptors = ownData(value);
  const result = {};
  for (const key of Reflect.ownKeys(descriptors)) {
    if (typeof key !== 'string') throw new TypeError('Unexpected result field');
    Object.defineProperty(result, key, {
      value: copyResult(descriptors[key].value),
      enumerable: true, writable: true, configurable: true,
    });
  }
  return result;
}

function copyBinding(binding) {
  const descriptors = closedData(binding, [
    'repository', 'modelCommit', 'cubicSha256', 'elderSha256',
    'contractSha256', 'label', 'composition',
  ]);
  for (const key of Object.keys(PINS)) {
    if (required(descriptors, key) !== PINS[key]) {
      throw new TypeError('Declared source pins do not match');
    }
  }
  const label = required(descriptors, 'label');
  if (typeof label !== 'string') throw new TypeError('Text label is required');
  const declared = required(descriptors, 'composition');
  let composition = null;
  if (declared !== null) {
    const parts = closedData(declared, ['commit', 'tree']);
    const commit = required(parts, 'commit'), tree = required(parts, 'tree');
    if (typeof commit !== 'string' || typeof tree !== 'string'
        || !/^[0-9a-f]{40}$/.test(commit) || !/^[0-9a-f]{40}$/.test(tree)) {
      throw new TypeError('Full declared composition pins are required');
    }
    composition = {commit, tree};
  }
  return {...PINS, label, composition};
}

function sourceFor(binding) {
  return deepFreeze({
    binding: copyBinding(binding),
    qualification: 'DECLARED_PINS_ONLY',
    inputProducer: 'DECLARED_LABEL_ONLY',
    consumer: 'UNQUALIFIED',
    build: 'UNQUALIFIED',
    assets: 'UNQUALIFIED',
    graph: 'UNMAPPED',
    review: 'UNKNOWN',
    kernel: 'UNMAPPED',
    scientificEffect: 'NONE',
    scientificStatusAuthority: false,
    urls: SOURCE_URLS,
  });
}

function copyCubicInput(input) {
  const descriptors = ownData(input);
  const r = required(descriptors, 'r'), k = required(descriptors, 'k');
  const b = required(descriptors, 'b');
  for (const value of [r, k, b]) {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      throw new TypeError('Primitive finite Numbers are required');
    }
  }
  // Range and derived representability remain the unchanged model's checks.
  return deepFreeze({r, k, b});
}

function copyElderInput(input) {
  const descriptors = ownData(input);
  const n = required(descriptors, 'n'), scale = required(descriptors, 'scale');
  const supplied = required(descriptors, 'values');
  if (typeof n !== 'number' || !Number.isSafeInteger(n) || n < 3 || n > 32
      || typeof scale !== 'bigint' || scale <= 0n || !Array.isArray(supplied)) {
    throw new TypeError('Native finite-grid inputs are required');
  }
  const slots = ownData(supplied, false);
  if (required(slots, 'length') !== n * n) {
    throw new TypeError('Dense square samples are required');
  }
  const values = [];
  for (let vertex = 0; vertex < n * n; vertex++) {
    const value = required(slots, String(vertex));
    if (typeof value !== 'bigint') {
      throw new TypeError('Primitive BigInt samples are required');
    }
    values.push(value);
  }
  return deepFreeze({n, scale, values});
}

// Reconstruct only the selected processed-edge graph. No threshold inference,
// whole-trace model call, model union-find state or event-survivor shortcut.
function selectedPrefix(result, eventEnd) {
  const active = Array(result.values.length).fill(false);
  const adjacency = Array.from({length: result.values.length}, () => []);
  const edges = [];
  let positiveCount = 0, zeroCount = 0, neutralCount = 0;
  for (let index = 0; index < eventEnd; index++) {
    const event = result.events[index];
    if (event.kind === 'activate') {
      active[event.vertex] = true;
    } else {
      const [a, b] = event.edge;
      edges.push([a, b]);
      adjacency[a].push(b);
      adjacency[b].push(a);
      if (event.kind === 'neutral') neutralCount++;
      else if (event.zero) zeroCount++;
      else positiveCount++;
    }
  }
  const activeVertices = [], components = [], seen = new Set();
  for (let vertex = 0; vertex < active.length; vertex++) {
    if (active[vertex]) activeVertices.push(vertex);
  }
  for (const first of activeVertices) {
    if (seen.has(first)) continue;
    const queue = [first], vertices = [];
    seen.add(first);
    for (let offset = 0; offset < queue.length; offset++) {
      const vertex = queue[offset];
      vertices.push(vertex);
      for (const next of adjacency[vertex]) {
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
    vertices.sort((a, b) => a - b);
    let elder = vertices[0];
    for (const vertex of vertices) {
      if (result.values[vertex] > result.values[elder]
          || (result.values[vertex] === result.values[elder] && vertex < elder)) {
        elder = vertex;
      }
    }
    components.push({elder, vertices});
  }
  components.sort((a, b) => a.elder - b.elder);
  return {
    activeVertices, edges, components, positiveCount, zeroCount, neutralCount,
    lastEvent: eventEnd === 0 ? null : result.events[eventEnd - 1],
  };
}

function navigationFor(result, eventEnd) {
  let previousLevel = null, nextLevel = null;
  for (let index = 0; index < result.levels.length; index++) {
    const end = result.levels[index].eventEnd;
    if (end < eventEnd) previousLevel = index;
    if (end > eventEnd && nextLevel === null) nextLevel = index;
  }
  return {
    previousEvent: eventEnd > 0 ? eventEnd - 1 : null,
    nextEvent: eventEnd < result.events.length ? eventEnd + 1 : null,
    previousLevel, nextLevel, reset: eventEnd > 0 ? 0 : null,
  };
}

function selectionFor(choice, result) {
  const descriptors = ownData(choice);
  const mode = required(descriptors, 'mode');
  if (mode === 'reset') {
    closedData(choice, ['mode']);
    return {mode: 'event_prefix', eventEnd: 0};
  }
  if (mode === 'event_prefix') {
    closedData(choice, ['mode', 'eventEnd']);
    const eventEnd = required(descriptors, 'eventEnd');
    if (typeof eventEnd !== 'number' || !Number.isInteger(eventEnd)
        || eventEnd < 0 || eventEnd > result.events.length) {
      throw new TypeError('Valid event prefix is required');
    }
    return {mode, eventEnd};
  }
  if (mode === 'complete_level') {
    closedData(choice, ['mode', 'levelIndex']);
    const levelIndex = required(descriptors, 'levelIndex');
    if (typeof levelIndex !== 'number' || !Number.isInteger(levelIndex)
        || levelIndex < 0 || levelIndex >= result.levels.length) {
      throw new TypeError('Actual complete-level index is required');
    }
    return {mode, levelIndex, eventEnd: result.levels[levelIndex].eventEnd};
  }
  throw new TypeError('Known selection mode is required');
}

export function createReaderController(binding) {
  let source;
  try {
    source = sourceFor(binding);
  } catch {
    throw new TypeError('Invalid reader source binding');
  }
  let committed = null;
  let pending = null;

  function stage(snapshot) {
    const owned = deepFreeze(snapshot);
    const ticket = Object.freeze({});
    const prepared = Object.freeze({kind: 'Ready', ticket, snapshot: owned});
    // Publish the pending slot only after all private construction succeeded.
    pending = {ticket, snapshot: owned};
    return prepared;
  }

  function prepareDocument(document) {
    try {
      const descriptors = closedData(document, ['model', 'input']);
      const model = required(descriptors, 'model');
      const input = required(descriptors, 'input');
      if (model !== 'cubic' && model !== 'elder') {
        return REFUSALS.INVALID_DOCUMENT;
      }
      const ownedInput = model === 'cubic'
        ? copyCubicInput(input) : copyElderInput(input);
      // Exactly one native call site per chosen document, with no replay in
      // selection, commit, current or any view operation.
      const result = deepFreeze(copyResult(model === 'cubic'
        ? cubicModel(ownedInput) : finiteElderTrace(ownedInput)));
      const selection = model === 'cubic'
        ? {mode: 'static'} : {mode: 'event_prefix', eventEnd: 0};
      return stage({
        kind: 'Snapshot',
        revision: (committed === null ? 0n : committed.revision) + 1n,
        documentRevision: (committed === null ? 0n : committed.documentRevision) + 1n,
        model, result, selection,
        selected: model === 'cubic' ? null : selectedPrefix(result, 0),
        navigation: model === 'cubic' ? null : navigationFor(result, 0),
        source,
      });
    } catch {
      return REFUSALS.INVALID_DOCUMENT;
    }
  }

  function prepareSelection(choice) {
    if (committed === null) return REFUSALS.NO_DOCUMENT;
    if (committed.model !== 'elder') return REFUSALS.SELECTION_UNAVAILABLE;
    try {
      const selection = selectionFor(choice, committed.result);
      return stage({
        kind: 'Snapshot',
        revision: committed.revision + 1n,
        documentRevision: committed.documentRevision,
        model: committed.model,
        result: committed.result,
        selection,
        selected: selectedPrefix(committed.result, selection.eventEnd),
        navigation: navigationFor(committed.result, selection.eventEnd),
        source,
      });
    } catch {
      return REFUSALS.INVALID_SELECTION;
    }
  }

  function current() {
    return committed;
  }

  function commit(ticket) {
    if (pending === null || ticket !== pending.ticket) {
      return REFUSALS.STALE_TICKET;
    }
    const snapshot = pending.snapshot;
    pending = null;
    committed = snapshot;
    return snapshot;
  }

  return Object.freeze({prepareDocument, prepareSelection, current, commit});
}
