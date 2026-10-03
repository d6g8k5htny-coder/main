const OPEN_CLASSIFICATIONS = new Set([
  'OPEN_ACTIVE',
  'OPEN_HISTORICAL',
  'AUTHOR_SIDE_CANDIDATE',
  'AUTHOR_SIDE_REDUCTION',
]);

const NON_RESEARCH_KINDS = new Set([
  'engineering',
  'integrity_control',
  'inventory',
  'register',
]);

export const PINNED_GRAPH_SOURCE = Object.freeze({
  repository: 'd6g8k5htny-coder/Math-',
  commit: '7858329974e28be79f29b22644370084ff43da4f',
  capturedAt: '2026-10-03T15:07:03Z',
  path: 'frontiers/downstream_gate_20260925/GRAPH.json',
  bytes: 38753,
  sha256: '8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09',
});

function text(value) {
  if (value === null || value === undefined) return '';
  if (Array.isArray(value)) return value.map(text).join(' ');
  if (typeof value === 'object') return Object.values(value).map(text).join(' ');
  return String(value);
}

function hex(bytes) {
  return [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

export async function parsePinnedGraph(sourceBytes, provenance, cryptoAPI = globalThis.crypto) {
  const bytes = sourceBytes instanceof Uint8Array ? sourceBytes : new Uint8Array(sourceBytes);
  const recorded = provenance?.files?.['GRAPH.json'];
  const provenanceMatches = provenance?.repository === PINNED_GRAPH_SOURCE.repository &&
    provenance?.commit === PINNED_GRAPH_SOURCE.commit &&
    provenance?.captured_at === PINNED_GRAPH_SOURCE.capturedAt &&
    recorded?.path === PINNED_GRAPH_SOURCE.path &&
    recorded?.bytes === PINNED_GRAPH_SOURCE.bytes &&
    recorded?.sha256 === PINNED_GRAPH_SOURCE.sha256;
  if (!provenanceMatches) throw new Error('Pinned graph provenance mismatch.');
  if (bytes.byteLength !== PINNED_GRAPH_SOURCE.bytes)
    throw new Error(`Pinned graph byte-count mismatch: expected ${PINNED_GRAPH_SOURCE.bytes}, received ${bytes.byteLength}.`);
  if (!cryptoAPI?.subtle) throw new Error('SHA-256 verification is unavailable in this browser.');
  const digest = hex(new Uint8Array(await cryptoAPI.subtle.digest('SHA-256', bytes)));
  if (digest !== PINNED_GRAPH_SOURCE.sha256) throw new Error('Pinned graph digest mismatch.');
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error('Pinned graph is not valid JSON.');
  }
}

export function classificationClass(classification) {
  return `classification-${String(classification || 'unknown').toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
}

function displayValue(value) {
  if (Array.isArray(value)) return value.map(displayValue).join('; ');
  if (value && typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function metadataEntries(node) {
  const values = [
    ['Identifier', node.id],
    ['Classification', node.classification],
    ['Layer', node.layer],
    ['Kind', node.kind],
    ['Controlling', node.controlling === true ? 'yes' : node.controlling === false ? 'no' : 'not recorded'],
    ['Recorded impact', `${node.impact} transitive dependent${node.impact === 1 ? '' : 's'}`],
  ];
  const fields = [
    ['source', 'Source'],
    ['fingerprint', 'Fingerprint'],
    ['scope', 'Scope'],
    ['notes', 'Notes'],
    ['author_provider', 'Author provider'],
    ['review_provider', 'Review provider'],
    ['review_providers', 'Review providers'],
    ['review_source', 'Review source'],
    ['review_url', 'Review URL'],
    ['review_issue', 'Review issue'],
    ['review_disposition', 'Review disposition'],
    ['review_basis', 'Review basis'],
    ['technical_checks_non_discharge', 'Technical checks (non-discharge)'],
  ];
  for (const [field, label] of fields) {
    if (node[field] !== undefined && displayValue(node[field]).length)
      values.push([label, displayValue(node[field])]);
  }
  return values;
}

export function validateGraph(graph) {
  const failures = [];
  if (!graph || typeof graph !== 'object' || Array.isArray(graph))
    return ['graph must be an object'];
  if (!graph.nodes || typeof graph.nodes !== 'object' || Array.isArray(graph.nodes))
    failures.push('nodes must be an object');
  if (!Array.isArray(graph.edges)) failures.push('edges must be an array');
  if (!Array.isArray(graph.terminal_classifications))
    failures.push('terminal_classifications must be an array');
  if (failures.length) return failures;

  const ids = new Set(Object.keys(graph.nodes));
  for (const [id, node] of Object.entries(graph.nodes)) {
    if (!id) failures.push('node id must be nonempty');
    if (!node || typeof node !== 'object' || Array.isArray(node)) {
      failures.push(`node ${id} must be an object`);
      continue;
    }
    for (const field of ['classification', 'kind', 'layer'])
      if (typeof node[field] !== 'string' || !node[field])
        failures.push(`node ${id} needs ${field}`);
  }

  const seen = new Set();
  graph.edges.forEach((edge, index) => {
    const label = `edge ${index}`;
    if (!edge || typeof edge !== 'object' || Array.isArray(edge)) {
      failures.push(`${label} must be an object`);
      return;
    }
    if (!ids.has(edge.from)) failures.push(`${label} has unknown from node ${edge.from}`);
    if (!ids.has(edge.to)) failures.push(`${label} has unknown to node ${edge.to}`);
    if (edge.from === edge.to) failures.push(`${label} is a self dependency`);
    if (typeof edge.required !== 'boolean') failures.push(`${label} needs Boolean required`);
    if (typeof edge.relation !== 'string' || !edge.relation)
      failures.push(`${label} needs relation`);
    const key = `${edge.from}\u0000${edge.to}\u0000${edge.relation}`;
    if (seen.has(key)) failures.push(`${label} duplicates ${edge.from} → ${edge.to}`);
    seen.add(key);
  });
  return failures;
}

function transitiveImpact(nodes, startId) {
  const visited = new Set([startId]);
  const queue = [...nodes.get(startId).dependents.map(edge => edge.id)];
  while (queue.length) {
    const id = queue.shift();
    if (visited.has(id)) continue;
    visited.add(id);
    queue.push(...nodes.get(id).dependents.map(edge => edge.id));
  }
  return visited.size - 1;
}

export function buildGraphIndex(graph) {
  const failures = validateGraph(graph);
  if (failures.length) throw new TypeError(failures.join('; '));
  const terminal = new Set(graph.terminal_classifications);
  const nodes = new Map(Object.entries(graph.nodes).map(([id, source]) => [id, {
    id,
    ...source,
    terminal: terminal.has(source.classification),
    dependencies: [],
    dependents: [],
    impact: 0,
    searchable: '',
  }]));
  for (const sourceEdge of graph.edges) {
    const edge = {
      from: sourceEdge.from,
      to: sourceEdge.to,
      id: sourceEdge.to,
      required: sourceEdge.required,
      relation: sourceEdge.relation,
    };
    nodes.get(sourceEdge.from).dependencies.push(edge);
    nodes.get(sourceEdge.to).dependents.push({
      ...edge,
      id: sourceEdge.from,
    });
  }
  for (const node of nodes.values()) {
    node.dependencies.sort((a, b) => a.id.localeCompare(b.id));
    node.dependents.sort((a, b) => a.id.localeCompare(b.id));
    node.impact = transitiveImpact(nodes, node.id);
    node.searchable = text([
      node.id,
      node.layer,
      node.kind,
      node.classification,
      node.source,
      node.notes,
      node.scope,
      node.author_provider,
      node.review_provider,
      node.review_providers,
      node.review_source,
      node.review_url,
      node.review_issue,
      node.review_disposition,
      node.review_basis,
      node.technical_checks_non_discharge,
      node.dependencies.map(edge => [edge.id, edge.relation]),
      node.dependents.map(edge => [edge.id, edge.relation]),
    ]).toLowerCase();
  }
  return {
    graph,
    nodes,
    terminal,
    counts: { nodes: nodes.size, edges: graph.edges.length },
  };
}

function isUnresolvedResearchTarget(node) {
  return OPEN_CLASSIFICATIONS.has(node.classification) &&
    !NON_RESEARCH_KINDS.has(node.kind);
}

export function unresolvedTargets(index) {
  return [...index.nodes.values()]
    .filter(isUnresolvedResearchTarget)
    .sort((a, b) => b.impact - a.impact || a.id.localeCompare(b.id));
}

export function searchNodes(index, query) {
  const needle = String(query || '').trim().toLowerCase();
  if (!needle) return [];
  return [...index.nodes.values()]
    .filter(node => node.searchable.includes(needle))
    .sort((a, b) => a.id.localeCompare(b.id));
}

export function dependencyPaths(index, startId) {
  if (!index.nodes.has(startId)) return [];
  const paths = [];
  const queue = [{ ids: [startId], edges: [] }];
  const shortestDepth = new Map([[startId, 0]]);
  const recordedTargets = new Set();
  while (queue.length) {
    const path = queue.shift();
    const currentId = path.ids.at(-1);
    const current = index.nodes.get(currentId);
    if (path.ids.length > 1 && isUnresolvedResearchTarget(current) && !recordedTargets.has(currentId)) {
      paths.push(path);
      recordedTargets.add(currentId);
    }
    for (const edge of current.dependencies) {
      if (path.ids.includes(edge.id)) continue;
      const nextDepth = path.ids.length;
      if (shortestDepth.has(edge.id) && shortestDepth.get(edge.id) <= nextDepth) continue;
      shortestDepth.set(edge.id, nextDepth);
      queue.push({
        ids: [...path.ids, edge.id],
        edges: [...path.edges, edge],
      });
    }
  }
  return paths.sort((a, b) =>
    a.ids.length - b.ids.length || a.ids.at(-1).localeCompare(b.ids.at(-1)));
}

export function normalizeSelection(index, candidate) {
  const id = String(candidate || '').trim();
  if (!id) return { id: null, error: null };
  if (index.nodes.has(id)) return { id, error: null };
  return {
    id: null,
    error: `Unknown node “${id}”. The saved link does not match this source snapshot.`,
  };
}
