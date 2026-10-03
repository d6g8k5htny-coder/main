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

function text(value) {
  if (value === null || value === undefined) return '';
  if (Array.isArray(value)) return value.map(text).join(' ');
  if (typeof value === 'object') return Object.values(value).map(text).join(' ');
  return String(value);
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
      node.dependencies.map(edge => [edge.id, edge.relation]),
      node.dependents.map(edge => [edge.id, edge.relation]),
    ]).toLocaleLowerCase();
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
  const needle = String(query || '').trim().toLocaleLowerCase();
  if (!needle) return [];
  return [...index.nodes.values()]
    .filter(node => node.searchable.includes(needle))
    .sort((a, b) => a.id.localeCompare(b.id));
}

export function dependencyPaths(index, startId) {
  if (!index.nodes.has(startId)) return [];
  const paths = [];
  const queue = [{ ids: [startId], edges: [] }];
  while (queue.length) {
    const path = queue.shift();
    const currentId = path.ids.at(-1);
    const current = index.nodes.get(currentId);
    if (path.ids.length > 1 && isUnresolvedResearchTarget(current)) {
      paths.push(path);
      continue;
    }
    for (const edge of current.dependencies) {
      if (path.ids.includes(edge.id)) continue;
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

