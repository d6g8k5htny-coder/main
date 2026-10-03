import {
  buildGraphIndex,
  classificationClass,
  dependencyPaths,
  metadataEntries,
  normalizeSelection,
  parsePinnedGraph,
  searchNodes,
  unresolvedTargets,
} from './dependency-model.mjs';

const byId = id => document.getElementById(id);

// Native fragment navigation happens before the verified graph changes the
// page height. Correct that initial position only after rendering, and yield
// permanently to any reader interaction or subsequent browser navigation.
function prepareInitialSelectionRestore({ windowObject = window, documentObject = document } = {}) {
  const hash = windowObject.location.hash;
  const node = new URLSearchParams(windowObject.location.search).get('node');
  if (hash !== '#node-detail' || !node || !windowObject.addEventListener)
    return { finish: () => false, cancel: () => {} };
  const events = ['wheel', 'touchstart', 'touchmove', 'keydown', 'pointerdown', 'pointermove', 'focusin', 'hashchange', 'popstate', 'pagehide'];
  const options = { capture: true, passive: true };
  let cancelled = false;
  let finished = false;
  const cleanup = () => events.forEach(event => windowObject.removeEventListener(event, cancel, options));
  const cancel = event => {
    if (event.type === 'pointermove' && !event.buttons) return;
    if (event.type === 'focusin' && event.target === documentObject.getElementById('node-detail')) return;
    cancelled = true;
    cleanup();
  };
  events.forEach(event => windowObject.addEventListener(event, cancel, options));
  return {
    cancel,
    finish() {
      if (finished) return false;
      finished = true;
      windowObject.requestAnimationFrame(() => {
        cleanup();
        const currentNode = new URLSearchParams(windowObject.location.search).get('node');
        if (cancelled || windowObject.location.hash !== hash || currentNode !== node) return;
        const target = documentObject.getElementById('node-detail');
        if (!target?.scrollIntoView) return;
        target.focus?.({ preventScroll: true });
        target.scrollIntoView({ block: 'start', behavior: 'instant' });
      });
      return true;
    },
  };
}

function element(tag, options = {}) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  return node;
}

function buttonFor(node, choose) {
  const button = element('button', { className: `node-button ${classificationClass(node.classification)}` });
  button.type = 'button';
  const id = element('strong', { text: node.id });
  const context = element('span', { text: `${node.classification} · ${node.layer} · ${node.kind}` });
  button.append(id, context);
  button.addEventListener('click', () => choose(node.id));
  return button;
}

function relationItem(edge, choose) {
  const item = element('li');
  const button = element('button', { text: edge.id });
  button.type = 'button';
  button.addEventListener('click', () => choose(edge.id));
  const note = element('span', { className: 'edge-note' });
  const kind = element('span', { className: 'edge-kind', text: edge.required ? 'required' : 'contextual' });
  note.append(kind, document.createTextNode(` · ${edge.relation}`));
  item.append(button, note);
  return item;
}

function emptyItem(message) {
  const item = element('li', { className: 'muted', text: message });
  return item;
}

function metadata(node) {
  const fragment = document.createDocumentFragment();
  for (const [label, value, href] of metadataEntries(node)) {
    const wrapper = element('div');
    const description = element('dd');
    if (href) {
      const link = element('a', { text: value });
      link.href = href;
      description.append(link);
    } else {
      description.textContent = value;
    }
    wrapper.append(element('dt', { text: label }), description);
    fragment.append(wrapper);
  }
  return fragment;
}

function renderPath(path) {
  const item = element('li');
  path.ids.forEach((id, index) => {
    if (index) {
      const edge = path.edges[index - 1];
      const arrow = element('span', {
        className: 'path-arrow',
        text: `→ ${edge.required ? 'requires' : 'references'} (${edge.relation}) →`,
      });
      item.append(arrow);
    }
    item.append(element('span', { className: 'path-step', text: id }));
  });
  return item;
}

async function load() {
  const [graphResponse, provenanceResponse] = await Promise.all([
    fetch('./dependency-source/GRAPH.json'),
    fetch('./dependency-source/PROVENANCE.json'),
  ]);
  if (!graphResponse.ok || !provenanceResponse.ok)
    throw new Error('A pinned source file could not be loaded.');
  const [graphBytes, provenance] = await Promise.all([
    graphResponse.arrayBuffer(),
    provenanceResponse.json(),
  ]);
  const graph = await parsePinnedGraph(graphBytes, provenance);
  return { index: buildGraphIndex(graph), provenance };
}

function run({ index, provenance }) {
  const targetList = unresolvedTargets(index);
  byId('node-count').textContent = String(index.counts.nodes);
  byId('edge-count').textContent = String(index.counts.edges);
  byId('unresolved-count').textContent = String(targetList.length);
  byId('load-status').textContent = `Pinned Math commit ${provenance.commit.slice(0, 12)} · captured ${provenance.captured_at}`;

  const search = byId('dependency-search');
  const searchResults = byId('search-results');
  const searchSummary = byId('search-summary');
  const targetRoot = byId('unresolved-targets');
  const detailHeading = byId('detail-heading');
  const detailContent = byId('detail-content');
  const detailEmpty = byId('detail-empty');
  const selectionError = byId('selection-error');
  const clearSelection = byId('clear-selection');

  function updateURL(id) {
    const url = new URL(window.location.href);
    if (id) {
      url.searchParams.set('node', id);
      url.hash = 'node-detail';
    } else {
      url.searchParams.delete('node');
      url.hash = '';
    }
    window.history.replaceState({ node: id }, '', url);
  }

  function renderSelection(candidate, { writeURL = true, focus = true } = {}) {
    const selection = normalizeSelection(index, candidate);
    selectionError.hidden = !selection.error;
    selectionError.textContent = selection.error || '';
    if (!selection.id) {
      detailHeading.textContent = selection.error ? 'Saved selection unavailable' : 'Choose a claim to inspect';
      detailContent.hidden = true;
      detailEmpty.hidden = Boolean(selection.error);
      clearSelection.hidden = !selection.error;
      if (writeURL && !selection.error) updateURL(null);
      if (focus) byId('node-detail').focus({ preventScroll: true });
      return;
    }
    const selected = index.nodes.get(selection.id);
    detailHeading.textContent = selected.id;
    detailEmpty.hidden = true;
    detailContent.hidden = false;
    clearSelection.hidden = false;
    byId('node-metadata').replaceChildren(metadata(selected));
    byId('dependency-list').replaceChildren(...(
      selected.dependencies.length
        ? selected.dependencies.map(edge => relationItem(edge, choose))
        : [emptyItem('No direct dependencies are recorded in this snapshot.')]
    ));
    byId('dependent-list').replaceChildren(...(
      selected.dependents.length
        ? selected.dependents.map(edge => relationItem(edge, choose))
        : [emptyItem('No direct dependents are recorded in this snapshot.')]
    ));
    const paths = dependencyPaths(index, selected.id);
    byId('dependency-paths').replaceChildren(...(
      paths.length ? paths.map(renderPath) : [emptyItem('No required path to another unresolved target is recorded below this node.')]
    ));
    if (writeURL) updateURL(selected.id);
    if (focus) byId('node-detail').focus({ preventScroll: true });
  }

  function choose(id) {
    renderSelection(id);
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    byId('node-detail').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
  }

  for (const target of targetList) {
    const item = element('li', { className: classificationClass(target.classification) });
    const content = element('div');
    const button = element('button', { text: target.id });
    button.type = 'button';
    button.addEventListener('click', () => choose(target.id));
    content.append(button, element('small', { text: `${target.classification} · ${target.layer} · ${target.kind}` }));
    const impact = element('span', { className: 'impact', text: `${target.impact} downstream` });
    item.append(content, impact);
    targetRoot.append(item);
  }

  function renderSearch() {
    const query = search.value.trim();
    const results = searchNodes(index, query);
    searchResults.replaceChildren(...results.map(node => {
      const item = element('li');
      item.append(buttonFor(node, choose));
      return item;
    }));
    searchSummary.textContent = !query
      ? 'Enter a claim ID, source, status, note, or relation.'
      : results.length
        ? `${results.length} result${results.length === 1 ? '' : 's'}.`
        : 'No node in this snapshot matches that search.';
  }
  search.disabled = false;
  search.addEventListener('input', renderSearch);
  renderSearch();

  clearSelection.addEventListener('click', event => {
    event.preventDefault();
    renderSelection(null);
  });
  window.addEventListener('popstate', () => {
    renderSelection(new URLSearchParams(window.location.search).get('node'), { writeURL: false, focus: false });
  });
  renderSelection(new URLSearchParams(window.location.search).get('node'), { writeURL: false, focus: false });
}

const initialSelectionRestore = prepareInitialSelectionRestore();
load().then(data => {
  run(data);
  initialSelectionRestore.finish();
}).catch(error => {
  initialSelectionRestore.cancel({ type: 'load-error' });
  byId('load-status').textContent = 'The local graph could not be loaded.';
  const notice = byId('selection-error');
  notice.hidden = false;
  notice.textContent = error instanceof Error ? error.message : 'The local graph could not be loaded.';
});
