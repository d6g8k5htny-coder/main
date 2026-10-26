// Compare verifies its own publication immediately. This separate source-search
// view keeps the existing current.mjs verifier and loads only when requested.
export function sourcePanelRequested(section, href) {
  const url = new URL(href);
  if (url.searchParams.has('source-q')) return true;
  if (!url.hash || url.hash === '#') return false;
  try {
    const target = section.ownerDocument.getElementById(decodeURIComponent(url.hash.slice(1)));
    return !!target && (target === section || section.contains(target));
  } catch { return false; }
}

export function installCompareSourcePanel(document, window) {
  const section = document.getElementById('current-research');
  const panel = section?.querySelector('.current-source-details');
  const note = document.getElementById('compare-source-note');
  const retry = document.getElementById('compare-source-retry');
  if (!section || !panel || !note || !retry) return;
  let state = 'idle';
  const setState = value => { state = value; section.dataset.compareSource = value; };
  setState('idle');

  async function load() {
    if (state === 'loading' || state === 'ready' || state === 'failed') return;
    setState('loading');
    panel.setAttribute('aria-busy', 'true');
    note.textContent = 'Verifying the separate current-source inventory…';
    try {
      const module = await import('./current.mjs?site-release=fc7e8a98c1e23f476962c4c83e95ed3ce817490007987112d9b2cf6d3d90b142');
      const data = await module.currentSnapshot;
      if (!data) throw new Error('Source inventory verification did not complete.');
      setState('ready');
      note.hidden = true;
    } catch {
      setState('failed');
      note.textContent = 'The separate source inventory is unavailable. Reload to retry its verification; the pinned source links remain available.';
      retry.hidden = false;
    } finally {
      panel.removeAttribute('aria-busy');
    }
  }

  function requested() {
    if (!sourcePanelRequested(section, window.location.href)) return;
    panel.open = true;
    // A source query explicitly requests the search disclosure. Other fragments
    // retain their native target and only reveal the target's own ancestors.
    const url = new URL(window.location.href);
    if (url.searchParams.has('source-q')) section.querySelector('.current-search')?.closest('details')?.setAttribute('open', '');
    else {
      let target;
      try { target = document.getElementById(decodeURIComponent(url.hash.slice(1))); } catch {}
      for (let node = target; node && section.contains(node); node = node.parentElement)
        if (node.tagName === 'DETAILS') node.open = true;
    }
    void load();
  }

  panel.addEventListener('toggle', () => { if (panel.open) void load(); });
  retry.addEventListener('click', () => {
    // current.mjs exports one evaluation promise. A fresh document re-runs its
    // unchanged verifier after either module acquisition or payload failure.
    const url = new URL(window.location.href);
    if (!sourcePanelRequested(section, url.href) || !url.hash) url.hash = 'current-research';
    window.history.replaceState(null, '', url);
    window.location.reload();
  });
  window.addEventListener('hashchange', requested);
  window.addEventListener('popstate', requested);
  requested();
  if (panel.open) void load();
}

if (typeof document !== 'undefined' && typeof window !== 'undefined')
  installCompareSourcePanel(document, window);
