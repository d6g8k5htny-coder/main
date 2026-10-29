// Native controls only; no source decisions or mathematical joins belong here.
export function enableAtlasView(select, view) {
  // AT can retain the disabled state of the original option objects. Introduce
  // the admitted, enabled control together; no handlers exist on the old one.
  const enabled = select.cloneNode(false);
  enabled.disabled = false;
  enabled.replaceChildren(new Option('Graph', 'graph'), new Option('List', 'list'));
  enabled.value = view === 'list' ? 'list' : 'graph';
  select.replaceWith(enabled);
}

export function reconcileEquationOptions(select, entries, selectedId = select.value) {
  const wanted = [{id: '', label: 'Choose a source equation excerpt'}, ...entries];
  const existing = new Map([...select.options].map(option => [option.value, option]));
  const ids = new Set(wanted.map(entry => entry.id));
  for (const option of [...select.options]) if (!ids.has(option.value)) option.remove();
  wanted.forEach((entry, index) => {
    const option = existing.get(entry.id) || new Option(entry.label, entry.id);
    if (option.text !== entry.label) option.text = entry.label;
    if (select.options[index] !== option) select.insertBefore(option, select.options[index] || null);
  });
  // A removed selection must not silently become the first matching source.
  select.value = ids.has(selectedId) ? selectedId : '';
}

export function createEquationAnnouncement(output, quietMs = 500) {
  let timer, current = '';
  const cancel = () => {
    clearTimeout(timer);
    if (output.textContent) output.textContent = '';
  };
  const publish = () => {
    cancel();
    if (document.hidden || !current) return;
    timer = setTimeout(() => { if (!document.hidden) output.textContent = current; }, quietMs);
  };
  // Hide/page-cache transitions cannot leave an old query queued to speak.
  document.addEventListener('visibilitychange', publish);
  window.addEventListener('pagehide', cancel);
  window.addEventListener('pageshow', publish);
  return text => { current = text; publish(); };
}
