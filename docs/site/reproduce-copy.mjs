// Copies visible source-pinned commands only. It never executes a calculation.
export function wireCommandCopy({button, block, status, clipboard}) {
  const fallback = 'Select the command block above and copy it manually. No commands have been run.';
  if (typeof clipboard?.writeText !== 'function') {
    status.textContent = fallback;
    return;
  }
  button.addEventListener('click', async () => {
    button.disabled = true;
    status.textContent = 'Copying commands…';
    try {
      await clipboard.writeText(block.textContent);
      status.textContent = 'Copied all commands. They have not run; paste them into your own terminal when ready.';
    } catch {
      status.textContent = fallback;
    } finally {
      button.disabled = false;
    }
  });
  button.disabled = false;
}

export function buildCoefficientExports(intervals, source) {
  if (!/^https:\/\/github\.com\/d6g8k5htny-coder\/Math-\/blob\/[a-f0-9]{40}\/coefficients\/side24_v1\/coefficient\.py$/.test(source)) throw new Error('A pinned coefficient source is required.');
  if (!Array.isArray(intervals) || !intervals.length) throw new Error('No recorded enclosures available.');
  const seen = new Set();
  const records = intervals.map(({dimension, lower, upper}) => {
    if (typeof dimension !== 'string' || !/^[23]$/.test(dimension) || seen.has(dimension)) throw new Error('Unsupported or duplicate dimension.');
    seen.add(dimension);
    for (const endpoint of [lower, upper]) if (typeof endpoint !== 'string' || !/^0\.[0-9]{1,100}$/.test(endpoint)) throw new Error('Endpoints must be exact decimal strings.');
    const scale = Math.max(lower.length, upper.length) - 2;
    const integer = value => BigInt(value.slice(2).padEnd(scale, '0'));
    if (integer(lower) >= integer(upper)) throw new Error('Empty or reversed strict enclosure.');
    return {dimension, lower, upper};
  });
  const boundary = 'Historical pinned coefficient enclosures. No calculation has run here; this export does not establish the parent lifetime theorem or elder pairing.';
  return {
    text: `${boundary}\n${records.map(r => `Dimension ${r.dimension}: ${r.lower} < c_${r.dimension},24 < ${r.upper}`).join('\n')}\nSource: ${source}`,
    json: JSON.stringify({schema:'universal-law/coefficient-enclosures/v1', source, source_state:'historical_pinned', execution:'not_performed', scope:boundary, coefficient:'c_{d,24}', bounds:'strict', intervals:records}, null, 2),
    latex: `% ${boundary}\n% Source: ${source}\n${records.map(r => `\\[${r.lower} < c_{${r.dimension},24} < ${r.upper}\\]`).join('\n')}`
  };
}

export function wireCoefficientCopy({buttons, status, exports, clipboard}) {
  const all = Object.values(buttons);
  const enabled = typeof clipboard?.writeText === 'function';
  let pending = false;
  const fallback = 'Select an export below and copy it manually. These are historical enclosures; no calculation has run here.';
  status.textContent = enabled ? 'Copy the historical enclosures with their pinned source. No calculation has run here.' : fallback;
  for (const [format, button] of Object.entries(buttons)) {
    button.disabled = !enabled;
    button.addEventListener('click', async () => {
      if (!enabled || pending || button.disabled) return;
      pending = true; all.forEach(b => {b.disabled = true;});
      status.textContent = 'Copying coefficient enclosures…';
      try { await clipboard.writeText(exports[format]); status.textContent = `Copied ${format === 'latex' ? 'LaTeX' : format === 'json' ? 'JSON' : 'text'} historical enclosures. No calculation has run here.`; }
      catch { status.textContent = fallback; }
      finally { pending = false; all.forEach(b => {b.disabled = false;}); }
    });
  }
}

if (typeof document !== 'undefined') {
  wireCommandCopy({button:document.getElementById('copy-commands'), block:document.getElementById('reproduction-commands'), status:document.getElementById('copy-commands-status'), clipboard:globalThis.navigator?.clipboard});
  const panel = document.getElementById('coefficient-exports');
  if (panel) {
    try {
      const intervals = [...document.querySelectorAll('#coefficient-enclosures tbody tr')].map(row => ({dimension:row.querySelector('th').textContent.trim(), lower:row.querySelectorAll('code')[0].textContent, upper:row.querySelectorAll('code')[1].textContent}));
      const exports = buildCoefficientExports(intervals, document.getElementById('coefficient-source').href);
      const buttons = {};
      for (const format of ['text','json','latex']) {
        document.getElementById(`coefficient-export-${format}`).textContent = exports[format];
        buttons[format] = document.getElementById(`copy-coefficients-${format}`);
      }
      wireCoefficientCopy({buttons, status:document.getElementById('copy-coefficients-status'), exports, clipboard:globalThis.navigator?.clipboard});
      panel.hidden = false;
    } catch {
      document.getElementById('copy-coefficients-status').textContent = 'Exports unavailable: recorded identity could not be read. The original table remains available; no calculation has run here.';
      panel.hidden = false;
    }
  }
}
