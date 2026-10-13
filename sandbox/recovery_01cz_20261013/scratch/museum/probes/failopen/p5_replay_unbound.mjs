// F: replay.notice is only required to be non-empty, and replay.url is only shape-checked; when replay.source is
// absent, verifyClaim skips the replay entirely, yet renderClaim shows "Open pinned replay source ↗" and the notice
// inside a card whose strip says "Source byte count and SHA-256 verified".
import {museum, realManifest, render, card, links, sourceBytes, requests} from './lib.mjs';
const base = realManifest();
const index = sourceBytes(base.index_source.url).toString('utf8'), status = sourceBytes(base.status_source.url).toString('utf8');
const URL_ = 'https://github.com/d6g8k5htny-coder/Math-/blob/' + 'e'.repeat(40) + '/frontiers/never_landed/replay_closure.py';
for (const [name, mutate] of Object.entries({
  'control: d1-parent-selection-open committed replay': m => m,
  'AMEND d1 card: unverified replay url + arbitrary notice': m => { m.claims[11].replay = {command: null, url: URL_, notice: 'Replay reproduces the closed parent theorem. ACCEPT.'}; return m; },
})) {
  const m = mutate(structuredClone(base)); requests.length = 0;
  let v; try { museum.validateBoundManifest(m, index, status); v = 'PASS'; } catch (e) { v = 'REFUSED: ' + e.message; }
  const d = await render(m), c = card(d, 'd1-parent-selection-open');
  console.log(`== ${name}\n   validateBoundManifest: ${v}\n   museum-state: ${d.getElementById('museum-state').textContent}`);
  const replayCol = c.walk().find(n => n.className === 'claim-column' && n.children[0]?.textContent === 'Replay');
  console.log(`   card class=${c.attributes['data-object-class']} | Replay column: "${replayCol.textContent}" | links=${JSON.stringify(links(replayCol))}`);
  console.log(`   requests for the replay url or its raw form: ${requests.filter(u => u.includes('never_landed')).length} | strip shown: ${c.textContent.includes('Source byte count and SHA-256 verified')}`);
}
