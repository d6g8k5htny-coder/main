
import {render, card, links, realManifest} from '../lib.mjs';
const d = await render(realManifest());
console.log('museum-state:', d.getElementById('museum-state').textContent);
for (const id of ['d5-pin-neighborhoods-open','d1-parent-selection-open','sard-g-a1-a6-open']) { const c = card(d, id);
  const rep = c.walk().find(n => n.className === 'claim-column' && n.children[0]?.textContent === 'Replay');
  console.log(id, '| class', c.attributes['data-object-class'], '|', JSON.stringify(links(c)[0]), '| NO COMPLETE PROOF YET:', c.textContent.includes('NO COMPLETE PROOF YET'), '| Replay:', JSON.stringify(rep.textContent)); }
