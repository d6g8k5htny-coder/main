// Renders a tampered museum.json through startMuseum from the tree in $MUSEUM_TREE (original cd32303 bytes or a
// scratch mutant) and prints what the page shows. usage: MUSEUM_TREE=<tree> node p3_render_tampered.mjs <tamper>
import {realManifest, render, card, links, sourceBytes} from './lib.mjs';
const tampers = {
  // AMEND row promoted to ACCEPT in the manifest
  promote_amend: m => { Object.assign(m.claims[11], {class: 'ACCEPT-scoped', source_label: 'ACCEPT'}); return [m, 'd1-parent-selection-open']; },
  // D6's byte-verified review shown as D5 all-height annulus's review (cross-claim review substitution)
  swap_review: m => { m.claims[3].review = structuredClone(m.claims[10].review); return [m, 'd5-all-height-annulus']; },
  // proof descriptor claims a digest the served bytes do not have
  bad_proof_digest: m => { m.claims[4].proof.sha256 = '0'.repeat(64); return [m, 'd5-height-window-annulus']; },
};
const name = process.argv[2];
const [m, focus] = tampers[name](structuredClone(realManifest()));
const d = await render(m);
const c = card(d, focus);
console.log(`[${process.env.MUSEUM_TREE ? 'mutant ' + process.env.MUSEUM_TREE.split('/').pop() : 'original cd32303'}] tamper=${name}`);
console.log('  museum-state:', d.getElementById('museum-state').textContent);
const cards = d.getElementById('claim-cards');
console.log('  rendered claim cards:', cards.walk().filter(n => n.attributes?.['data-object-id']).length, '| refusal cards:', cards.walk().filter(n => n.className === 'museum-card refused').length);
if (c) console.log(`  ${focus}: data-object-class=${c.attributes['data-object-class']} | ${JSON.stringify(links(c).find(l => l.startsWith('Open review')) || links(c)[0])} | strip="${c.textContent.includes('Source byte count and SHA-256 verified') ? 'Source byte count and SHA-256 verified…' : ''}"`);
else console.log(`  ${focus}: refused (no card)`);
