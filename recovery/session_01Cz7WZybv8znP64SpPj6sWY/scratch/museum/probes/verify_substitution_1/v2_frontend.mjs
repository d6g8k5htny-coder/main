// V2b: frontend guard at cd32303. Network is stubbed out; every source is served from local bytes
// and museum.mjs/core.mjs verify them against the pinned SHA-256 before use.
import fs from 'node:fs';
const SCRATCH = '<SCRATCH>/museum';
const TREE = `${SCRATCH}/cd32303`, HERE = `${SCRATCH}/probes/verify_substitution_1`;
globalThis.fetch = async url => { throw Error(`NETWORK BLOCKED: ${url}`); };
const museum = await import(`${TREE}/docs/site/museum.mjs`);
const MATH = 'd6628da09384728992dcbe6e921cc28ba85aebb0', MAIN = '71400b94f6cb354a8cf7aba73ffede2138a64efa';
const local = (repo, commit, p) => {
  const f = repo.endsWith('/Math-') && commit === MATH ? `${HERE}/fixture/Math-/${p}` : repo.endsWith('/main') && commit === MAIN ? `${TREE}/${p}` : null;
  return f && fs.existsSync(f) ? fs.readFileSync(f) : null;
};
const shipped = () => JSON.parse(fs.readFileSync(`${TREE}/docs/site/museum.json`, 'utf8'));
const fetcherFor = (manifest, misses) => async url => {
  const u = String(url);
  if (u === 'museum.json') return new Response(JSON.stringify(manifest));
  const m = u.match(/^https:\/\/raw\.githubusercontent\.com\/(d6g8k5htny-coder\/[^/]+)\/([0-9a-f]{40})\/(.+)$/);
  const b = m && local(m[1], m[2], decodeURIComponent(m[3]));
  if (!b) { misses.add(u.split('/').slice(-1)[0]); return new Response('missing', {status: 404}); }
  return new Response(b);
};
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return this._t+this.children.map(c=>c.textContent).join('');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} }
function doc(){ const html=fs.readFileSync(`${TREE}/docs/site/museum.html`,'utf8');
  const nodes=new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m=>[m[2],new El(m[1])]));
  return {createElement:t=>new El(t),getElementById:id=>nodes.get(id)??null}; }
const walk=(n,f)=>{if(!n||typeof n!=='object')return;f(n);for(const c of n.children??[])walk(c,f);};
const findCard=(root,id)=>{let h=null;walk(root,n=>{if(!h&&n.attributes?.['data-object-id']===id)h=n;});return h;};
const index = fs.readFileSync(`${HERE}/fixture/Math-/PROOF_INDEX.md`,'utf8'), status = fs.readFileSync(`${TREE}/STATUS.md`,'utf8');
const built = JSON.parse(fs.readFileSync(`${HERE}/v2_builder_claims.json`,'utf8'));

async function run(label, mutate, id) {
  const m = shipped(); mutate(m);
  let v; try { museum.validateBoundManifest(m, index, status); v = 'validateBoundManifest ACCEPTS'; } catch (e) { v = `validateBoundManifest REFUSES: ${e.message}`; }
  const misses = new Set(), document = doc();
  await museum.startMuseum({document, fetcher: fetcherFor(m, misses), search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const card = findCard(document.getElementById('claim-cards'), id);
  const texts = [], links = [];
  if (card) walk(card, n => { if (n.tagName === 'A') links.push([n._t, n.href]); else if (n._t && /review|Unmerged|OPEN/i.test(n._t)) texts.push(n._t); });
  const c = m.claims.find(x => x.id === id);
  console.log(`\n[${label}]\n  ${v}\n  card rendered: ${!!card}${card ? '' : ' -- ' + document.getElementById('claim-cards').textContent.slice(0, 160)}`);
  if (card) {
    console.log('  review-related text:', JSON.stringify(texts));
    const rv = links.find(([t]) => /review/i.test(t)), pf = links.find(([t]) => /pinned (full proof|source contract)/i.test(t));
    console.log('  "Open review" ->', rv?.[1], '\n  proof link    ->', pf?.[1], '\n  review link == proof link:', rv?.[1] === pf?.[1]);
  }
  console.log('  sources unavailable offline (unrelated cards):', [...misses].join(',') || 'none');
}
await run('baseline shipped', m => {}, 'd5-two-scale');
await run('control: d5-two-scale review := pr16 review (not in its bullet)', m => { m.claims[5] = built.ctl; }, 'd5-two-scale');
await run('A: d5-two-scale review := its own proof (builder output)', m => { m.claims[5] = built.A; }, 'd5-two-scale');
await run('P: p15-counterexample review := LANDING_CLAIMS.json (builder output)', m => { m.claims[9] = built.P; }, 'p15-demand-one-counterexample');
await run('D5pin: d5-pin-neighborhoods-open review := index_source, pointer_only dropped (hand-edit / builder code change)', m => { m.claims[12].review = structuredClone(m.index_source); }, 'd5-pin-neighborhoods-open');
// Show which sourceContainsPath clause lets D5pin through: the STATUS row only links Math-/blob/main/PROOF_INDEX.md
const row = status.split('\n').find(l => l.startsWith('| **D5 — pin'));
console.log('\nD5 AMEND row contains "](PROOF_INDEX.md)":', row.includes('](PROOF_INDEX.md)'), '| pinned html_url:', row.includes(shipped().index_source.html_url), '| "/PROOF_INDEX.md)":', row.includes('/PROOF_INDEX.md)'), '| mutable link present:', row.includes('Math-/blob/main/PROOF_INDEX.md'));
