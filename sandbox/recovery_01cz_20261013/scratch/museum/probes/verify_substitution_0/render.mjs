// Render each variant's builder-generated museum.json with the ORIGINAL cd32303 museum.mjs and a
// local-bytes fetcher (global fetch stubbed to throw). Prints the review label and link per card.
import fs from 'node:fs';
const P = '<SCRATCH>/museum/probes/verify_substitution_0';
const T = '<SCRATCH>/museum/cd32303';
globalThis.fetch = async u => { throw Error('NETWORK BLOCKED ' + u); };
const museum = await import(`${T}/docs/site/museum.mjs`);
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._text='';}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??'').join('');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} }
const walk=(n,f)=>{if(!n||typeof n!=='object')return;f(n);for(const c of n.children??[])walk(c,f);};
for (const V of process.argv.slice(2)) {
  const R = `${P}/runs/${V}`;
  const manifest = JSON.parse(fs.readFileSync(`${R}/docs/site/museum.json`, 'utf8'));
  const html = fs.readFileSync(`${T}/docs/site/museum.html`, 'utf8');
  const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new El(m[1])]));
  const document = {createElement: t => new El(t), getElementById: id => nodes.get(id) ?? null};
  const fetcher = async url => {
    const u = String(url); if (u === 'museum.json') return new Response(JSON.stringify(manifest));
    const m = u.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/[0-9a-f]{40}\/(.+)$/);
    const f = m && (m[1] === 'main' ? `${R}/${decodeURIComponent(m[2])}` : `${P}/math/${decodeURIComponent(m[2])}`);
    return f && fs.existsSync(f) ? new Response(fs.readFileSync(f)) : new Response('missing', {status: 404});
  };
  await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  console.log(`== ${V}: ${nodes.get('museum-state').textContent}`);
  for (const host of nodes.get('claim-cards').children) {
    const card = host.children[0]; const id = card.attributes?.['data-object-id'];
    if (!['d2-lifetime-remainder','d5-two-scale','d5-pin-neighborhoods-open'].includes(id)) continue;
    const out = []; let proofHref, reviewHref;
    walk(card, n => { if (/Byte-verified review|Review pointer in the verified index|Unmerged review pointer|OPEN \/ NOT LANDED/.test(n._text)) out.push(n._text);
      if (n.tagName === 'A' && /^Open pinned/.test(n._text)) proofHref = n.href; if (n.tagName === 'A' && n._text === 'Open review ↗') reviewHref = n.href; });
    console.log(`  ${id} [${card.attributes['data-object-class']}]: ${JSON.stringify(out)}\n     proof  link ${proofHref}\n     review link ${reviewHref}${proofHref === reviewHref ? '   <== SAME FILE AS PROOF' : ''}`);
  }
}
