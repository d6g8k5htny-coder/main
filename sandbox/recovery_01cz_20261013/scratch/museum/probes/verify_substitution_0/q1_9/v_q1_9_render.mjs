// Render the builder manifest with the ORIGINAL cd32303 museum.mjs; global fetch stubbed to throw; a local
// fetcher serves main@71400b94 via read-only `git show` and Math- from the byte-verified fixture.
import fs from 'node:fs'; import { execFileSync } from 'node:child_process';
const T = '<SCRATCH>/museum/cd32303';
const [manifestPath, math] = process.argv.slice(2);
globalThis.fetch = async u => { throw Error('NETWORK BLOCKED ' + u); };
const museum = await import(`${T}/docs/site/museum.mjs`);
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._text='';}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??'').join('');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} }
const walk=(n,f)=>{if(!n||typeof n!=='object')return;f(n);for(const c of n.children??[])walk(c,f);};
const manifest = fs.readFileSync(manifestPath);
const html = fs.readFileSync(`${T}/docs/site/museum.html`, 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new El(m[1])]));
const document = {createElement: t => new El(t), getElementById: id => nodes.get(id) ?? null};
const fetcher = async url => { const u = String(url); if (u === 'museum.json') return new Response(manifest);
  const m = u.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/); if (!m) return new Response('x', {status: 404});
  const p = decodeURIComponent(m[3]);
  try { return new Response(m[1] === 'main' ? execFileSync('git', ['-C', '/home/user/main', 'show', `${m[2]}:${p}`]) : fs.readFileSync(`${math}/${p}`)); }
  catch { return new Response('missing', {status: 404}); } };
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
console.log(`state: ${nodes.get('museum-state').textContent}`);
let all = ''; for (const n of nodes.values()) all += n.textContent + '\n';
for (const host of nodes.get('claim-cards').children) {
  const card = host.children[0]; const id = card.attributes?.['data-object-id'];
  if (!['d2-lifetime-remainder','d3-side24-coefficient','d4-fixed-remote-rn'].includes(id)) continue;
  const labels = []; walk(card, n => { if (/^Source label:/.test(n._text)) labels.push(n._text); });
  const t = card.textContent;
  console.log(`${id} [${card.attributes['data-object-class']}]: ${JSON.stringify(labels)}; card mentions FAIL_CLOSED/HOLD_WITH_DOMAIN/landing disposition: ${/FAIL_CLOSED|HOLD_WITH_DOMAIN|landing disposition/i.test(t)}; mentions 'Theorem A' open parent: ${/still-open Theorem A|parent/.test(t)}`);
}
console.log(`whole rendered page mentions FAIL_CLOSED: ${all.includes('FAIL_CLOSED')}, HOLD_WITH_DOMAIN: ${all.includes('HOLD_WITH_DOMAIN')}, 'landing disposition': ${all.includes('landing disposition')} (only inside the D6 PROOF_INDEX quote: ${(all.match(/landing disposition/g)||[]).length} hit(s))`);
