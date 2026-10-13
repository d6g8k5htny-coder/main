// v2_render.mjs <worktree> <manifest.json> : run <worktree>/docs/site/museum.mjs startMuseum() with NO network.
// Strict raw.githubusercontent.com emulation: a main URL /<repo>/<commit>/<path> is answered ONLY by
// `git show <commit>:<path>` (worktree .git -> alternate of /home/user/main objects), 404 otherwise; a Math- URL
// only at commit d6628da, from ./fixture/Math-. Any other URL -> 404. Prints what the viewer would show.
import fs from 'node:fs'; import {execFileSync} from 'node:child_process';
const HERE = new URL('.', import.meta.url).pathname; const [wt, manifestPath] = process.argv.slice(2);
const museum = await import(`file://${wt}/docs/site/museum.mjs`);
const manifestText = fs.readFileSync(manifestPath, 'utf8'), manifest = JSON.parse(manifestText);
const git = (...a) => execFileSync('git', [`--git-dir=${wt}/.git`, ...a], {stdio: ['ignore', 'pipe', 'ignore'], env: {...process.env, GIT_OPTIONAL_LOCKS: '0'}});
const requests = [];
const fetcher = async url => { url = String(url);
  if (url === 'museum.json') return new Response(manifestText);
  const m = url.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  let bytes = null;
  if (m && m[1] === 'main') { try { bytes = git('show', `${m[2]}:${decodeURIComponent(m[3])}`); } catch { bytes = null; } }
  else if (m && m[2] === 'd6628da09384728992dcbe6e921cc28ba85aebb0') { const f = `${HERE}fixture/Math-/${decodeURIComponent(m[3])}`; if (fs.existsSync(f)) bytes = fs.readFileSync(f); }
  requests.push(`${bytes ? 200 : 404} ${m ? m[1] + '@' + m[2].slice(0, 8) + ':' + decodeURIComponent(m[3]) : url}`);
  return bytes ? new Response(bytes) : new Response('', {status: 404}); };
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._text='';}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const html = fs.readFileSync(`${wt}/docs/site/museum.html`, 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new El(m[1])]));
const document = {createElement: t => new El(t), createTextNode: v => ({textContent: v}), getElementById: id => nodes.get(id) ?? null};
let validated = 'n/a';
try { const st = git('show', `${manifest.status_source.commit}:STATUS.md`).toString(); const ix = fs.readFileSync(`${HERE}fixture/Math-/PROOF_INDEX.md`, 'utf8');
  museum.validateBoundManifest(structuredClone(manifest), ix, st); validated = 'ACCEPTED'; } catch (e) { validated = 'REFUSED: ' + e.message; }
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
const snap = manifest.status_source.commit, tree = git('ls-tree', '-d', '--name-only', snap, 'incoming/').toString().trim().split('\n');
console.log(`  manifest status_source.commit=${snap.slice(0, 8)} packets=${JSON.stringify(manifest.packets.map(p => p.id + '@' + p.result.commit.slice(0, 8)))}`);
console.log(`  git ls-tree -d ${snap.slice(0, 8)} incoming/ = ${JSON.stringify(tree)}`);
console.log(`  validateBoundManifest(manifest, PROOF_INDEX, STATUS@${snap.slice(0, 8)}) -> ${validated}`);
console.log(`  404 responses: ${requests.filter(r => r.startsWith('404')).length} of ${requests.length}`);
console.log(`  [museum-state] ${nodes.get('museum-state').textContent}`);
console.log(`  [claim-cards] ${nodes.get('claim-cards').children.length} cards`);
console.log(`  [packet-cards] ${nodes.get('packet-cards').children.length} child(ren): ${nodes.get('packet-cards').textContent.replace(/\s+/g, ' ').slice(0, 160)}`);
