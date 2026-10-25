import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const site=new URL('../docs/site/',import.meta.url);
const page=name=>fs.readFileSync(new URL(name+'.html',site),'utf8');

test('Home explains the landscape before publication machinery',()=>{
  const html=page('index');
  assert.ok(html.indexOf('id="the-idea"')<html.indexOf('<!-- CURRENT-SITE:START -->'),'The short explanation must precede source inventory');
  const hero=html.slice(html.indexOf('class="hero'),html.indexOf('<!-- CURRENT-SITE:START -->'));
  assert.match(hero,/href="explore.html"/);assert.match(hero,/href="research.html"/);
  assert.match(hero,/<svg[^>]+aria-labelledby="landscape-title landscape-desc"/);
});

test('Explore reaches a real local control and result before source publication',()=>{
  const html=page('explore'),source=html.indexOf('<!-- CURRENT-SITE:START -->');
  for(const id of ['curvature-center','curvature-diagram','curvature-summary'])assert.ok(html.indexOf(`id="${id}"`)<source,id+' must precede publication details');
  const lesson=html.slice(html.indexOf('<section id="peaks"'),html.indexOf('<section id="distance"'));
  assert.ok(lesson.indexOf('class="experiment"')<lesson.indexOf('At a stationary point'),'Begin with the working example, then explain the geometry');
});

test('Research opens with questions, explicit editorial date and an accessible preserved archive',()=>{
  const html=page('research');
  assert.ok(html.indexOf('id="reading-questions"')>0 && html.indexOf('id="reading-questions"')<html.indexOf('id="reading-cut-20261010"'),'Question index must precede long historical cuts');
  const first=html.slice(html.indexOf('<main'),html.indexOf('id="reading-questions"'));
  assert.match(first,/<time datetime="2026-10-10T20:24:00Z">/);
  assert.match(first,/editorial|Editorial/);
  assert.match(html,/<details\b[^>]*data-reader-archive[^>]*open[^>]*>[\s\S]*<section id="reading-cut-20261010"/,'Static/no-script reading cuts stay readable; the shell progressively collapses their archive');
});

test('Research reading map routes to all four preserved introductory paths',()=>{
  const html=page('research');
  const guide=html.slice(html.indexOf('<details id="reading-guide"'),html.indexOf('<details id="reading-archive"'));
  const navigation=guide.match(/<nav\b[^>]*aria-label="Introductory reading paths"[^>]*>([\s\S]*?)<\/nav>/);
  assert.ok(navigation,'The reading map must offer its four introductory paths');
  const destinations=[...navigation[1].matchAll(/<a\b[^>]*href="([^"]+)"[^>]*>/g)].map(match=>match[1]);
  assert.deepEqual(destinations,['#lifetimes','#peaks','#neighbors','#finite']);
  for(const destination of destinations)assert.ok(html.includes(`<section id="${destination.slice(1)}"`),'Preserved target: '+destination);
});

test('Library routes present-day discovery separately from dated records',()=>{
  const html=page('workspace'),hero=html.slice(html.indexOf('<main'),html.indexOf('<!-- CURRENT-SITE:START -->'));
  for(const route of ['nodes/index.html#sources','#inventory','#board','#coefficient','#contribute'])assert.ok(hero.includes(`href="${route}"`),route);
  assert.ok(html.indexOf('<section id="inventory"')<html.indexOf('<section id="board"'),'Readers reach the searchable catalog before the dated status board');
  const contribute=html.slice(html.indexOf('<section id="contribute"'));
  assert.match(contribute,/issues\/539/);assert.doesNotMatch(contribute,/issues\/475">active coordination board/);
});

test('Deep archived fragments reveal every ancestor without focus or history writes',async()=>{
  assert.ok(fs.existsSync(new URL('reader-shell.mjs',site)),'A shared disclosure fragment restorer is needed');
  const {revealReaderFragment,prepareReaderArchives}=await import('../docs/site/reader-shell.mjs');
  const archive={tagName:'DETAILS',open:true,parentElement:null};
  const source={tagName:'DETAILS',open:false,parentElement:archive};
  const target={tagName:'SECTION',parentElement:source,scrollIntoView(){this.scrolled=true;},focus(){assert.fail('Must not steal focus');}};
  const other={tagName:'DETAILS',open:true,parentElement:null};
  const document={getElementById:id=>id==='old proof'?target:null,querySelectorAll:()=>[archive,other]};
  prepareReaderArchives(document,'#old%20proof');
  assert.equal(archive.open,true);assert.equal(other.open,false);
  assert.equal(revealReaderFragment(document,'#old%20proof'),true);
  assert.equal(source.open,true);assert.equal(archive.open,true);assert.equal(target.scrolled,true);
  target.scrolled=false;
  assert.equal(revealReaderFragment(document,'#old%20proof'),false,'Already visible targets use native fragment scrolling');
  assert.equal(target.scrolled,false);
  for(const hash of ['', '#unknown', '#%E0%A4%A'])assert.equal(revealReaderFragment(document,hash),false);
});
