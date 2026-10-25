import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const html=fs.readFileSync(new URL('../docs/site/dependencies.html',import.meta.url),'utf8');
const at=id=>html.indexOf(`id="${id}"`);

test('Atlas begins with its task and working search before the source inventory',()=>{
  assert.equal((html.match(/<h1\b/g)||[]).length,1,'The current workbench needs one page heading');
  assert.ok(at('atlas-heading')<at('atlas-search'));
  assert.ok(at('atlas-search')<html.indexOf('<!-- CURRENT-SITE:START -->'),'Search must precede source publication machinery');
  assert.ok(at('atlas-list')<at('atlas-coverage'),'Useful relationships must precede the verbose coverage explanation');
});

test('The selector is the sole view switch while controller compatibility stays present',()=>{
  assert.match(html,/<button\b[^>]*id="atlas-toggle-view"[^>]*\bhidden\b/,'Retain the dereferenced compatibility button without showing a duplicate switch');
  assert.match(html,/<select\b[^>]*id="atlas-view"/);
});

test('Secondary navigation is disclosed and graph-specific tools have a separate scope',()=>{
  const tools=html.match(/<details\b[^>]*class="[^"]*atlas-navigation[^"]*"[^>]*>[\s\S]*?<\/details>/)?.[0]||'';
  for(const id of ['atlas-reset','atlas-back','atlas-forward','atlas-focus','atlas-previous','atlas-next','atlas-zoom-in','atlas-zoom-out'])assert.ok(tools.includes(`id="${id}"`),id+' belongs in the navigation disclosure');
  assert.match(tools,/class="atlas-graph-tools"/,'Zoom and pan instructions must be hidden in List view');
});

test('Neighborhood coverage stays available without delaying the useful graph or list',()=>{
  const coverage=html.match(/<details\b[^>]*class="atlas-coverage-controls"[^>]*>([\s\S]*?)<\/details>/)?.[1]||'';
  assert.ok(coverage.includes('id="atlas-count"'),'Full counts and the edge legend belong in the coverage disclosure');
  assert.match(coverage,/<summary>Neighborhood coverage<\/summary>/);
  assert.match(html,/<p class="atlas-partial-cue">[^<]*partial[^<]*<\/p>/i,'The partial graph scope remains visible');
  assert.match(html,/<button id="atlas-expand"[^>]*aria-label="[^"]*frontier[^"]*"[^>]*>Expand<\/button>/,'A compact expansion button keeps its exact action accessible');
});

test('The historical viewer is explicitly archived with preserved entry fragments',()=>{
  const archive=html.match(/<details\b[^>]*id="dependency-archive"[^>]*>([\s\S]*?)<\/main>/)?.[1]||'';
  assert.match(archive,/<summary[\s\S]*3 October 2026/);
  for(const id of ['snapshot-heading','classification-vocabulary','search-heading','targets-heading','node-detail','object-scope-panel','object-evidence','object-reading-context','object-audit','object-context','boundary-heading'])assert.ok(archive.includes(`id="${id}"`),id+' must remain in the native archive disclosure');
});

test('A closed historical archive cannot conceal a failed source or refused bookmark',()=>{
  for(const id of ['load-status','filter-error','selection-error'])assert.ok(at(id)<at('dependency-archive'),id+' must stay outside the collapsed archive');
});

test('Sharing the selected source snapshot remains a visible reading action',()=>{
  assert.ok(at('atlas-share')<html.indexOf('<details class="atlas-reading-guide"'),'The share action must remain outside the collapsed reading guide');
});

test('Atlas layout loads after current styles without replacing the shared brand',()=>{
  const current=html.indexOf('href="current.css'),layout=html.indexOf('href="atlas-layout.css');
  assert.ok(layout>current,'Atlas layout must follow current.css in the cascade');
  assert.match(html,/<body\b[^>]*class="[^"]*\batlas-page\b/);
});
