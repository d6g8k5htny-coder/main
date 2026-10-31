import test from 'node:test';
import assert from 'node:assert/strict';
import { paletteModel } from '../docs/site/explore-models.mjs';
import { readExploreState } from '../docs/site/explore-state.mjs';

let feature = {};
try { feature = await import('../docs/site/palette-export.mjs'); } catch {}
function api(name) {
  assert.equal(typeof feature[name], 'function', `Palette export ${name} is available`);
  return feature[name];
}
const timestamp = '2026-10-03T12:34:56.000Z';
const sourceURL = 'https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/frontiers/full_price_20260924/PROOF.md#5-sharpness-and-the-exact-demand-boundary';
// Hand-checked partitions of the existing P15 teaching example. In particular,
// the full triple has no partition, even though two objects are drawn in boxes.
const cases = [
  {objects: [], assignments: [[], []], summary: '0 included objects fit into the two capacity-one groups. Group 1: empty. Group 2: empty.'},
  {objects: [1], assignments: [[1], []], summary: '1 included object fits into the two capacity-one groups. Group 1: object 1. Group 2: empty.'},
  {objects: [2], assignments: [[2], []], summary: '1 included object fits into the two capacity-one groups. Group 1: object 2. Group 2: empty.'},
  {objects: [3], assignments: [[3], []], summary: '1 included object fits into the two capacity-one groups. Group 1: object 3. Group 2: empty.'},
  {objects: [1, 2], assignments: [[1], [2]], summary: '2 included objects fit into the two capacity-one groups. Group 1: object 1. Group 2: object 2.'},
  {objects: [1, 3], assignments: [[1], [3]], summary: '2 included objects fit into the two capacity-one groups. Group 1: object 1. Group 2: object 3.'},
  {objects: [2, 3], assignments: [[2], [3]], summary: '2 included objects fit into the two capacity-one groups. Group 1: object 2. Group 2: object 3.'},
  {objects: [1, 2, 3], assignments: null, summary: 'Three included objects need three places, but only two are available. Removing any one object makes a valid assignment possible.'}
];
const themes = {
  dark: {background: 'rgb(7, 17, 31)', surface: 'rgb(14, 27, 46)', raised: 'rgb(19, 36, 59)', text: 'rgb(239, 245, 253)', muted: 'rgb(176, 194, 214)', link: 'rgb(120, 193, 255)', edge: 'rgb(115, 144, 175)', gold: 'rgb(233, 198, 117)'},
  light: {background: 'rgb(246, 248, 252)', surface: 'rgb(255, 255, 255)', raised: 'rgb(237, 242, 249)', text: 'rgb(23, 38, 59)', muted: 'rgb(73, 94, 118)', link: 'rgb(16, 85, 141)', edge: 'rgb(106, 126, 150)', gold: 'rgb(120, 87, 11)'}
};
// A finite fixture of the current drawPalette output, independent of the new
// export module. A wrong selected ID, connector or label must be refused.
function diagram(example = cases.at(-1), theme = 'dark') {
  const colors = themes[theme];
  const defaults = {fill: 'rgb(0, 0, 0)', stroke: 'none', 'fill-opacity': '1', 'stroke-opacity': '1', opacity: '1', 'stroke-width': '1px', 'stroke-dasharray': 'none', 'stroke-linecap': 'butt', 'stroke-linejoin': 'miter', 'font-family': 'system-ui, sans-serif', 'font-size': '16px', 'font-weight': '400', 'font-style': 'normal', 'text-anchor': 'start'};
  const overrides = {
    'diagram-small': {fill: colors.muted, 'font-size': '18px'},
    'diagram-object': {fill: colors.surface, stroke: colors.link, 'stroke-width': '2.5px'},
    'diagram-muted-object': {fill: colors.background, stroke: colors.edge, 'stroke-width': '2px', 'stroke-dasharray': '5px, 5px'},
    'diagram-box': {fill: colors.raised, stroke: colors.edge, 'stroke-width': '2px'},
    'diagram-line': {fill: 'none', stroke: colors.link, 'stroke-width': '2.5px'},
    'diagram-boundary': {fill: 'none', stroke: colors.edge, 'stroke-width': '2px', 'stroke-dasharray': '7px, 6px'},
    'diagram-point': {fill: colors.gold, stroke: colors.background, 'stroke-width': '3px'},
    'diagram-on-accent': {fill: colors.background}
  };
  const node = (tag, attributes, text) => ({tag, attributes: Object.fromEntries(Object.entries(attributes).map(([key, value]) => [key, String(value)])), ...(text === undefined ? {} : {text}), style: ['title', 'desc'].includes(tag) ? {} : {...defaults, ...(tag === 'text' ? {fill: colors.text, 'font-size': '21px', 'text-anchor': 'middle'} : {}), ...overrides[attributes.class]}});
  const label = (x, y, text, cls = '') => node('text', {x, y, class: cls, 'text-anchor': 'middle'}, text);
  const nodes = [node('title', {id: 'palette-diagram-title'}, 'Assigning objects to two groups'), node('desc', {id: 'palette-diagram-description'}, example.summary)];
  for (const [object, x] of [[1, 112], [2, 300], [3, 488]]) {
    const included = example.objects.includes(object);
    nodes.push(node('circle', {cx: x, cy: 60, r: 27, class: included ? 'diagram-object' : 'diagram-muted-object'}), label(x, 67, String(object)));
    if (!included) nodes.push(label(x, 110, 'removed', 'diagram-small'));
  }
  const displayed = example.assignments ?? [[1], [2]];
  for (const [i, x] of [[0, 36], [1, 244]]) {
    nodes.push(node('rect', {x, y: 157, width: 178, height: 90, rx: 12, class: 'diagram-box'}), label(x + 89, 279, `Group ${i + 1}`));
    if (displayed[i].length) {
      const object = displayed[i][0];
      nodes.push(node('line', {x1: [112, 300, 488][object - 1], y1: 89, x2: x + 89, y2: 157, class: 'diagram-line'}), node('circle', {cx: x + 89, cy: 202, r: 25, class: 'diagram-object'}), label(x + 89, 209, String(object)));
    } else nodes.push(label(x + 89, 209, 'empty', 'diagram-small'));
  }
  if (example.assignments === null) nodes.push(node('line', {x1: 488, y1: 88, x2: 510, y2: 169, class: 'diagram-boundary'}), node('circle', {cx: 510, cy: 203, r: 26, class: 'diagram-point'}), label(510, 210, '3', 'diagram-on-accent'), label(510, 279, 'no place', 'diagram-small'));
  return {attributes: {id: 'palette-diagram', viewBox: '0 0 600 300', role: 'img', 'aria-labelledby': 'palette-diagram-title palette-diagram-description'}, background: colors.background, nodes};
}
function metadata(svg) {
  const raw = svg.match(/<metadata[^>]*>([\s\S]*?)<\/metadata>/)?.[1];
  assert.ok(raw, 'Export has machine-readable provenance');
  return JSON.parse(raw.replaceAll('&quot;', '"').replaceAll('&apos;', "'").replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&'));
}

test('all eight selections export the displayed capacity-one assignments with 1-based IDs', () => {
  const build = api('paletteFigureMetadata');
  for (const example of cases) {
    const params = {objects: example.objects};
    const result = build(params, timestamp);
    assert.deepEqual(result.params, params);
    assert.equal(result.capacity, 1);
    assert.equal(result.groups, 2);
    assert.equal(result.decomposable, example.assignments !== null);
    assert.deepEqual(result.assignments, example.assignments);
    assert.deepEqual(paletteModel(example.objects.map(value => value - 1)).parts?.map(part => part.map(value => value + 1)) ?? null, result.assignments);
    assert.equal(result.generated_at, timestamp);
    const reopened = readExploreState(new URL(result.permalink).search);
    assert.deepEqual(reopened.invalid, []);
    assert.deepEqual(reopened.state.objects, example.objects);
    assert.equal(new URL(result.permalink).hash, '#groups');
  }
});

test('the failed triple explicitly records an attempted placement rather than a valid partition', () => {
  const build = api('paletteFigureMetadata');
  const result = build({objects: [1, 2, 3]}, timestamp);
  assert.equal(result.assignments, null);
  assert.deepEqual(result.attempted_placement, {groups: [[1], [2]], unplaced: [3], valid_partition: false});
  for (const example of cases.slice(0, -1)) assert.equal(build({objects: example.objects}, timestamp).attempted_placement, null);
});

test('normalization sorts without mutating input and excludes unrelated state from the permalink', () => {
  const input = {objects: [3, 1], r: 0.25, region: 'remote', url: 'https://evil.test/?private=secret#distance'};
  const result = api('paletteFigureMetadata')(input, timestamp);
  assert.deepEqual(result.params, {objects: [1, 3]});
  assert.deepEqual(result.assignments, [[1], [3]]);
  assert.deepEqual(input.objects, [3, 1]);
  assert.equal(result.permalink, 'https://d6g8k5htny-coder.github.io/main/site/explore.html?objects=1%2C3#groups');
  assert.equal(api('paletteFigureMetadata')({objects: []}, timestamp).permalink, 'https://d6g8k5htny-coder.github.io/main/site/explore.html?objects=#groups');
});

test('malformed selections and noncanonical or false ISO timestamps are refused', () => {
  const build = api('paletteFigureMetadata');
  for (const objects of [undefined, null, '123', 2, [0], [4], [1, 1], ['1'], [1.5], [NaN], [Infinity], [1, 2, 3, 4]]) assert.throws(() => build({objects}, timestamp));
  for (const params of [undefined, null, []]) assert.throws(() => build(params, timestamp));
  for (const time of ['', 'not-a-date', '2026-02-30T00:00:00.000Z', '2026-10-03', '2026-10-03T12:34:56Z']) assert.throws(() => build({objects: []}, time));
});

test('metadata carries the exact pinned finite source and non-certifying scope', () => {
  const result = api('paletteFigureMetadata')({objects: [1, 2, 3]}, timestamp);
  assert.equal(result.schema, 'universal-law/palette-teaching-figure/v1');
  assert.equal(result.mode, 'teaching_model');
  assert.equal(result.verification, 'not_performed');
  assert.deepEqual(result.source, {repository: 'd6g8k5htny-coder/Math-', commit: 'd6628da09384728992dcbe6e921cc28ba85aebb0', path: 'frontiers/full_price_20260924/PROOF.md', blob: '582180e41dca0ad815ad0f18574df42040912149', bytes: 11352, sha256: '87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9', url: sourceURL});
  assert.match(result.limits, /one finite.*example/i);
  assert.match(result.limits, /does not establish.*full.price theorem.*arbitrary famil.*prize/i);
  assert.match(result.limits, /not.*acceptance/i);
  assert.match(result.limits, /source.*verification.*not performed/i);
  assert.equal(Object.hasOwn(result, 'probability'), false);
});

test('plaintext citations identify selected objects, actual empty groups and failed placement', () => {
  const citation = api('paletteFigureCitation');
  const success = citation({objects: [3, 1], url: 'https://evil.test/?private=secret'});
  assert.match(success, /objects: 1, 3/);
  assert.match(success, /Group 1: object 1\. Group 2: object 3/);
  assert.ok(success.includes(sourceURL));
  assert.match(success, /teaching figure/i);
  assert.doesNotMatch(success, /evil|secret|private|<[^>]*>/);
  const empty = citation({objects: []});
  assert.match(empty, /objects: none/);
  assert.match(empty, /Group 1: empty\. Group 2: empty/);
  const triple = citation({objects: [1, 2, 3]});
  assert.match(triple, /not decomposable/i);
  assert.match(triple, /attempted placement.*Group 1: object 1\. Group 2: object 2.*object 3.*no place/i);
  assert.match(triple, /not a valid partition/i);
});

test('expected nodes match every current drawPalette selection including empty labels and failed connectors', () => {
  const expected = api('paletteExpectedNodes');
  for (const example of cases) assert.deepEqual(expected({objects: example.objects}), diagram(example).nodes.map(({style, ...node}) => node));
});

test('SVG preserves displayed geometry and both concrete themes with source metadata and a readable scope footer', () => {
  const serialize = api('serializePaletteSVG');
  for (const theme of ['dark', 'light']) for (const example of cases) {
    const svg = serialize(diagram(example, theme), {objects: example.objects}, timestamp);
    const result = metadata(svg);
    assert.deepEqual(result.assignments, example.assignments);
    assert.equal(result.generated_at, timestamp);
    assert.match(svg, /^<\?xml version="1\.0" encoding="UTF-8"\?>\n<svg/);
    assert.match(svg, /aria-labelledby="palette-diagram-title palette-diagram-description"/);
    assert.match(svg, /viewBox="0 0 600 [4-9]\d\d"/);
    assert.ok(svg.includes(`fill="${themes[theme].background}"`));
    assert.ok(svg.includes(`fill="${themes[theme].raised}"`));
    assert.match(svg, /<rect[^>]*x="36"[^>]*y="157"[^>]*width="178"[^>]*height="90"/);
    assert.match(svg, /Teaching figure.*capacity 1.*two groups/i);
    assert.match(svg, /Source: P15.*§5/);
    assert.match(svg, /non-certifying/);
    assert.doesNotMatch(svg, /<script|foreignObject|\bon[a-z]+=|\bhref=|\sclass=|\sstyle=|var\(|url\(|<style|<a\b/i);
    if (example.assignments === null) {
      assert.match(svg, /<circle[^>]*cx="510"[^>]*cy="203"[^>]*r="26"/);
      assert.match(svg, /attempted placement.*not a valid partition/i);
    }
  }
});

test('stale selection, labels, connectors, arbitrary roots and missing content are refused', () => {
  const serialize = api('serializePaletteSVG');
  assert.throws(() => serialize(diagram(cases[5]), {objects: [1, 2]}, timestamp));
  for (const mutate of [record => { record.nodes[1].text = 'Valid partition'; }, record => { record.nodes.find(node => node.tag === 'line').attributes.x1 = '300'; }, record => { record.nodes.find(node => node.tag === 'text').text = '<script>stale</script>'; }, record => { record.attributes.id = 'another-diagram'; }, record => { record.attributes.viewBox = '0 0 600 301'; }, record => { record.nodes.pop(); }, record => { record.nodes.push({...record.nodes[3]}); }]) {
    const record = diagram(); mutate(record);
    assert.throws(() => serialize(record, {objects: [1, 2, 3]}, timestamp));
  }
});

test('the visible footer records the same ISO generation timestamp as export metadata', () => {
  const svg = api('serializePaletteSVG')(diagram(cases[5]), {objects: [1, 3]}, timestamp);
  assert.equal(metadata(svg).generated_at, timestamp);
  assert.match(svg, /<text[^>]*>Generated: 2026-10-03T12:34:56\.000Z<\/text>/);
});

test('inherited native font stacks on actual non-text primitives remain self-contained', () => {
  const record = diagram(cases[5]);
  for (const node of record.nodes.filter(node => !['title', 'desc', 'text'].includes(node.tag))) node.style['font-family'] = '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';
  const svg = api('serializePaletteSVG')(record, {objects: [1, 3]}, timestamp);
  assert.match(svg, /<circle[^>]*font-family="-apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, system-ui, sans-serif"/);
  assert.doesNotMatch(svg, /url\(|https?:\/\/[^<]*font|@import/);
});

test('unsupported active, resource, nested or hidden content cannot pass a matching geometry export', () => {
  const serialize = api('serializePaletteSVG');
  for (const mutate of [record => { record.nodes[2].attributes.onclick = 'alert(1)'; }, record => { record.nodes[2].attributes.href = 'https://evil.test'; }, record => { record.nodes[2].attributes.class = 'arbitrary-class'; }, record => { record.nodes[2].attributes.transform = 'translate(1)'; }, record => { record.nodes[2].style.fill = 'url(https://evil.test)'; }, record => { record.nodes[2].style.fill = 'var(--ul-text)'; }, record => { record.nodes[2].style.opacity = '0'; }, record => { record.nodes[2].style.filter = 'blur(2px)'; }, record => { record.nodes[2].style.display = 'none'; }, record => { record.nodes[2].nodes = [record.nodes[3]]; }, record => { record.background = 'rgba(7, 17, 31, 0)'; }, record => { record.attributes.style = 'display:none'; }]) {
    const record = diagram(); mutate(record);
    assert.throws(() => serialize(record, {objects: [1, 2, 3]}, timestamp));
  }
});
