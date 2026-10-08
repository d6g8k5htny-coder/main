// Bounded teaching export of Explore's current finite P15 diagram. Source
// custody is provenance; this module performs no mathematical verification.
import { paletteModel } from './explore-models.mjs?site-release=9936ad8dd3c73dd29a736230182b3b8c5e830717f1839dc7941e91c644349b2f';
import { serializeTeachingSVG } from './teaching-export.mjs?site-release=9936ad8dd3c73dd29a736230182b3b8c5e830717f1839dc7941e91c644349b2f';

const PUBLIC_EXPLORE = 'https://d6g8k5htny-coder.github.io/main/site/explore.html';
const SOURCE = Object.freeze({
  repository: 'd6g8k5htny-coder/Math-',
  commit: 'd6628da09384728992dcbe6e921cc28ba85aebb0',
  path: 'frontiers/full_price_20260924/PROOF.md',
  blob: '582180e41dca0ad815ad0f18574df42040912149',
  bytes: 11352,
  sha256: '87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9',
  url: 'https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/frontiers/full_price_20260924/PROOF.md#5-sharpness-and-the-exact-demand-boundary'
});
const LIMITS = 'Illustrative, non-certifying model of one finite capacity-family example in source §5: one original block, a = 1, d = 2, n = 3, K = 2 and no macro edges. A valid assignment partitions the selected objects into at most two capacity-one groups. The full triple is not decomposable; its displayed placement is an attempt, not a valid partition. This example does not establish the full-price theorem, extend to arbitrary families, or close a prize problem. No probabilities, prices or theorem constants are computed. This teaching figure is not mathematical result acceptance. Source identity is recorded provenance; source content and hash verification are not performed by this export.';

function parameters(params) {
  const objects = params?.objects;
  if (!params || typeof params !== 'object' || Array.isArray(params) || !Array.isArray(objects)
    || objects.length > 3 || !Array.from(objects).every(value => Number.isInteger(value) && value >= 1 && value <= 3)
    || new Set(objects).size !== objects.length)
    throw new RangeError('Select a subset of objects 1, 2 and 3');
  return {objects: [...objects].sort((a, b) => a - b)};
}
function modelFor(params) {
  return paletteModel(params.objects.map(value => value - 1));
}
function assignmentsFor(model) {
  return model.parts?.map(part => part.map(value => value + 1)) ?? null;
}
function groupsText(assignments) {
  return assignments.map((part, i) => `Group ${i + 1}: ${part.length ? `object ${part[0]}` : 'empty'}`).join('. ');
}
function summaryFor(model) {
  const names = model.selected.map(value => value + 1);
  return model.decomposable
    ? `${names.length} included ${names.length === 1 ? 'object fits' : 'objects fit'} into the two capacity-one groups. ${groupsText(assignmentsFor(model))}.`
    : 'Three included objects need three places, but only two are available. Removing any one object makes a valid assignment possible.';
}
function permalink(params) {
  const url = new URL(PUBLIC_EXPLORE);
  url.searchParams.set('objects', params.objects.join(','));
  url.hash = 'groups';
  return url.href;
}
export function paletteFigureMetadata(params, generatedAt = new Date().toISOString()) {
  const current = parameters(params);
  if (typeof generatedAt !== 'string' || !Number.isFinite(Date.parse(generatedAt)) || new Date(generatedAt).toISOString() !== generatedAt)
    throw new RangeError('Generation time must be a valid ISO timestamp');
  const model = modelFor(current);
  return {
    schema: 'universal-law/palette-teaching-figure/v1', mode: 'teaching_model',
    params: current, capacity: model.a, groups: model.K, decomposable: model.decomposable,
    assignments: assignmentsFor(model),
    attempted_placement: model.decomposable ? null : {groups: [[1], [2]], unplaced: [3], valid_partition: false},
    generated_at: generatedAt, source: {...SOURCE}, permalink: permalink(current),
    verification: 'not_performed', limits: LIMITS
  };
}
export function paletteFigureCitation(params) {
  const current = parameters(params);
  const model = modelFor(current);
  const placement = model.decomposable
    ? `decomposable. ${groupsText(assignmentsFor(model))}.`
    : 'not decomposable; attempted placement: Group 1: object 1. Group 2: object 2; object 3 has no place. This is not a valid partition.';
  return `Universal Law, “An impossible fit” teaching figure. Selected objects: ${current.objects.length ? current.objects.join(', ') : 'none'}; capacity 1 in each of two groups; ${placement} ${permalink(current)}\nTeaching source: P15 full-price proof, §5 (one-block capacity example), ${SOURCE.url}\n${LIMITS}`;
}

// Match the displayed drawPalette sequence exactly, including its title,
// summary, removed labels, empty groups and failed-triple attempted placement.
// This is a validator for captured nodes, never an alternate export renderer.
export function paletteExpectedNodes(params) {
  const model = modelFor(parameters(params));
  const nodes = [];
  const shape = (tag, attributes, text) => {
    nodes.push({tag, attributes: Object.fromEntries(Object.entries(attributes).map(([key, value]) => [key, String(value)])), ...(text === undefined ? {} : {text})});
  };
  const write = (x, y, text, cls = '') => shape('text', {x, y, class: cls, 'text-anchor': 'middle'}, text);
  shape('title', {id: 'palette-diagram-title'}, 'Assigning objects to two groups');
  shape('desc', {id: 'palette-diagram-description'}, summaryFor(model));
  const centers = [112, 300, 488];
  for (let i = 0; i < 3; i++) {
    const selected = model.selected.includes(i);
    shape('circle', {cx: centers[i], cy: 60, r: 27, class: selected ? 'diagram-object' : 'diagram-muted-object'});
    write(centers[i], 67, String(i + 1));
    if (!selected) write(centers[i], 110, 'removed', 'diagram-small');
  }
  const parts = model.parts || [[0], [1]];
  for (let i = 0; i < 2; i++) {
    const x = 36 + i * 208;
    shape('rect', {x, y: 157, width: 178, height: 90, rx: 12, class: 'diagram-box'});
    write(x + 89, 279, `Group ${i + 1}`);
    if (parts[i].length) {
      const selected = parts[i][0];
      shape('line', {x1: centers[selected], y1: 89, x2: x + 89, y2: 157, class: 'diagram-line'});
      shape('circle', {cx: x + 89, cy: 202, r: 25, class: 'diagram-object'});
      write(x + 89, 209, String(selected + 1));
    } else write(x + 89, 209, 'empty', 'diagram-small');
  }
  if (!model.decomposable) {
    shape('line', {x1: 488, y1: 88, x2: 510, y2: 169, class: 'diagram-boundary'});
    shape('circle', {cx: 510, cy: 203, r: 26, class: 'diagram-point'});
    write(510, 210, '3', 'diagram-on-accent');
    write(510, 279, 'no place', 'diagram-small');
  }
  return nodes;
}
export function serializePaletteSVG(record, params, generatedAt = new Date().toISOString()) {
  const metadata = paletteFigureMetadata(params, generatedAt);
  const placement = metadata.decomposable
    ? `Valid assignment: ${groupsText(metadata.assignments)}.`
    : 'Not decomposable; attempted placement, not a valid partition.';
  const footerLines = [
    `Teaching figure — capacity 1 in each of two groups; objects: ${metadata.params.objects.join(', ') || 'none'}.`,
    placement,
    ...(metadata.decomposable ? [] : ['Group 1: object 1. Group 2: object 2; object 3 has no place.']),
    'One finite example only; no full-price theorem or prize closure.',
    'No extension to arbitrary families; non-certifying; verification not performed.',
    'Source: P15 full-price §5; Math-@d6628da09384; exact identity in metadata.',
    `Generated: ${metadata.generated_at}`,
    'Figure permalink:',
    metadata.permalink
  ];
  return serializeTeachingSVG(record, {
    id: 'palette-diagram', viewBox: '0 0 600 300',
    titleId: 'palette-diagram-title', descriptionId: 'palette-diagram-description',
    expectedNodes: paletteExpectedNodes(metadata.params), metadata, footerLines
  });
}
