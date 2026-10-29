import {revealReaderFragment} from './reader-shell.mjs?site-release=1807a8b9c16f5c2dbe8a6e77d0d195dc69602870ee01cbb0dc2fcb2fbc335add';

// Reading aids only. The dated HTML, source pins and scientific scope stay intact.
const READINGS = Object.freeze([
  ['reading-cut-20261010', ['reading-cut-20261010-details']],
  ['reading-cut-20261009', ['reading-cut-20261009-details']],
  ['reading-cut-20261007', ['reading-cut-20261007-details']],
  ['shrinking-bin-sampling', ['sampling-details']],
  ['pair-endpoint-rate', ['pair-endpoint-details']],
  ['reading-addendum', ['bar-occurrence-density-details', 'designated-pair-failure-details']],
  ['latest-work', ['latest-work-details']],
]);

function targetMatches(document, id) {
  // IDs come from the fixed navigation table, never reader input.
  return document.querySelectorAll(`[id="${id}"]`);
}
function uniqueTarget(document, id) {
  const matches = targetMatches(document, id);
  return matches.length === 1 ? matches[0] : null;
}

function anchor(document, target, label) {
  const link = document.createElement('a');
  link.href = `#${target}`;
  link.textContent = label;
  return link;
}

export function installReadingNavigation(document) {
  const questions = uniqueTarget(document, 'reading-questions');
  const map = uniqueTarget(document, 'reading-guide');
  const archive = uniqueTarget(document, 'reading-archive');
  if (!questions || !map || !archive) return 0;
  let installed = 0;
  for (const [id, sourceIds] of READINGS) {
    const section = uniqueTarget(document, id);
    if (!section || section.tagName !== 'SECTION' || !section.classList.contains('latest-work')
      || !archive.contains(section) || section.querySelector(':scope > .reading-jump')) continue;
    const heading = section.querySelector(':scope > .section-heading');
    const sources = [...section.querySelectorAll('details.latest-identities')];
    if (!heading || sources.length !== sourceIds.length) continue;
    // The oldest preserved disclosure has no HTML ID. Assign its explicit local
    // navigation ID at runtime; never rewrite its historical source text.
    if (sources.some((source, i) => source.id
      ? source.id !== sourceIds[i] || uniqueTarget(document, sourceIds[i]) !== source
      : targetMatches(document, sourceIds[i]).length !== 0)) continue;
    const nav = document.createElement('nav');
    nav.className = 'reading-jump';
    nav.setAttribute('aria-label', `Reading navigation: ${section.querySelector('time')?.textContent.trim() || id}`);
    nav.append(anchor(document, 'reading-questions', 'Questions'), anchor(document, 'reading-guide', 'Reading map'));
    sources.forEach((source, i) => {
      source.id ||= sourceIds[i];
      source.tabIndex = -1;
      nav.append(anchor(document, sourceIds[i], sources.length === 1 ? 'Exact sources' : `Exact sources ${i + 1}`));
    });
    const end = document.createElement('p');
    end.className = 'reading-return';
    end.append(anchor(document, 'reading-questions', '↑ Back to questions'));
    questions.tabIndex = -1;
    map.tabIndex = -1;
    heading.after(nav);
    section.append(end);
    installed += 1;
  }
  return installed;
}

if (typeof document !== 'undefined') {
  installReadingNavigation(document);
  // The oldest source disclosure receives its ID during enhancement. Restore
  // that bookmark through the existing archive revealer after assigning it.
  if (typeof window !== 'undefined') revealReaderFragment(document, window.location.hash);
}
