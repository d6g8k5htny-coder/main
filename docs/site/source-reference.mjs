/** Build syntax-checked source references locally. No source existence or review claim. */
const REPOSITORIES = new Set(['main', 'Math-', 'query-', 'Universal-Law-Workspace']);
// Source validation forbids backslashes and percent escapes; accepted TeX specials
// are rendered literally, without interpreting source paths as TeX commands.
// BibTeX counts even escaped braces, so literal brace macros keep database depth balanced.
const bibtexLiteral = value => value.replace(/[#$&{}_~^]/gu, character => ({
  '#': '\\#', '$': '\\$', '&': '\\&', '{': '\\textbraceleft{}', '}': '\\textbraceright{}', '_': '\\_',
  '~': '\\textasciitilde{}', '^': '\\textasciicircum{}'
})[character]);
export function buildReference(repository, commit, path = '', sha256 = '') {
  if (!REPOSITORIES.has(repository)) throw new Error('Choose a listed research repository.');
  if (typeof commit !== 'string' || !/^[a-f0-9]{40}$/i.test(commit)) throw new Error('Enter the full 40-character Git commit, not a branch or short SHA.');
  if (typeof path !== 'string' || path.length > 500 || /[%\\\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/u.test(path) ||
      (path && path.split('/').some(part => !part || part === '.' || part === '..'))) {
    throw new Error('Use a relative file path with ordinary / separators; no dot segments, encoded characters or control characters.');
  }
  const sha = commit.toLowerCase();
  if (typeof sha256 !== 'string' || (sha256 && !/^[a-f0-9]{64}$/i.test(sha256))) throw new Error('Use a complete 64-character SHA-256, or leave it empty.');
  if (sha256 && !path) throw new Error('A supplied file SHA-256 needs a file path.');
  let encoded;
  try { encoded = path.split('/').map(encodeURIComponent).join('/'); }
  catch { throw new Error('The path must contain valid Unicode characters.'); }
  const project = `d6g8k5htny-coder/${repository}`;
  const url = `https://github.com/${project}/${path ? 'blob' : 'tree'}/${sha}${path ? '/' + encoded : ''}`;
  const digest = sha256.toLowerCase();
  const data = {schema: 'universal-law/source-reference/v1', repository: project, commit: sha, path: path || null, sha256: digest || null, url, verification: 'not_performed'};
  const note = `Git commit ${sha}.${path ? ' File path: ' + path + '.' : ''}${digest ? ' Supplied SHA-256 (not checked): ' + digest + '.' : ''} Source existence, authorship and review status are not verified.`;
  const bibtex = `@misc{source_reference_template,\n  title = {Repository source reference},\n  howpublished = {${bibtexLiteral(project)}},\n  note = {${bibtexLiteral(note)}},\n  url = {${url}}\n}`;
  return {url, bibtex, text: `${project}${path ? ' — ' + path : ''}. Git commit ${sha}.${digest ? ' Supplied SHA-256 (not checked): ' + digest + '.' : ''} ${url}`, json: JSON.stringify(data, null, 2)};
}
const STATE_KEYS = ['repo', 'commit', 'path', 'sha256'];
export function readReferenceState(search) {
  const params = new URLSearchParams(search);
  if (!STATE_KEYS.some(key => params.has(key))) return null;
  // URLSearchParams replaces invalid UTF-8. Exact source paths must not be repaired silently.
  try { decodeURIComponent(search); }
  catch { throw new Error('The shared link has malformed URL encoding. Enter the source identity again.'); }
  for (const key of STATE_KEYS) if (params.getAll(key).length > 1) throw new Error('The shared link has duplicate source fields. Enter the identity again.');
  const state = {repository: params.get('repo'), commit: params.get('commit'), path: params.get('path') || '', sha256: params.get('sha256') || ''};
  buildReference(state.repository, state.commit, state.path, state.sha256);
  return {...state, commit: state.commit.toLowerCase(), sha256: state.sha256.toLowerCase()};
}
export function referenceStateURL(href, state) {
  buildReference(state.repository, state.commit, state.path, state.sha256);
  const url = new URL(href);
  for (const key of STATE_KEYS) url.searchParams.delete(key);
  url.searchParams.set('repo', state.repository);
  url.searchParams.set('commit', state.commit.toLowerCase());
  if (state.path) url.searchParams.set('path', state.path);
  if (state.sha256) url.searchParams.set('sha256', state.sha256.toLowerCase());
  return url.href;
}
export function wireReferenceForm({form, repository, commit, path, sha256, output, json, bibtex, link, status, actions, copyText, copyJSON, copyBibTeX, share, clipboard, location, history, events}) {
  let current = null, revision = 0, copying = false;
  const copyButtons = [copyText, copyJSON, copyBibTeX].filter(Boolean);
  const canCopy = typeof clipboard?.writeText === 'function';
  const setCopy = enabled => copyButtons.forEach(button => { button.disabled = !enabled; });
  const clear = () => {
    revision++; current = null;
    output.textContent = ''; if (json) json.textContent = ''; if (bibtex) bibtex.textContent = '';
    link.hidden = true; link.removeAttribute('href');
    if (actions) actions.hidden = true;
    if (share) { share.hidden = true; share.removeAttribute('href'); }
    setCopy(false);
    status.textContent = 'Enter the source identity, then build a reference. Edits must be built before copying or sharing; the page address may still describe the previous reference.';
  };
  const values = () => ({repository: repository.value, commit: commit.value, path: path.value, sha256: sha256?.value || ''});
  const render = state => {
    current = buildReference(state.repository, state.commit, state.path, state.sha256);
    output.textContent = current.text; if (json) json.textContent = current.json; if (bibtex) bibtex.textContent = current.bibtex;
    link.href = current.url; link.textContent = 'Open this exact source on GitHub ↗'; link.hidden = false;
    if (actions) actions.hidden = false;
    if (share && location) { share.href = referenceStateURL(location.href, state); share.hidden = false; }
    setCopy(canCopy && !copying);
    status.textContent = 'Reference built locally. Source existence, supplied hash, authorship and review status are not checked.' + (canCopy ? '' : ' Select the text, JSON or BibTeX and copy it manually.');
    if (copying) status.textContent += ' A previous copy is still finishing; copying is temporarily disabled.';
  };
  form.addEventListener('input', clear);
  form.addEventListener('change', clear);
  form.addEventListener('submit', event => {
    event.preventDefault(); clear();
    try {
      render(values());
      if (share && history && share.href !== location.href) {
        try { history.pushState(null, '', share.href); }
        catch { status.textContent += ' The page address could not be updated; use the share link below.'; }
      }
    } catch (error) { status.textContent = error.message; }
  });
  for (const [button, key] of [[copyText, 'text'], [copyJSON, 'json'], [copyBibTeX, 'bibtex']]) button?.addEventListener('click', async () => {
    if (!current || !canCopy || copying || button.disabled) return;
    const started = revision, text = current[key];
    copying = true; setCopy(false); status.textContent = 'Copying source reference…';
    try {
      await clipboard.writeText(text);
      if (started === revision) status.textContent = `Copied ${key === 'json' ? 'JSON' : key === 'bibtex' ? 'BibTeX' : 'source reference'}. The source and supplied hash are not verified.`;
    } catch {
      if (started === revision) status.textContent = 'Clipboard unavailable. Select the text, JSON or BibTeX and copy it manually. The source is not verified.';
    } finally {
      copying = false; setCopy(Boolean(current) && canCopy);
      if (started !== revision && current) status.textContent = 'Previous copy finished. Copy the current reference when ready. The source and supplied hash are not verified.';
    }
  });
  const restore = () => {
    clear(); repository.value = 'main'; commit.value = ''; path.value = ''; if (sha256) sha256.value = '';
    try {
      const state = readReferenceState(location.search);
      if (!state) return;
      repository.value = state.repository; commit.value = state.commit; path.value = state.path; if (sha256) sha256.value = state.sha256;
      render(state);
    } catch (error) { status.textContent = `Shared reference unavailable: ${error.message}`; }
  };
  events?.addEventListener('popstate', restore);
  events?.addEventListener('hashchange', () => {
    if (current && share) share.href = referenceStateURL(location.href, values());
  });
  if (location && STATE_KEYS.some(key => new URLSearchParams(location.search).has(key))) restore();
}
