/** Build syntax-checked source references locally. No source existence or review claim. */
const REPOSITORIES = new Set(['main', 'Math-', 'query-', 'Universal-Law-Workspace']);
export function buildReference(repository, commit, path = '') {
  if (!REPOSITORIES.has(repository)) throw new Error('Choose a listed research repository.');
  if (typeof commit !== 'string' || !/^[a-f0-9]{40}$/i.test(commit)) throw new Error('Enter the full 40-character Git commit, not a branch or short SHA.');
  if (typeof path !== 'string' || path.length > 500 || /[%\\\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/u.test(path) ||
      (path && path.split('/').some(part => !part || part === '.' || part === '..'))) {
    throw new Error('Use a relative file path with ordinary / separators; no dot segments, encoded characters or control characters.');
  }
  const sha = commit.toLowerCase();
  let encoded;
  try { encoded = path.split('/').map(encodeURIComponent).join('/'); }
  catch { throw new Error('The path must contain valid Unicode characters.'); }
  const project = `d6g8k5htny-coder/${repository}`;
  const url = `https://github.com/${project}/${path ? 'blob' : 'tree'}/${sha}${path ? '/' + encoded : ''}`;
  return {url, text: `${project}${path ? ' — ' + path : ''}. Git commit ${sha}. ${url}`};
}
export function wireReferenceForm({form, repository, commit, path, output, link, status}) {
  const clear = () => { output.textContent = ''; link.hidden = true; link.removeAttribute('href'); status.textContent = 'Enter the source identity, then build a reference.'; };
  form.addEventListener('input', clear);
  form.addEventListener('change', clear);
  form.addEventListener('submit', event => {
    event.preventDefault(); clear();
    try {
      const reference = buildReference(repository.value, commit.value, path.value);
      output.textContent = reference.text;
      link.href = reference.url; link.textContent = 'Open this exact source on GitHub ↗'; link.hidden = false;
      status.textContent = 'Reference built locally. Source existence, authorship and review status are not checked.';
    } catch (error) { status.textContent = error.message; }
  });
}
