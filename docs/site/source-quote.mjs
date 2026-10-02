/** Small presentation-only projection of already verified source quotes.
 * This is not a general Markdown renderer. Unrecognized content stays literal;
 * native text nodes, never HTML parsing, carry every source-supplied value.
 */
const repositories = new Set(['main', 'Math-', 'query-', 'Universal-Law-Workspace']);
const controls = /[\\\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/u;
const safeParts = path => path.split('/').every(part => part && part !== '.' && part !== '..');
export function quotedSourceLink(target, source) {
  if (typeof target !== 'string' || controls.test(target) || target.includes('%')) return null;
  if (target.startsWith('https://')) {
    // Preserve quoted branch links as branch links; do not imply byte verification.
    if (/\s/u.test(target)) return null;
    const match=target.match(/^https:\/\/github\.com\/d6g8k5htny-coder\/([^/]+)\/(.+?)(?:#([\w.:-]+))?$/u);
    if (!match || !repositories.has(match[1]) || !safeParts(match[2])) return null;
    if (!/^(?:issues|pull)\/[1-9][0-9]*$/.test(match[2]) && !/^blob\/(?:main|[a-f0-9]{40})\/.+/.test(match[2])) return null;
    if (target.includes('?')) return null;
    return target;
  }
  if (!source || !/^d6g8k5htny-coder\/(main|Math-|query-|Universal-Law-Workspace)$/.test(source.repository) || !/^[a-f0-9]{40}$/.test(source.commit)) return null;
  if (target.startsWith('/') || target.includes(':') || target.includes('?')) return null;
  const [path,fragment,...extra]=target.split('#');
  if (extra.length || !safeParts(path) || (fragment !== undefined && !/^[\w.:-]+$/.test(fragment))) return null;
  if (typeof source.path !== 'string' || controls.test(source.path) || !safeParts(source.path)) return null;
  const parent=source.path.split('/').slice(0,-1);
  try {
    return `https://github.com/${source.repository}/blob/${source.commit}/${[...parent,...path.split('/')].map(encodeURIComponent).join('/')}${fragment===undefined?'':'#'+fragment}`;
  } catch { return null; }
}
export function renderSourceQuote(document, quote, source) {
  const projection=document.createElement('blockquote');
  projection.className='source-quote source-quote-readable';
  const syntax=/`([^`\n]+)`|\*\*([^*\n]+)\*\*|\[([^\]\n]+)\]\(([^)\n]+)\)/g;
  let offset=0;
  for (const token of quote.matchAll(syntax)) {
    projection.append(document.createTextNode(quote.slice(offset,token.index)));
    let node;
    if (quote[token.index-1]==='\\' || (token[3]!==undefined && quote[token.index-1]==='!')) {
      node=document.createTextNode(token[0]);
    } else if (token[1]!==undefined || token[2]!==undefined) {
      node=document.createElement(token[1]!==undefined?'code':'strong');
      node.textContent=token[1]??token[2];
    } else {
      const href=quotedSourceLink(token[4],source);
      if (href) { node=document.createElement('a');node.href=href;node.textContent=token[3]; }
      else node=document.createTextNode(token[0]);
    }
    projection.append(node);offset=token.index+token[0].length;
  }
  projection.append(document.createTextNode(quote.slice(offset)));
  const disclosure=document.createElement('details'),summary=document.createElement('summary');
  summary.textContent='Exact source quote';
  const original=document.createElement('blockquote');original.className='source-quote source-quote-original';original.textContent=quote;
  disclosure.append(summary,original);
  return [projection,disclosure];
}
