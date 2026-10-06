#!/usr/bin/env python3
"""Refresh Git-tracked local asset URLs; CI uses --check.

Stage new/deleted site inputs before running this tool. Release identities use
tracked membership and current working-tree bytes, with explicit POSIX ordering.
Only recognized live URL spans are normalized/rewritten. This bounded stdlib
scanner refuses unsupported escaped URLs and ambiguous JavaScript contexts
before any writes; it is not a complete JavaScript parser.
"""
import argparse
import hashlib
import html
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
EDITABLE = {'.html', '.js', '.mjs', '.css'}
ASSETS = {'.js', '.mjs', '.css', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.ico', '.woff', '.woff2'}

def inventory(site):
    """Tracked membership, worktree bytes; no filesystem fallback."""
    site = site.resolve()
    result = subprocess.run(['git', '-C', str(site), 'rev-parse', '--show-toplevel'],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError('site must belong to a Git worktree; stage new inputs')
    repo = Path(result.stdout.decode('utf-8').strip()).resolve()
    roots = [site]
    catalog = site.parent / 'public-math'
    if catalog.is_dir():
        roots.append(catalog)
    files = []
    for root in roots:
        if not root.is_relative_to(repo):
            raise ValueError('site/catalog must remain inside the same Git worktree')
        relative = root.relative_to(repo).as_posix()
        tracked = subprocess.run(
            ['git', '-C', str(repo), 'ls-files', '--cached', '-z', '--',
             ':(literal)' + relative],
            capture_output=True, check=False)
        if tracked.returncode:
            raise ValueError('cannot read Git-tracked site inventory')
        group = []
        for entry in tracked.stdout.split(b'\0'):
            if not entry:
                continue
            path = repo / entry.decode('utf-8')
            if not path.is_relative_to(root):
                continue
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f'missing/outside local asset in tracked inventory: {path}')
            group.append(path)
        files.extend(sorted(group, key=lambda p: p.relative_to(root).as_posix()))
    if not any(p.is_relative_to(site) for p in files):
        raise ValueError('site has no tracked inputs; stage them first')
    return files

def without_release(value):
    """Remove only reserved query entries without decoding other parameters."""
    url = urlsplit(value)
    query = '&'.join(part for part in url.query.split('&')
                     if part.split('=', 1)[0] != 'site-release')
    return urlunsplit((url.scheme, url.netloc, url.path, query, url.fragment))

def transform(text, suffix, convert):
    amended = text
    for begin, end in reversed(source_spans(text, suffix)):
        raw = text[begin:end]
        value = html.unescape(raw) if suffix == '.html' else raw
        converted = convert(value)
        if converted == value:
            continue
        if suffix == '.html':
            converted = html.escape(converted, quote=True)
        amended = amended[:begin] + converted + amended[end:]
    return amended

def release(site, files=None):
    files = inventory(site) if files is None else files
    digest = hashlib.sha256(b'public-site-asset-release-v1\0')
    for path in files:
        data = path.read_bytes()
        if path.suffix in EDITABLE:
            data = transform(data.decode('utf-8'), path.suffix,
                             without_release).encode('utf-8')
        name = path.relative_to(site.resolve().parent).as_posix().encode('utf-8')
        for part in (name, data):
            digest.update(len(part).to_bytes(8, 'big'))
            digest.update(part)
    return digest.hexdigest()

def version(value, path, site, token, files=None):
    clean = without_release(value)
    url = urlsplit(clean)
    if url.scheme or url.netloc or not url.path or Path(url.path).suffix not in ASSETS:
        return value
    target = (path.parent / url.path).resolve()
    allowed = set(inventory(site) if files is None else files)
    if not target.is_relative_to(site.resolve()) or not target.is_file() or target not in allowed:
        raise ValueError(f'{path.name}: missing/outside local asset {value} (must be tracked)')
    query = url.query + ('&' if url.query else '') + 'site-release=' + token
    return urlunsplit((url.scheme, url.netloc, url.path, query, url.fragment))

def updates(site, token, files=None):
    files = inventory(site) if files is None else files
    for path in files:
        if not path.is_relative_to(site.resolve()) or path.suffix not in EDITABLE:
            continue
        text = path.read_bytes().decode('utf-8')
        amended = transform(text, path.suffix,
                            lambda value: version(value, path, site, token, files))
        if amended != text:
            yield path, amended

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT/'docs/site')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.site = args.site.resolve()
    if not args.site.is_dir():
        parser.error('site must be an existing directory')
    try:
        files = inventory(args.site)
        token = release(args.site, files)
        pending = list(updates(args.site, token, files))
    except (ValueError, UnicodeError, OSError) as error:
        parser.error(str(error))
    if args.check and pending:
        print('Asset URLs are stale; run python -B tools/site_asset_release.py')
        print('Expected release: ' + token)
        for path, _ in pending:
            print(path.relative_to(args.site).as_posix())
        return 1
    for path, text in pending:
        path.write_bytes(text.encode('utf-8'))
    print(token)
    return 0

from html.parser import HTMLParser

_NAME = re.compile(r'(?:[^\W\d]|[$_])[\w$]*')
_CSS_NAME = re.compile(r'(?:[^\W\d]|[-_])[-\w]*')
_PUNCT = re.compile(
    r'===|!==|\*\*=|=>|\?\.|\?\?|&&|\|\||'
    r'==|!=|<=|>=|\+\+|--|\*\*|'
    r'[-+*/%&|^]=|.'
)
_HTML_ATTR = re.compile(
    r'''([^\s/>=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?'''
)
_REGEX_PREFIX = {
    '(', '[', '{', ',', ':', ';', '=', '!', '?',
    '+', '-', '*', '/', '%', '&', '|', '^', '~',
    '=>', '&&', '||', '??', '==', '!=', '===', '!==',
    '<', '>', '<=', '>=', '+=', '-=', '*=', '/=', '%=',
    '&=', '|=', '^=', '**', '**=',
    'return', 'throw', 'case', 'delete', 'void', 'typeof',
    'new', 'in', 'instanceof', 'yield', 'await',
    'else', 'do', 'default', 'break', 'continue',
}
_CONTROL = {'if', 'while', 'for', 'with', 'switch', 'catch'}


def _quoted_end(text, start):
    quote = text[start]
    pos = start + 1
    while pos < len(text):
        if text[pos] == '\\':
            pos += 2
        elif text[pos] == quote:
            return pos + 1
        else:
            pos += 1
    raise ValueError('Unterminated string literal')


def _trivia(text, pos, js=False):
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
        elif text.startswith('/*', pos):
            end = text.find('*/', pos + 2)
            if end < 0:
                raise ValueError('Unterminated block comment')
            pos = end + 2
        elif js and text.startswith('//', pos):
            end = text.find('\n', pos + 2)
            pos = len(text) if end < 0 else end + 1
        else:
            break
    return pos


def _regex_end(text, start):
    """Return a possible regex end; do not decide its lexical context."""
    pos = start + 1
    bracket = False
    while pos < len(text):
        char = text[pos]
        if char in '\r\n':
            return None
        if char == '\\':
            if pos + 1 >= len(text) or text[pos + 1] in '\r\n':
                return None
            pos += 2
            continue
        if char == '[':
            bracket = True
        elif char == ']':
            bracket = False
        elif char == '/' and not bracket:
            pos += 1
            while pos < len(text) and text[pos].isalpha():
                pos += 1
            return pos
        pos += 1
    return None


def _js_tokens(text, start=0, template_expression=False):
    # Tokens are (kind, spelling, start, end).
    tokens = []
    pos = start
    braces = 0
    parens = []
    previous = None
    while pos < len(text):
        pos = _trivia(text, pos, js=True)
        if pos == len(text):
            break
        char = text[pos]
        if template_expression and char == '}' and braces == 0:
            return tokens, pos + 1
        begin = pos
        if char in '"\'':
            pos = _quoted_end(text, pos)
            token = ('string', text[begin:pos], begin, pos)
        elif char == '`':
            pos += 1
            while pos < len(text):
                if text[pos] == '\\':
                    pos += 2
                elif text[pos] == '`':
                    pos += 1
                    break
                elif text.startswith('${', pos):
                    inner, pos = _js_tokens(text, pos + 2, True)
                    if any(t[0] == 'word' and t[1] == 'import'
                           for t in inner):
                        raise ValueError(
                            'Import in template expression requires '
                            'a full JavaScript parser'
                        )
                else:
                    pos += 1
            else:
                raise ValueError('Unterminated template literal')
            token = ('template', text[begin:pos], begin, pos)
        elif char == '/':
            end = _regex_end(text, pos)
            property_word = (previous is not None and previous[0] == 'word'
                             and len(tokens) >= 2
                             and tokens[-2][1] in ('.', '?.'))
            prefix = (previous is None or previous[0] == 'control-close'
                      or (previous[1] in _REGEX_PREFIX and not property_word))
            ambiguous = previous is not None and previous[1] in (')', '}')
            if prefix:
                if end is None:
                    raise ValueError('Unsupported or unterminated regex')
                pos = end
                token = ('regex', text[begin:pos], begin, pos)
            elif ambiguous and end is not None:
                candidate = text[begin:end]
                if re.search(r'\b(?:import|export)\b', candidate):
                    raise ValueError(
                        'Ambiguous regex/division containing module syntax'
                    )
                # No module syntax can be lost within this candidate.
                pos = end
                token = ('regex', candidate, begin, pos)
            else:
                match = _PUNCT.match(text, pos)
                pos = match.end()
                token = ('punct', match[0], begin, pos)
        else:
            match = _NAME.match(text, pos)
            if match:
                pos = match.end()
                token = ('word', match[0], begin, pos)
            else:
                if char == '\\' or ord(char) > 127:
                    raise ValueError('Unsupported JavaScript identifier syntax')
                match = _PUNCT.match(text, pos)
                pos = match.end()
                token = ('punct', match[0], begin, pos)

        if token[1] == '(':
            parens.append(previous is not None
                          and previous[0] == 'word'
                          and previous[1] in _CONTROL
                          and (len(tokens) < 2 or tokens[-2][1] not in ('.', '?.')))
        elif token[1] == ')':
            control = parens.pop() if parens else False
            if control:
                token = ('control-close', ')', begin, pos)
        elif token[1] == '{':
            braces += 1
        elif token[1] == '}':
            braces -= 1
        tokens.append(token)
        previous = token
    if template_expression:
        raise ValueError('Unterminated template expression')
    return tokens, pos


def _module_spans(text):
    tokens, _ = _js_tokens(text)
    spans = []

    def spelling(index):
        return tokens[index][1] if index < len(tokens) else None

    def literal(index):
        if index < len(tokens) and tokens[index][0] == 'string':
            token = tokens[index]
            if '\\' in token[1]:
                raise ValueError('Escaped module specifier is unsupported')
            spans.append((token[2] + 1, token[3] - 1))

    def clause(index):
        # Optional default binding followed by named/namespace bindings.
        if index < len(tokens) and tokens[index][0] == 'word':
            index += 1
            if spelling(index) != ',':
                return index
            index += 1
        if spelling(index) == '*':
            if (spelling(index + 1) == 'as'
                    and index + 2 < len(tokens)
                    and tokens[index + 2][0] == 'word'):
                return index + 3
            return None
        if spelling(index) == '{':
            depth = 1
            index += 1
            while index < len(tokens):
                if spelling(index) == '{':
                    depth += 1
                elif spelling(index) == '}':
                    depth -= 1
                    if depth == 0:
                        return index + 1
                index += 1
        return None

    for index, token in enumerate(tokens):
        if token[0] != 'word' or token[1] not in ('import', 'export'):
            continue
        if index and tokens[index - 1][1] in ('.', '?.'):
            continue
        following = index + 1
        if token[1] == 'import':
            if spelling(following) == '(':
                # Computed/concatenated arguments are deliberately untouched.
                if spelling(following + 2) in (')', ','):
                    literal(following + 1)
                continue
            if following < len(tokens) and tokens[following][0] == 'string':
                literal(following)
                continue
        elif spelling(following) not in ('*', '{'):
            continue
        if token[1] == 'export' and spelling(following) == '*':
            following += 1
            if spelling(following) == 'as':
                following += 2
        else:
            following = clause(following)
        if following is not None and spelling(following) == 'from':
            literal(following + 1)
    return spans


def _css_spans(text):
    spans = []
    pos = 0
    while pos < len(text):
        pos = _trivia(text, pos)
        if pos == len(text):
            break
        if text[pos] in '"\'':
            pos = _quoted_end(text, pos)
            continue
        if text[pos] == '\\':
            raise ValueError('Escaped CSS identifier syntax is unsupported')
        if text[pos] == '@':
            match = _CSS_NAME.match(text, pos + 1)
            if match:
                pos = match.end()
                if match[0].lower() == 'import':
                    pos = _trivia(text, pos)
                    if pos < len(text) and text[pos] in '"\'':
                        end = _quoted_end(text, pos)
                        if '\\' in text[pos + 1:end - 1]:
                            raise ValueError('Escaped CSS URL is unsupported')
                        spans.append((pos + 1, end - 1))
                        pos = end
                continue
        match = _CSS_NAME.match(text, pos)
        if not match:
            pos += 1
            continue
        pos = match.end()
        if match[0].lower() != 'url' or pos >= len(text) or text[pos] != '(':
            continue
        pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] in '"\'':
            begin = pos + 1
            end = _quoted_end(text, pos)
            finish = end - 1
            pos = end
        else:
            begin = pos
            while pos < len(text) and text[pos] not in ') \t\r\n\f':
                pos += 1
            finish = pos
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != ')':
            raise ValueError('Unsupported CSS url() syntax')
        if '\\' in text[begin:finish]:
            raise ValueError('Escaped CSS URL is unsupported')
        if begin != finish:
            spans.append((begin, finish))
        pos += 1
    return spans


def _html_spans(text):
    class AssetTags(HTMLParser):
        # Also protect HTML raw-text and escapable-raw-text examples.
        CDATA_CONTENT_ELEMENTS = (
            *HTMLParser.CDATA_CONTENT_ELEMENTS,
            'textarea', 'title', 'xmp', 'iframe', 'noembed', 'noframes',
        )

        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.spans = []
            self.lines = [0]
            self.lines.extend(m.end() for m in re.finditer('\n', text))
            self.plaintext = False

        def handle_starttag(self, tag, attrs):
            if self.plaintext:
                return
            if tag == 'plaintext':
                self.plaintext = True
                return
            wanted = {'script': 'src', 'link': 'href'}.get(tag)
            if wanted is None:
                return
            raw = self.get_starttag_text()
            line, column = self.getpos()
            offset = self.lines[line - 1] + column
            head = re.match(r'<[^\s/>]+', raw)
            seen = set()
            for match in _HTML_ATTR.finditer(raw, head.end()):
                name = match[1].lower()
                if name in seen:
                    continue
                seen.add(name)
                if name != wanted:
                    continue
                for group in (2, 3, 4):
                    if match[group] is not None:
                        self.spans.append((
                            offset + match.start(group),
                            offset + match.end(group),
                        ))
                        break

        handle_startendtag = handle_starttag

    parser = AssetTags()
    parser.feed(text)
    parser.close()
    return parser.spans


def source_spans(text, suffix):
    """Discover supported live URL content spans; preserve all other bytes."""
    if suffix == '.html':
        spans = _html_spans(text)
    elif suffix == '.css':
        spans = _css_spans(text)
    else:
        spans = _module_spans(text)
    return sorted(set(spans))

if __name__ == '__main__': raise SystemExit(main())
