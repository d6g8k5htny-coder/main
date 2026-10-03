#!/usr/bin/env python3
"""Refresh local asset URLs after presentation/data changes; CI uses --check.

Run `python -B tools/site_asset_release.py` before committing a site release.
The token selects a new cache key, not immutable historical hosting. Frozen data
and proof bytes are hashed as inputs but never rewritten. Existing HTML routes
and external source identities are preserved.
"""
import argparse
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r'[?&]site-release=[0-9a-f]{64}')
HTML = re.compile(r'(<(?:script|link)\b[^>]*?\b(?:src|href)\s*=\s*)([\"\'])([^\"\']+)(\2)', re.I)
IMPORT = re.compile(r'(\bfrom\s*|\bimport\s*\(\s*|\bimport\s*)([\"\'])([^\"\']+)(\2)')
CSS = re.compile(r'(url\(\s*)([\"\']?)([^\"\'\s)]+)(\2\s*\))')
EDITABLE = {'.html', '.js', '.mjs', '.css'}
ASSETS = {'.js', '.mjs', '.css', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.ico', '.woff', '.woff2'}

def release(site):
    """Hash the complete local site and sibling catalog, with tokens removed."""
    digest = hashlib.sha256(b'public-site-asset-release-v1\0')
    roots = [site]
    catalog = site.parent/'public-math'
    if catalog.is_dir(): roots.append(catalog)
    for root in roots:
        for path in sorted(p for p in root.rglob('*') if p.is_file()):
            data = path.read_bytes()
            if path.suffix in EDITABLE:
                data = TOKEN.sub('', data.decode('utf-8')).encode('utf-8')
            name = path.relative_to(site.parent).as_posix().encode('utf-8')
            for part in (name, data):
                digest.update(len(part).to_bytes(8, 'big')); digest.update(part)
    return digest.hexdigest()

def version(value, path, site, token):
    clean = TOKEN.sub('', value)
    url = urlsplit(clean)
    if url.scheme or url.netloc or not url.path or Path(url.path).suffix not in ASSETS:
        return value
    target = (path.parent/url.path).resolve()
    if not target.is_relative_to(site.resolve()) or not target.is_file():
        raise ValueError(f'{path.name}: missing/outside local asset {value}')
    query = url.query + ('&' if url.query else '') + 'site-release=' + token
    return urlunsplit((url.scheme, url.netloc, url.path, query, url.fragment))

def updates(site, token):
    for path in sorted(p for p in site.rglob('*') if p.is_file() and p.suffix in EDITABLE):
        text = path.read_text(encoding='utf-8')
        pattern = HTML if path.suffix == '.html' else CSS if path.suffix == '.css' else IMPORT
        amended = pattern.sub(lambda m: m[1]+m[2]+version(m[3], path, site, token)+m[4], text)
        if amended != text: yield path, amended

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT/'docs/site')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.site.is_dir(): parser.error('site must be an existing directory')
    try:
        token = release(args.site)
        pending = list(updates(args.site, token))
    except (ValueError, UnicodeError) as error:
        parser.error(str(error))
    if args.check and pending:
        print('Asset URLs are stale; run python -B tools/site_asset_release.py')
        for path, _ in pending: print(path.relative_to(args.site))
        return 1
    for path, text in pending: path.write_text(text, encoding='utf-8', newline='')
    print(token)
    return 0

if __name__ == '__main__': raise SystemExit(main())
