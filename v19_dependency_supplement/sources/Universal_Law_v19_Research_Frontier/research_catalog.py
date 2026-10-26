"""Pinned read-only research documents; captured source code is never executed.

read_document(id) returns (descriptor: dict, content: bytes). Descriptors are safe
for JSON. Callers must render text as text, never interpret captured HTML/scripts.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

ROOT = Path(__file__).resolve().parent
ALLOWED_ROOTS = frozenset({'research', 'history', 'literature', 'mathematics'})
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024


class ResearchCatalogError(ValueError):
    """Catalog metadata is invalid or a requested document is unavailable."""


class UnknownDocument(KeyError):
    """The caller requested an identifier absent from the catalog."""


class DocumentIntegrityError(ResearchCatalogError):
    """Pinned bytes are missing, unsafe, or changed."""


def _relative(value):
    if not isinstance(value, str) or '\\' in value or '\0' in value:
        raise ResearchCatalogError('invalid document path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] not in ALLOWED_ROOTS:
        raise ResearchCatalogError('document path outside approved roots')
    if str(path) != value:
        raise ResearchCatalogError('document path is not canonical')
    return path.parts


def _read_at(root, parts, maximum):
    """Pin every directory descriptor; no symlink or path-check/open race."""
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise DocumentIntegrityError('document is not a bounded regular file')
            with os.fdopen(os.dup(descriptor), 'rb') as stream:
                result = stream.read(maximum + 1)
            if len(result) > maximum:
                raise DocumentIntegrityError('document exceeds read budget')
            return result
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


class ResearchCatalog:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        try:
            raw = _read_at(self.root, ('research_catalog.json',), 8 * 1024 * 1024)
            data = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ResearchCatalogError('cannot load research catalog') from error
        if not isinstance(data, dict) or data.get('schema') != 'UL-final-research-1':
            raise ResearchCatalogError('unsupported research catalog schema')
        self._data = data
        self._documents = {}
        for item in data.get('documents', []):
            identity = item.get('id')
            if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,199}', identity):
                raise ResearchCatalogError('invalid document ID')
            if identity in self._documents:
                raise ResearchCatalogError('duplicate document ID')
            if item.get('path') is not None:
                _relative(item['path'])
                if not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', ''))):
                    raise ResearchCatalogError('invalid document digest')
                if type(item.get('bytes')) is not int or not 0 <= item['bytes'] <= MAX_DOCUMENT_BYTES:
                    raise ResearchCatalogError('invalid document byte budget')
            self._documents[identity] = item

    def snapshot(self):
        """Metadata only: document bytes are validated each time they are read."""
        return deepcopy({key: self._data[key] for key in ('documents', 'alignments', 'status', 'limits')})

    def read_document(self, document_id):
        try:
            item = self._documents[document_id]
        except (KeyError, TypeError):
            raise UnknownDocument(document_id) from None
        if item.get('path') is None:
            raise ResearchCatalogError('external reference: full text is not bundled')
        try:
            raw = _read_at(self.root, _relative(item['path']), item['bytes'])
        except (OSError, ValueError) as error:
            raise DocumentIntegrityError('document is missing, unsafe, or exceeds its pinned size') from error
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise DocumentIntegrityError('document hash or size mismatch')
        descriptor = deepcopy(item)
        descriptor['integrity'] = 'sha256_and_size_verified'
        return descriptor, raw
