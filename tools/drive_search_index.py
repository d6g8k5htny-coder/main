"""Disposable metadata candidates, never a source, body reader or authority.

SQLite's optional FTS5 trigram index accelerates repeated ASCII substring
queries. The caller still applies the original matcher and ordering. One index
is kept per thread, only in memory: no database downloaded or found on disk is
trusted. Unsupported runtimes and queries keep the original scan behavior.
"""
import hashlib
import json
import threading

try:
    import sqlite3
except ImportError:  # Some Python builds omit the optional SQLite extension.
    sqlite3 = None

FIELDS = ('title', 'path', 'path_snapshot', 'id', 'context')
_LOCAL = threading.local()


def supported_query(query):
    # Casefold is performed by Python before SQLite sees either side. Restrict
    # the fast path so tokenizer Unicode/control behavior cannot lose matches.
    return len(query) >= 3 and query.isascii() and query.isprintable()


class MetadataIndex:
    """One bounded, replaceable corpus index; outputs always use current rows.

    Exact field tuples and row order are compared on every indexed request.
    Non-search fields need no invalidation: no result rows are stored here.
    Callers must not mutate their record sequence concurrently with a query.
    """

    def __init__(self):
        self._projection = None
        self._connection = None
        self._unsafe_rows = ()
        self._changes = None
        self._schema_version = None
        self.fingerprint = None

    def close(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._projection = None
        self.fingerprint = None

    def _build(self, projection):
        self.close()
        self._projection = projection
        self.fingerprint = hashlib.sha256(json.dumps(
            {'version': 1, 'fields': FIELDS, 'rows': projection},
            ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
        connection = sqlite3.connect(':memory:')
        try:
            connection.execute("CREATE VIRTUAL TABLE metadata USING fts5("
                               "haystack, tokenize='trigram case_sensitive 1')")
            rows, unsafe = [], []
            for number, fields in enumerate(projection):
                text = '\n'.join(fields).casefold()
                # SQLite text indexing can truncate at NUL; lone surrogates
                # cannot be UTF-8 encoded. Scan these rows through the caller.
                try:
                    text.encode('utf-8')
                    encodable = True
                except UnicodeEncodeError:
                    encodable = False
                if '\0' in text or not encodable:
                    unsafe.append(number)
                else:
                    rows.append((number + 1, text))
            connection.executemany('INSERT INTO metadata(rowid,haystack) VALUES (?,?)', rows)
            connection.commit()
            connection.execute('PRAGMA query_only = ON')
            self._changes = connection.total_changes
            self._schema_version = connection.execute('PRAGMA schema_version').fetchone()[0]
            self._unsafe_rows = tuple(unsafe)
            self._connection = connection
        except sqlite3.Error:
            connection.close()
            # Remember this projection so a runtime without FTS5 does not
            # retry building the same unavailable index for every query.
            raise

    def candidates(self, records, query):
        """Return candidate rows, or None to request the complete original scan."""
        if sqlite3 is None or not supported_query(query):
            return None
        projection = tuple(tuple(str(row.get(key, '')) for key in FIELDS) for row in records)
        try:
            if projection != self._projection:
                self._build(projection)
            elif self._connection is None:
                return None
            elif (self._connection.total_changes != self._changes or
                  self._connection.execute('PRAGMA schema_version').fetchone()[0] != self._schema_version):
                # Unexpected writes invalidate the disposable cache, including
                # deleting postings or replacing its schema. Rebuild from rows.
                self._build(projection)
            phrase = '"' + query.replace('"', '""') + '"'
            ids = self._connection.execute(
                'SELECT rowid FROM metadata WHERE metadata MATCH ?', (phrase,)).fetchall()
            selected = {number - 1 for (number,) in ids} | set(self._unsafe_rows)
            return [records[number] for number in sorted(selected)]
        except sqlite3.Error:
            if self._connection is not None:
                self.close()
            return None


def candidates(records, query):
    index = getattr(_LOCAL, 'index', None)
    if index is None:
        index = _LOCAL.index = MetadataIndex()
    return index.candidates(records, query)
