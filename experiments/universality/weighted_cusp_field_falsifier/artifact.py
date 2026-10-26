"""Canonical, lossless exact artifacts; no float or implicit type coercion.

The wire grammar uses strings/bools/null directly and tagged JSON arrays:
["i",decimal], ["q",numerator,denominator], ["l",items], ["t",items],
["d",[[key,value],...]]. Decimal components are strings. Dictionary entries
sort by the complete ASCII encoding of their strict int/string key. No JSON
object is legal, so duplicate JSON object keys are rejected with all objects.
Depth counts typed parent/child edges from root depth zero; keys are children.
Nodes count typed values and dictionary keys, not wire-only tags/pair arrays.
The 64 MiB wire ceiling is an outer safety bound; callers enforce smaller
record budgets (including the production 32 MiB cell-record reservation).
"""
from fractions import Fraction
from io import BytesIO
import json

MAX_DEPTH = 64
MAX_NODES = 1000000
MAX_INTEGER_DIGITS = 10000
MAX_BYTES = 67108864


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _scalar_text(value):
    _need(all(not 0xD800 <= ord(character) <= 0xDFFF for character in value),
          'Unicode scalar text required; surrogate codepoints are forbidden')
    return value


def _integer_text(value):
    """Avoid interpreter-global decimal conversion limits or configuration."""
    _need(type(value) is int, 'Strict integer required')
    _need(value.bit_length() <= 4 * MAX_INTEGER_DIGITS, 'Integer digit limit')
    negative = value < 0
    remaining = -value if negative else value
    parts = []
    while remaining:
        remaining, part = divmod(remaining, 1000000000)
        parts.append(part)
    text = str(parts.pop()) if parts else '0'
    while parts:
        text += '%09d' % parts.pop()
    _need(len(text) <= MAX_INTEGER_DIGITS, 'Integer digit limit')
    return '-' + text if negative else text


def _integer_value(text):
    _need(type(text) is str and text, 'Decimal string required')
    negative = text.startswith('-')
    digits = text[1:] if negative else text
    _need(0 < len(digits) <= MAX_INTEGER_DIGITS, 'Integer digit limit')
    _need(all('0' <= ch <= '9' for ch in digits), 'ASCII decimal digits required')
    _need((len(digits) == 1 or digits[0] != '0') and not (negative and digits == '0'),
          'Canonical decimal sign and leading digits required')
    value = 0
    for offset in range(0, len(digits), 9):
        chunk = digits[offset:offset + 9]
        value = value * (10 ** len(chunk)) + int(chunk)
    return -value if negative else value


def _quoted(value):
    _need(len(value) <= MAX_BYTES, 'Artifact byte limit')
    _scalar_text(value)
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def _key_bytes(key):
    if type(key) is str:
        return _quoted(key)
    _need(type(key) is int, 'Dictionary keys must be strict strings or integers')
    return b'["i","' + _integer_text(key).encode('ascii') + b'"]'


def dumps(value):
    """Return canonical ASCII bytes with exactly one final LF, or reject."""
    output = BytesIO()
    nodes = 0

    def put(raw):
        _need(output.tell() + len(raw) <= MAX_BYTES, 'Artifact byte limit')
        output.write(raw)

    def visit(depth):
        nonlocal nodes
        _need(depth <= MAX_DEPTH, 'Artifact depth limit')
        nodes += 1
        _need(nodes <= MAX_NODES, 'Artifact node limit')

    def encode(item, depth):
        visit(depth)
        kind = type(item)
        if item is None:
            put(b'null')
        elif kind is bool:
            put(b'true' if item else b'false')
        elif kind is str:
            put(_quoted(item))
        elif kind is int:
            put(b'["i","' + _integer_text(item).encode('ascii') + b'"]')
        elif kind is Fraction:
            put(b'["q","' + _integer_text(item.numerator).encode('ascii') + b'","' +
                _integer_text(item.denominator).encode('ascii') + b'"]')
        elif kind in (list, tuple):
            put(b'["l",[' if kind is list else b'["t",[')
            for index, child in enumerate(item):
                if index:
                    put(b',')
                encode(child, depth + 1)
            put(b']]')
        elif kind is dict:
            # Sorting encoded keys keeps int 8 distinct from string "8".
            ordered = sorted((_key_bytes(key), key) for key in item)
            put(b'["d",[')
            for index, (encoded_key, key) in enumerate(ordered):
                if index:
                    put(b',')
                put(b'[')
                visit(depth + 1)
                put(encoded_key)
                put(b',')
                encode(item[key], depth + 1)
                put(b']')
            put(b']]')
        else:
            raise ValueError('Unsupported exact artifact type')

    encode(value, 0)
    put(b'\n')
    return output.getvalue()


class _Decoder:
    """Read only the typed grammar, enforcing quotas before building nodes."""
    def __init__(self, text):
        self.text = text
        self.position = 0
        self.nodes = 0

    def take(self, token):
        _need(self.text.startswith(token, self.position), 'Malformed typed artifact')
        self.position += len(token)

    def string(self):
        self.take('"')
        try:
            value, end = json.decoder.scanstring(self.text, self.position, True)
        except (ValueError, OverflowError) as error:
            raise ValueError('Malformed artifact string') from error
        self.position = end
        return _scalar_text(value)

    def items(self, depth):
        self.take('[')
        values = []
        if self.text.startswith(']', self.position):
            self.take(']')
            return values
        while True:
            values.append(self.value(depth))
            if self.text.startswith(']', self.position):
                self.take(']')
                return values
            self.take(',')

    def mapping(self, depth):
        self.take('[')
        values = {}
        if self.text.startswith(']', self.position):
            self.take(']')
            return values
        while True:
            self.take('[')
            key = self.value(depth)
            _need(type(key) in (str, int), 'Dictionary keys must be strict strings or integers')
            _need(key not in values, 'Duplicate decoded dictionary key')
            self.take(',')
            values[key] = self.value(depth)
            self.take(']')
            if self.text.startswith(']', self.position):
                self.take(']')
                return values
            self.take(',')

    def value(self, depth):
        _need(depth <= MAX_DEPTH, 'Artifact depth limit')
        self.nodes += 1
        _need(self.nodes <= MAX_NODES, 'Artifact node limit')
        _need(self.position < len(self.text), 'Missing artifact value')
        initial = self.text[self.position]
        if initial == '"':
            return self.string()
        for token, result in (('null', None), ('true', True), ('false', False)):
            if self.text.startswith(token, self.position):
                self.take(token)
                return result
        # Bare numbers, objects, floats, and unknown tags are never admitted.
        self.take('[')
        tag = self.string()
        self.take(',')
        if tag == 'i':
            result = _integer_value(self.string())
        elif tag == 'q':
            numerator = _integer_value(self.string())
            self.take(',')
            denominator = _integer_value(self.string())
            _need(denominator > 0, 'Positive fraction denominator required')
            result = Fraction(numerator, denominator)
            _need(result.numerator == numerator and result.denominator == denominator,
                  'Reduced fraction and canonical sign required')
        elif tag in ('l', 't'):
            result = self.items(depth + 1)
            if tag == 't':
                result = tuple(result)
        elif tag == 'd':
            result = self.mapping(depth + 1)
        else:
            raise ValueError('Unknown artifact type tag')
        self.take(']')
        return result


def loads(raw, max_bytes):
    """Decode bounded canonical bytes while preserving all exact types."""
    _need(type(raw) is bytes, 'Artifact bytes required')
    _need(type(max_bytes) is int and max_bytes >= 0, 'Strict nonnegative byte quota required')
    _need(len(raw) <= min(max_bytes, MAX_BYTES), 'Artifact byte quota exceeded')
    _need(raw.endswith(b'\n'), 'Canonical final LF required')
    try:
        text = raw[:-1].decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('ASCII artifact required') from error
    decoder = _Decoder(text)
    value = decoder.value(0)
    _need(decoder.position == len(text), 'Trailing artifact bytes')
    _need(dumps(value) == raw, 'Noncanonical artifact bytes')
    return value
