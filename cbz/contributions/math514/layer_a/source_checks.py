"""Bounded lexical/AST checks. See SPEC.md for claims, trust profile and nonclaims."""
import ast
from dataclasses import dataclass
import re

from canonical_lean import (lex_tokens as canonical_identifier_tokens,
                            _CHAR_LIT as canonical_character_literal,
                            _NUMERAL as canonical_numeral_spelling)


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    column: int
    quoted: bool = False


def finding(code, detail, line=0):
    return {'code': code, 'detail': detail, 'line': line}


def lean_tokens(text):
    """Tokenize source without interpreting syntax, retaining accurate source locations."""
    tokens, gaps = [], []
    i, line, column, n = 0, 1, 0, len(text)

    def advance(end):
        nonlocal i, line, column
        segment = text[i:end]
        count = segment.count('\n')
        column = len(segment.rsplit('\n', 1)[-1]) if count else column + len(segment)
        line += count
        i = end

    def error(detail):
        gaps.append(finding('LEAN_LEXICAL_GAP', detail, line))

    def literal():
        tokens.append(Token('<literal>', line, column))

    while i < n:
        c = text[i]
        if c.isspace():
            advance(i + 1)
        elif text.startswith('--', i):
            end = text.find('\n', i + 2)
            advance(n if end < 0 else end)
        elif text.startswith('/-', i):
            j, depth = i + 2, 1
            while j < n and depth:
                if text.startswith('/-', j):
                    depth += 1; j += 2
                elif text.startswith('-/', j):
                    depth -= 1; j += 2
                else:
                    j += 1
            if depth:
                error('unterminated nested block comment')
            advance(j)
        elif (raw := re.match(r'r(#+)?"', text[i:])) is not None:
            literal()
            marker = '"' + (raw.group(1) or '')
            end = text.find(marker, i + raw.end())
            if end < 0:
                error('unterminated raw string literal'); advance(n)
            else:
                advance(end + len(marker))
        elif c == '"':
            literal()
            if i > 0 and text[i - 1] == '!':
                error('interpolated string can contain executable terms; requires elaborated certification')
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                elif text[j] == '"':
                    break
                else:
                    j += 1
            if j >= n:
                error('unterminated string literal'); advance(n)
            else:
                advance(j + 1)
        elif c == '«':
            end = text.find('»', i + 1)
            if end < 0:
                error('unterminated escaped identifier'); advance(n)
            else:
                tokens.append(Token(text[i + 1:end], line, column, True)); advance(end + 1)
        elif c == "'":
            match = canonical_character_literal.match(text, i)
            if match is not None:
                literal()
                advance(match.end())
            else:
                error('unsupported or unterminated character literal'); advance(i + 1)
        elif c.isdigit():
            # Preserve the canonical numeral boundary: valid Lean permits a
            # numeral immediately before a declaration keyword (0theorem).
            # Keep spelling/location for literal-hypothesis checks; no numeric
            # evaluation or independent numeral grammar is introduced here.
            match = canonical_numeral_spelling.match(text, i)
            end = match.end() if match is not None else i + 1
            tokens.append(Token(text[i:end], line, column)); advance(end)
        elif c.isalpha() or c == '_':
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "_'!?" or (text[j] == '.' and j + 1 < n and (text[j + 1].isalpha() or text[j + 1] == '_'))):
                j += 1
            tokens.append(Token(text[i:j], line, column)); advance(j)
        else:
            pair = text[i:i + 2]
            if pair in {':=', '=>', '->', '<=', '>=', '!=', '==', '<-'}:
                tokens.append(Token(pair, line, column)); advance(i + 2)
            else:
                tokens.append(Token(c, line, column)); advance(i + 1)
    return tokens, gaps


def _contradictions(header):
    """Detect only declared source shapes; this does not interpret Lean propositions."""
    hypotheses = []
    stack = []
    for i, t in enumerate(header):
        if t in {'(', '{', '['}:
            stack.append((i, t))
        elif t in {')', '}', ']'} and stack:
            start, opening = stack.pop()
            interior = header[start + 1:i]
            depth, colon = 0, None
            for j, word in enumerate(interior):
                if word in {'(', '{', '['}: depth += 1
                if word in {')', '}', ']'}: depth -= 1
                if word == ':' and depth == 0:
                    colon = j; break
            if colon is not None:
                hypotheses.append(tuple(interior[colon + 1:]))
    for i, t in enumerate(header[:-1]):
        if t == 'False' and header[i + 1] in {'→', '->'}:
            hypotheses.append(('False',))
    candidates = []
    for h in hypotheses:
        if h in {('False',), ('Not', 'True'), ('¬', 'True')}:
            candidates.append('explicit false hypothesis: ' + ' '.join(h))
        if len(h) == 3 and h[0] == h[2] and h[1] in {'<', '≠', '!='}:
            candidates.append('self-strict/disequality hypothesis: ' + ' '.join(h))
        if len(h) == 3 and re.fullmatch(r'[0-9]+',h[0]) and re.fullmatch(r'[0-9]+',h[2]) and h[1] == '=' and int(h[0]) != int(h[2]):
            candidates.append('incompatible literal equality candidate: ' + ' '.join(h))
        positive = h[1:] if h and h[0] in {'¬', 'Not'} else None
        if positive and positive in hypotheses:
            candidates.append('opposing hypotheses: ' + ' '.join(positive))
    return sorted(set(candidates)), [' '.join(h) for h in hypotheses]


def check_lean(text):
    tokens, findings = lean_tokens(text)
    imports, declarations = [], []
    placeholders = {'sorry', 'admit', 'sorryAx', 'admitGoal', 'mkSorry'}
    native = {'native_decide', 'native', 'ofReduceBool', 'ofReduceNat', 'ofReduceInt', 'trustCompiler', 'skipKernelTC'}
    kernel_bypass = {'addDeclWithoutChecking'}
    expansion = {'macro', 'macro_rules', 'elab', 'elab_rules', 'syntax', 'unsafe', 'run_elab', 'initialize', 'builtin_initialize', 'implemented_by', 'extern'}
    hazards = []
    lexical_errors = []
    for metadata in canonical_identifier_tokens(text, lexical_errors):
        if not isinstance(metadata, tuple) or len(metadata) != 7:
            findings.append(finding('LEAN_LEXICAL_GAP', 'unsupported canonical lexer metadata shape'))
            continue
        raw_name, line, column, first, plus_before, following, field_before = metadata
        # The stricter hazard rules inspect every identifier; native is refused
        # regardless of the canonical classifier's seventh field/context hint.
        parts = raw_name.split('.')
        quoted = parts[-1].startswith('\x00')
        name = '.'.join(part.removeprefix('\x00').replace('\x01', '.') for part in parts)
        hazards.append(Token(name, line, column - 1, quoted))
    for line, column, kind in dict.fromkeys(lexical_errors):
        findings.append(finding('LEAN_LEXICAL_GAP', 'canonical lexer: ' + kind, line))
    for token in hazards:
        word = token.text
        tail = word.rsplit('.', 1)[-1]
        if tail in placeholders and (not token.quoted or tail in {'sorryAx','admitGoal','mkSorry'}):
            findings.append(finding('LEAN_PLACEHOLDER', word, token.line))
        if tail in {'axiom', 'constant', 'constants'}:
            findings.append(finding('LEAN_AXIOM', 'owned/new axiom declaration or reference requires refusal: ' + word, token.line))
        if tail in native:
            findings.append(finding('LEAN_NATIVE', 'compiler/native trust escape: ' + word, token.line))
        if tail in kernel_bypass:
            findings.append(finding('LEAN_KERNEL_BYPASS', 'unchecked environment declaration insertion: ' + word, token.line))
        if tail in expansion:
            findings.append(finding('LEAN_EXPANSION_GAP', 'requires G2/G4 expanded environment certificate: ' + word, token.line))
        if tail in {'Float','Float32'}:
            findings.append(finding('LEAN_FLOAT_REVIEW_GAP', 'floating-point use requires a relevant exact mathematical/nonmathematical certificate', token.line))
    for i, token in enumerate(tokens):
        word = token.text
        if word == 'import':
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if t.line > token.line and t.column == 0:
                    break
                if t.quoted or not re.fullmatch(r"[^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*", t.text, re.UNICODE):
                    findings.append(finding('LEAN_IMPORT_GAP', 'unsupported import operand or trailing syntax: ' + t.text, t.line))
                    break
                imports.append(t.text); j += 1
            if j == i + 1:
                findings.append(finding('LEAN_IMPORT_GAP', 'missing or unsupported import syntax', token.line))
        if word in {'theorem', 'lemma', 'example'}:
            if word == 'example':
                name, j = 'example', i + 1
            else:
                if i + 1 >= len(tokens):
                    findings.append(finding('LEAN_DECLARATION_GAP', 'missing declaration name', token.line)); continue
                name, j = tokens[i + 1].text, i + 2
            header, depth = [], 0
            while j < len(tokens):
                t = tokens[j].text
                if depth == 0 and t in {':=', 'where'}:
                    break
                if depth == 0 and t in {'theorem', 'lemma', 'example'}:
                    break
                if t in {'(', '{', '['}: depth += 1
                if t in {')', '}', ']'}: depth -= 1
                header.append(t); j += 1
            if j == len(tokens) or depth != 0:
                findings.append(finding('LEAN_DECLARATION_GAP', 'unsupported/incomplete theorem header', token.line))
            contradictions, hypotheses = _contradictions(header)
            declarations.append({'key': name + '@' + str(token.line), 'line': token.line,
                                 'contradictions': contradictions, 'hypotheses': hypotheses})
    return {'findings': findings, 'imports': imports, 'declarations': declarations}


EXACT_BUILTINS = frozenset({'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'int', 'len', 'list', 'max',
                            'min', 'range', 'reversed', 'set', 'sorted', 'sum', 'tuple', 'zip', 'frozenset'})
ALLOWED_AST = frozenset({
    'Module','ImportFrom','alias','FunctionDef','arguments','arg','Return','Assign','AnnAssign','AugAssign',
    'Expr','Constant','Name','Load','Store','Del','List','Tuple','Set','Dict','Subscript','Slice',
    'BinOp','UnaryOp','BoolOp','Compare','If','IfExp','For','Break','Continue','Pass','Assert',
    'ListComp','SetComp','DictComp','GeneratorExp','comprehension','Call','keyword','Attribute',
    'Add','Sub','Mult','FloorDiv','Mod','Pow','USub','UAdd','Not','Invert','And','Or',
    'Eq','NotEq','Lt','LtE','Gt','GtE','Is','IsNot','In','NotIn','BitAnd','BitOr','BitXor','LShift','RShift',
    'Div','Import', 'Lambda', 'ClassDef', 'AsyncFunctionDef',
})



def _attach_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._cbz_parent = parent


def _exact_integer_expr(node, names):
    """Exact integer literal, pure integer operator, or module-level established name."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return True
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        return node.id in names and not _name_bound_in_enclosing_scope(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
        return _exact_integer_expr(node.operand, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
                                                            ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
        return _exact_integer_expr(node.left, names) and _exact_integer_expr(node.right, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return (_exact_integer_expr(node.left, names)
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) is int and node.right.value >= 0)
    return False


def _name_bound_in_enclosing_scope(name_node):
    """True when an enclosing function/comprehension/lambda binds the same name (fail closed)."""
    parent = getattr(name_node, '_cbz_parent', None)
    while parent is not None:
        if isinstance(parent, ast.comprehension):
            for target in ast.walk(parent.target):
                if isinstance(target, ast.Name) and target.id == name_node.id:
                    return True
        elif isinstance(parent, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            gens = parent.generators
            for gen in gens:
                for target in ast.walk(gen.target):
                    if isinstance(target, ast.Name) and target.id == name_node.id:
                        return True
        elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = parent.args
            if any(arg.arg == name_node.id for arg in args.args + args.posonlyargs + args.kwonlyargs):
                return True
            if args.vararg and args.vararg.arg == name_node.id:
                return True
            if args.kwarg and args.kwarg.arg == name_node.id:
                return True
            # Any Store/Del of this name in the function is a nested binding (fail closed).
            for node in ast.walk(parent):
                if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id == name_node.id:
                    return True
        elif isinstance(parent, ast.Lambda):
            if any(arg.arg == name_node.id for arg in parent.args.args + parent.args.posonlyargs + parent.args.kwonlyargs):
                return True
        elif isinstance(parent, ast.ClassDef):
            pass
        parent = getattr(parent, '_cbz_parent', None)
    return False


def _module_level_integer_names(tree):
    """Uninterrupted module-level integer assignment chains; nested scopes are ignored here."""
    # Fail closed: any binding outside an accepted top-level plain-Name chain
    # statement (destructuring, for/with/except/walrus/match targets, nested
    # bodies, del, global/nonlocal, def/class/import names) bans the name.
    accepted = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
            accepted.update(id(t) for t in stmt.targets)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(stmt.target, ast.Name):
            accepted.add(id(stmt.target))
    banned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in accepted:
            banned.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            banned.update(node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            banned.add(node.name)
        elif isinstance(node, ast.alias):
            banned.add(node.asname or node.name.split('.')[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            banned.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) if hasattr(ast, 'MatchAs') else False:
            if node.name:
                banned.add(node.name)
        elif hasattr(ast, 'MatchMapping') and isinstance(node, ast.MatchMapping) and node.rest:
            banned.add(node.rest)
    names = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if all(isinstance(t, ast.Name) and t.id not in banned for t in stmt.targets) and _exact_integer_expr(stmt.value, names):
                for t in stmt.targets:
                    names.add(t.id)
            else:
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        names.discard(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None and _exact_integer_expr(stmt.value, names):
                names.add(stmt.target.id)
            else:
                names.discard(stmt.target.id)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            ok = False
            if isinstance(stmt.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
                                    ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
                ok = stmt.target.id in names and _exact_integer_expr(stmt.value, names)
            elif isinstance(stmt.op, ast.Pow):
                ok = (stmt.target.id in names and isinstance(stmt.value, ast.Constant)
                      and type(stmt.value.value) is int and stmt.value.value >= 0)
            if not ok:
                names.discard(stmt.target.id)
    return names - banned


def _exact_integer_mod_operands(node, names):
    """Mod is exact remainder only for integer-origin operands; formatting fails closed."""
    if isinstance(node, ast.BinOp):
        return _exact_integer_expr(node.left, names) and _exact_integer_expr(node.right, names)
    # AugAssign: established integer name %= integer-origin value
    if isinstance(node.target, ast.Name):
        return (node.target.id in names and not _name_bound_in_enclosing_scope(node.target)
                and _exact_integer_expr(node.value, names))
    return False


def check_python(text, mathematical=True):
    findings, imports, import_candidates = [], [], []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {'findings': [finding('PY_SYNTAX', str(exc), getattr(exc, 'lineno', 0) or 0)], 'imports': []}
    _attach_parents(tree)
    integer_names = _module_level_integer_names(tree)
    aliases = {'Fraction'}
    local_functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    dynamic_aliases = {'__import__', 'eval', 'exec', 'compile', 'globals', 'locals', '__builtins__', 'vars', 'getattr', 'setattr', 'delattr'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                if alias.name in {'importlib', 'builtins'}:
                    dynamic_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            name = '.' * node.level + (node.module or '')
            imports.append(name)
            import_candidates.extend(name + '.' + alias.name for alias in node.names if alias.name != '*')
            if node.module == 'fractions' and node.level == 0:
                for alias in node.names:
                    if alias.name == 'Fraction':
                        aliases.add(alias.asname or alias.name)
            if node.module in {'importlib', 'builtins'}:
                for alias in node.names:
                    dynamic_aliases.add(alias.asname or alias.name)
    allowed_calls = EXACT_BUILTINS | aliases | local_functions
    protected = EXACT_BUILTINS | aliases | local_functions | {'__builtins__'}
    for node in ast.walk(tree):
        line = getattr(node, 'lineno', 0)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in dynamic_aliases:
            findings.append(finding('PY_DYNAMIC_IMPORT', 'dynamic executor/importer name requires dependency certifier: ' + node.id, line))
        if isinstance(node, ast.Attribute) and node.attr in {'__builtins__','__globals__','__dict__','__class__','__base__','__bases__','__mro__','__subclasses__','__getattribute__','__getattr__','__code__','__closure__','func_globals','gi_frame','cr_frame','f_globals','f_builtins'}:
            findings.append(finding('PY_DYNAMIC_IMPORT', 'reflective loader entrypoint requires dependency certifier: ' + node.attr, line))
        if isinstance(node, ast.ImportFrom) and (node.level or any(a.name == '*' for a in node.names)):
            findings.append(finding('PY_IMPORT_GAP', 'relative/star import requires explicit dependency certifier', line))
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            dynamic = (name in dynamic_aliases or
                       isinstance(node.func, ast.Attribute) and
                       (node.func.attr in {'import_module','__import__','exec_module','load_module','find_spec','spec_from_file_location'} or
                        isinstance(node.func.value, ast.Name) and node.func.value.id in dynamic_aliases))
            if dynamic:
                findings.append(finding('PY_DYNAMIC_IMPORT', 'dynamic execution/loading cannot establish closed dependency boundary', line))
        if not mathematical:
            continue
        name = type(node).__name__
        if name not in ALLOWED_AST or isinstance(node, (ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef)):
            findings.append(finding('PY_AST_GAP', 'unsupported exact-program AST node: ' + name, line))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in {'float','complex','__builtins__'}:
            findings.append(finding('PY_INEXACT', 'inexact/reflective value can escape through callbacks: ' + node.id, line))
        if isinstance(node, ast.alias) and node.asname in EXACT_BUILTINS:
            findings.append(finding('PY_REBIND', 'import shadows exact builtin: ' + node.asname, line))
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                findings.append(finding('PY_AST_GAP', 'decorators can replace exact functions', line))
            if node.name in EXACT_BUILTINS | aliases | {'__builtins__'}:
                findings.append(finding('PY_REBIND', 'protected arithmetic name rebound: ' + node.name, line))
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in protected:
            findings.append(finding('PY_REBIND', 'protected arithmetic name rebound: ' + node.id, line))
        if isinstance(node, ast.arg) and node.arg in protected:
            findings.append(finding('PY_REBIND', 'protected arithmetic parameter: ' + node.arg, line))
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):
            findings.append(finding('PY_INEXACT', 'float/complex literal forbidden in mathematical program', line))
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Div):
            findings.append(finding('PY_INEXACT', '/ may produce float; use Fraction(n,d)', line))
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
            exponent = node.right if isinstance(node, ast.BinOp) else node.value
            if not (isinstance(exponent, ast.Constant) and type(exponent.value) is int and exponent.value >= 0):
                findings.append(finding('PY_INEXACT', '** requires nonnegative integer literal exponent', line))
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Mod):
            if not _exact_integer_mod_operands(node, integer_names):
                findings.append(finding('PY_INEXACT', '% requires exact integer-origin operands; string/bytes formatting and unestablished values fail closed', line))
        if isinstance(node, ast.Import):
            findings.append(finding('PY_IMPORT_GAP', 'only from fractions import Fraction [as alias] is in the exact profile', line))
        if isinstance(node, ast.ImportFrom):
            if node.module != 'fractions' or node.level or any(a.name != 'Fraction' for a in node.names):
                findings.append(finding('PY_IMPORT_GAP', 'import outside exact fractions profile', line))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:
                findings.append(finding('PY_CALL_GAP', 'call is not a statically named exact builtin/Fraction/local function', line))
            if any(k.arg is None for k in node.keywords) or any(isinstance(a, ast.Starred) for a in node.args):
                findings.append(finding('PY_CALL_GAP', 'dynamic argument expansion is outside exact profile', line))
        if isinstance(node, ast.Attribute) and node.attr not in {'numerator','denominator'}:
            findings.append(finding('PY_ATTRIBUTE_GAP', 'only exact numerator/denominator reads are supported', line))
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            findings.append(finding('PY_ATTRIBUTE_GAP', 'attribute mutation is outside exact profile', line))
        if isinstance(node, ast.comprehension) and node.is_async:
            findings.append(finding('PY_AST_GAP', 'async comprehension is outside exact profile', line))
    return {'findings': findings, 'imports': sorted(set(imports)), 'import_candidates': sorted(set(import_candidates))}
