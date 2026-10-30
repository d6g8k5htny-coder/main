#!/usr/bin/env python3
"""CBZ-0 gate G11 census: stub scan and tool documentation check (stdlib only).

Reads the exact blob bytes of one git revision and reports:

  stub      a Python function whose body, after any docstring, is only `pass`,
            `...` or `raise NotImplementedError`, or is the docstring alone.
            A stub under @abstractmethod or @overload, or in a Protocol, ABC or
            exception class, is reported with that context and does not fail.
  marker    TODO, FIXME, XXX, TBD or STUB (upper case) or "not implemented" in
            a code comment. In Python the comment is found by tokenizing; in
            .mjs/.js/.cjs/.sh a comment is found by a quote-parity heuristic.
            The same words in string data are recorded as mentions only.
  noop      a workflow step whose `run:` does nothing: `true`, `:` or `exit 0`.
  tools     every executable source under tools/ (not tests, not __init__.py),
            with the tracked Markdown files that name it and the evidence that
            something exercises it: a test or workflow naming or importing it,
            or an import from a tool that is itself exercised.

Every path has an area. Only `tooling` (tools/, .github/) and `site`
(docs/site/ outside vendor/) are production; findings elsewhere are reported
and never fail. A production file that is not UTF-8, or Python that does not
parse, fails closed.

Exit status: 0 when nothing in production fails, 1 when something does, 2 when
the revision cannot be read. The census is evidence for a reviewer, never a
G11 certificate.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import posixpath
import shlex
import io
import json
import re
import subprocess
import sys
import tokenize

MARKER = re.compile(r"\b(?:TODO|FIXME|XXX|TBD|STUB)\b|\b(?i:not implemented)\b")
CODE_SUFFIXES = (".py", ".mjs", ".js", ".cjs", ".sh")
SCANNED = CODE_SUFFIXES + (".md", ".yml", ".yaml")
LEGIT_DECORATORS = {"abstractmethod", "overload"}
LEGIT_BASES = {"Protocol", "ABC"}
NOOP_RUN = {"true", ":", "exit 0"}
PRODUCTION = {"tooling", "site"}


def area(path):
    name = path.rsplit("/", 1)[-1]
    if (path.startswith("tests/") or "/tests/" in path or name == "conftest.py"
            or name.startswith("test_") or name.endswith("_test.py") or ".test." in name):
        return "tests"
    if path.startswith("tools/") or path.startswith(".github/"):
        return "tooling"
    if path.startswith("docs/site/vendor/"):
        return "vendor"
    if path.startswith("docs/site/"):
        return "site"
    if path.startswith("experiments/"):
        return "research"
    return "evidence"


def _decorator_name(node):
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _base_name(node):
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""

def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return base + "." + node.attr if base else ""
    return ""


def _target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names = set()
        for item in node.elts:
            names.update(_target_names(item))
        return names
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return set()


def _scope_bindings(scope):
    """Names bound in one lexical scope; nested scope bodies are kept separate."""
    names = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            names.add(arg.arg)
        if args.vararg:
            names.add(args.vararg.arg)
        if args.kwarg:
            names.add(args.kwarg.arg)

    def visit(node, root=False):
        if not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if not root and isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(scope, True)
    return names


def _import_bindings(tree):
    modules, symbols, shadowed = {}, {}, set()
    for node in tree.body:
        if type(node) is ast.Import:
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                modules[bound] = alias.name if alias.asname else alias.name.split(".", 1)[0]
        elif type(node) is ast.ImportFrom and node.level == 0:
            for alias in node.names:
                symbols[alias.asname or alias.name] = (node.module or "", alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                shadowed.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            shadowed.update(_target_names(node.target))
    for name in shadowed:
        modules.pop(name, None)
        symbols.pop(name, None)
    return {"modules": modules, "symbols": symbols, "shadowed": shadowed}


def _resolve_import(node, imports):
    if isinstance(node, ast.Name):
        if node.id in imports["shadowed"]:
            return None
        return imports["symbols"].get(node.id)
    dotted = _dotted_name(node)
    if not dotted:
        return None
    parts = dotted.split(".")
    if parts[0] in imports["shadowed"]:
        return None
    module = imports["modules"].get(parts[0])
    if module is None or len(parts) < 2:
        return None
    resolved = module + "." + ".".join(parts[1:])
    return tuple(resolved.rsplit(".", 1))


def _trusted_decorator(node, imports, scope_shadowed=()):
    target = node
    while isinstance(target, ast.Call):
        target = target.func
    dotted = _dotted_name(target)
    binding = dotted.split(".", 1)[0] if dotted else ""
    if binding and binding in scope_shadowed:
        return False
    resolved = _resolve_import(target, imports)
    return resolved in {("abc", "abstractmethod"), ("typing", "overload"),
                        ("typing_extensions", "overload")}


def _trusted_abstract_base(node, imports):
    resolved = _resolve_import(node, imports)
    return resolved in {("abc", "ABC"), ("typing", "Protocol"),
                        ("typing_extensions", "Protocol")}


def _builtin_exception_base(node, imports):
    resolved = _resolve_import(node, imports)
    name = None
    if resolved and resolved[0] == "builtins":
        name = resolved[1]
    elif isinstance(node, ast.Name) and node.id in builtins.__dict__ and node.id not in imports["shadowed"]:
        name = node.id
    if not name:
        return False
    candidate = getattr(builtins, name, None)
    return isinstance(candidate, type) and issubclass(candidate, BaseException)


def _http_request_handler_base(node, imports):
    return _resolve_import(node, imports) in {
        ("http.server", "BaseHTTPRequestHandler"),
        ("http.server", "SimpleHTTPRequestHandler"),
        ("http.server", "CGIHTTPRequestHandler"),
    }




def _stub_kind(body):
    rest = body
    if rest and isinstance(rest[0], ast.Expr) and isinstance(rest[0].value, ast.Constant) \
            and isinstance(rest[0].value.value, str):
        rest = rest[1:]
        if not rest:
            return "docstring only"
    if len(rest) != 1:
        return None
    stmt = rest[0]
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return "ellipsis"
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return "raise NotImplementedError"
    return None


def _legit_context(func, classes, imports, scope_shadowed=()):
    for dec in func.decorator_list:
        if _trusted_decorator(dec, imports, scope_shadowed):
            return "@" + _decorator_name(dec)
    for cls in reversed(classes):
        for base in cls.bases:
            name = _base_name(base)
            if _trusted_abstract_base(base, imports):
                return "method of a %s class" % name
            if _builtin_exception_base(base, imports):
                return "method of an exception class"
            if _http_request_handler_base(base, imports) and func.name == "log_message":
                return "log_message override that silences http.server request logging"
    return None


def python_stubs(tree, imports=None):
    found = []
    if imports is None:
        imports = _import_bindings(tree)

    def visit(node, classes, scope_shadowed):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, classes + [child], scope_shadowed | _scope_bindings(child))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = _stub_kind(child.body)
                if kind:
                    found.append({"line": child.lineno, "name": child.name, "kind": kind,
                                  "context": _legit_context(child, classes, imports, scope_shadowed)})
                visit(child, [], scope_shadowed | _scope_bindings(child))
            else:
                visit(child, classes, scope_shadowed)
    visit(tree, [], set())
    return found


def python_words(raw):
    markers, mentions = [], []
    for tok in tokenize.tokenize(io.BytesIO(raw).readline):
        if tok.type == tokenize.COMMENT:
            target = markers
        elif tok.type == tokenize.STRING or tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            target = mentions
        else:
            continue
        for m in MARKER.finditer(tok.string):
            target.append({"line": tok.start[0] + tok.string[:m.start()].count("\n"), "word": m.group(0)})
    return markers, mentions


def _comment_start(line, opener):
    # Index where a comment opens outside quotes, by quote parity; -1 if none.
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif line.startswith(opener, i) and (opener != "#" or i == 0 or line[i - 1].isspace()):
            return i
        i += 1
    return -1


def text_words(text, path):
    markers, mentions = [], []
    opener = "#" if path.endswith(".sh") else "//"
    in_block = False
    for number, line in enumerate(text.splitlines(), 1):
        comment_parts, data_parts = [], []
        if opener == "//":
            rest = line
            while rest:
                if in_block:
                    end = rest.find("*/")
                    comment_parts.append(rest if end < 0 else rest[:end])
                    if end < 0:
                        rest = ""
                    else:
                        in_block, rest = False, rest[end + 2:]
                    continue
                line_at, block_at = _comment_start(rest, "//"), _comment_start(rest, "/*")
                starts = [p for p in (line_at, block_at) if p >= 0]
                if not starts:
                    data_parts.append(rest)
                    break
                first = min(starts)
                data_parts.append(rest[:first])
                if first == line_at:
                    comment_parts.append(rest[first:])
                    break
                in_block, rest = True, rest[first + 2:]
        else:
            at = _comment_start(line, "#")
            data_parts.append(line if at < 0 else line[:at])
            if at >= 0:
                comment_parts.append(line[at:])
        for part in comment_parts:
            markers.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(part))
        for part in data_parts:
            mentions.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(part))
    return markers, mentions



def _yaml_comment_start(line):
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if quote == '"' and ch == "\\":
                i += 2
                continue
            if quote == "'" and ch == "'" and i + 1 < len(line) and line[i + 1] == "'":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return i
        i += 1
    return -1


def yaml_words(text):
    markers, mentions = [], []
    for number, line in enumerate(text.splitlines(), 1):
        at = _yaml_comment_start(line)
        data = line if at < 0 else line[:at]
        comment = "" if at < 0 else line[at + 1:]
        mentions.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(data))
        markers.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(comment))
    return markers, mentions


def _strip_shell_comment(line):
    at = _comment_start(line, "#")
    return line if at < 0 else line[:at]


def _workflow_commands(text):
    found, lines = [], text.splitlines()
    for number, line in enumerate(lines, 1):
        m = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*?)\s*$", line)
        if not m:
            continue
        value = m.group(2).strip()
        if value.startswith("#"):
            value = ""
        if not value or value[0] in "|>":
            key_column = line.index("run:")
            body = []
            for nxt in lines[number:]:
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= key_column:
                    break
                body.append(nxt.strip())
            if value and value[0] in "|>":
                commands = [b for b in body if b and not b.startswith("#")]
                commands = [_strip_shell_comment(b).strip() for b in commands]
                commands = [b for b in commands if b]
            else:
                commands = []
        else:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            commands = [_strip_shell_comment(value).strip()]
            commands = [_strip_shell_comment(c).strip() for c in commands]
            commands = [c for c in commands if c]
        found.append((number, commands))
    return found


def noop_steps(text):
    found = []
    for number, commands in _workflow_commands(text):
        if not commands:
            found.append({"line": number, "run": "<empty>"})
        if commands and all(c in NOOP_RUN for c in commands):
            found.append({"line": number, "run": " ; ".join(commands)})
    return found


def tool_paths(files):
    return sorted(p for p in files if p.startswith("tools/") and p.endswith(CODE_SUFFIXES)
                  and area(p) == "tooling" and not p.endswith("__init__.py"))


def _module(path):
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _names_tool(text, path, unique):
    # The full path always counts. The module name and the basename count only
    # when no other tracked file shares the basename, so neither is ambiguous.
    pattern = r"\b" + re.escape(path) + r"\b"
    if unique:
        module = re.escape(_module(path))
        pattern += (r"|\b(?:import|from)\s+(?:[\w.]+\.)?" + module + r"\b"
                    r"|\bfrom\s+[\w.]+\s+import\s+(?:\([^)]*|[^\n]*)\b" + module + r"\b"
                    r"|\b" + re.escape(path.rsplit("/", 1)[-1]) + r"\b")
    return re.search(pattern, text) is not None



def _tool_for_module(module, tools, unique):
    if not module:
        return None
    module = module.strip(".")
    candidate = module.replace(".", "/") + ".py"
    if candidate in tools:
        return candidate
    basename = module.rsplit(".", 1)[-1]
    if unique.get(basename):
        for path in tools:
            if path.rsplit("/", 1)[-1] == basename + ".py":
                return path
    return None


def _tool_for_token(token, tools):
    token = token.strip().strip("'\"").replace("\\", "/")
    while token.startswith("./"):
        token = token[2:]
    if token in tools:
        return token
    for path in tools:
        if token.endswith("/" + path):
            return path
    return None


def python_tool_references(text, tools, unique):
    references = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return references
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                path = _tool_for_module(alias.name, tools, unique)
                if path:
                    references.add(path)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            path = _tool_for_module(module, tools, unique)
            if path:
                references.add(path)
            else:
                for alias in node.names:
                    path = _tool_for_module((module + "." + alias.name).strip("."), tools, unique)
                    if path:
                        references.add(path)
        elif isinstance(node, ast.Call):
            call_name = _dotted_name(node.func)
            if call_name in {"subprocess.run", "subprocess.call", "subprocess.Popen",
                             "subprocess.check_call", "subprocess.check_output", "os.system",
                             "runpy.run_path"}:
                for arg in node.args:
                    if isinstance(arg, (ast.List, ast.Tuple)):
                        tokens = [elt.value for elt in arg.elts
                                  if isinstance(elt, ast.Constant) and isinstance(elt.value, str)]
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        try:
                            tokens = shlex.split(arg.value)
                        except ValueError:
                            tokens = []
                    else:
                        tokens = []
                    if tokens and tokens[0] in {"python", "python3", "python3.11", "bash", "sh"}:
                        # Inline -c payloads are code strings, not proof that a path was executed.
                        if "-c" in tokens[1:]:
                            continue
                        for token in tokens[1:]:
                            if token == "-m":
                                continue
                            path = _tool_for_token(token, tools)
                            if path:
                                references.add(path)
                                break
                    elif tokens:
                        path = _tool_for_token(tokens[0], tools)
                        if path:
                            references.add(path)
    return references


def _strip_js_comments(text):
    out = []
    i = 0
    quote = None
    line_comment = False
    block_comment = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                out.extend((" ", " "))
                block_comment = False
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if quote:
            if quote == "`":
                if ch == "\\" and i + 1 < len(text):
                    out.extend((" ", " "))
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                out.append("\n" if ch == "\n" else " ")
                i += 1
                continue
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'" + chr(96):
            quote = ch
            out.append(" " if ch == chr(96) else ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            out.extend((" ", " "))
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            out.extend((" ", " "))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def js_tool_references(text, source_path, tools):
    references = set()
    visible = _strip_js_comments(text)
    pattern = re.compile(r"(?m)^\s*import\s+(?:[^'\";]+?\s+from\s+)?['\"]([^'\"]+)['\"]\s*;?")
    for match in pattern.finditer(visible):
        specifier = match.group(1)
        if specifier.startswith("."):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), specifier))
        else:
            target = specifier
        if target in tools:
            references.add(target)
    return references


def workflow_tool_references(text, tools):
    references = set()
    for _number, commands in _workflow_commands(text):
        for command in commands:
            try:
                tokens = shlex.split(command)
            except ValueError:
                continue
            if not tokens:
                continue
            if tokens[0] in {"python", "python3", "python3.11", "bash", "sh"}:
                # Do not treat a path mentioned inside an inline -c program as an invocation.
                if "-c" in tokens[1:]:
                    continue
                for index, token in enumerate(tokens[1:], 1):
                    if token == "-m" and index + 1 < len(tokens):
                        module_path = tokens[index + 1].replace(".", "/") + ".py"
                        if module_path in tools:
                            references.add(module_path)
                        break
                    if token.startswith("-"):
                        continue
                    path = _tool_for_token(token, tools)
                    if path:
                        references.add(path)
                    break
            else:
                path = _tool_for_token(tokens[0], tools)
                if path:
                    references.add(path)
    return references


def _markdown_visible(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def tool_report(files, texts, all_paths):
    tools = tool_paths(files)
    basenames = {}
    for p in all_paths:
        name = p.rsplit("/", 1)[-1]
        basenames[name] = basenames.get(name, 0) + 1
    unique = {p: basenames[p.rsplit("/", 1)[-1]] == 1 for p in tools}
    unique_modules = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0]: unique[p] for p in tools}
    docs = [p for p in texts if p.endswith(".md")]
    exercisers = [p for p in texts if area(p) == "tests" or
                  (p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml")))]
    direct, tool_dependencies = {p: [] for p in tools}, {}
    for path in tools:
        tool_dependencies[path] = python_tool_references(texts.get(path, ""), tools, unique_modules)
    for exerciser in exercisers:
        if exerciser.endswith(".py"):
            refs = python_tool_references(texts[exerciser], tools, unique_modules)
        elif exerciser.endswith((".mjs", ".js", ".cjs")):
            refs = js_tool_references(texts[exerciser], exerciser, tools)
        else:
            refs = workflow_tool_references(texts[exerciser], tools)
        for path in refs:
            direct[path].append(exerciser)
    report = {}
    for path in tools:
        documented = [d for d in docs if _names_tool(_markdown_visible(texts[d]), path, unique[path])]
        direct_paths = direct[path]
        docstring = None
        if path.endswith(".py") and path in texts:
            try:
                docstring = bool(ast.get_docstring(ast.parse(files[path])))
            except (SyntaxError, ValueError):
                docstring = False
        report[path] = {"documented_by": documented[:5], "documented": bool(documented),
                        "exercised_by": direct_paths[:5], "exercised": bool(direct_paths),
                        "module_docstring": docstring}
    changed = True
    while changed:
        changed = False
        for path in tools:
            if report[path]["exercised"] or path not in texts:
                continue
            for other in tools:
                if other != path and report[other]["exercised"] and other in texts and path in tool_dependencies.get(other, set()):
                    report[path]["exercised"] = True
                    report[path]["exercised_by"] = ["via " + other]
                    changed = True
                    break
    return report


def census(files, all_paths=None):
    """Census of a {path: bytes} tree. Pure: no I/O. `all_paths` lists every
    tracked path (default: the keys of `files`) for basename ambiguity."""
    findings, texts = [], {}
    for path in sorted(files):
        raw = files[path]
        kind = area(path)
        if not path.endswith(SCANNED):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"path": path, "area": kind, "class": "undecodable", "line": 0, "detail": "not UTF-8"})
            continue
        texts[path] = text
        if path.endswith(".py"):
            try:
                tree = ast.parse(raw, filename=path)
                markers, mentions = python_words(raw)
            except (SyntaxError, ValueError, tokenize.TokenError) as exc:
                findings.append({"path": path, "area": kind, "class": "unparseable", "line": 0, "detail": str(exc)[:120]})
                continue
            imports = _import_bindings(tree)
            for stub in python_stubs(tree, imports):
                findings.append({"path": path, "area": kind, "class": "stub" if not stub["context"] else "stub-context",
                                 "line": stub["line"], "detail": "%s: %s%s" % (stub["name"], stub["kind"],
                                                                              "; " + stub["context"] if stub["context"] else "")})
        elif path.endswith((".mjs", ".js", ".cjs", ".sh")):
            markers, mentions = text_words(text, path)
        elif path.endswith((".yml", ".yaml")):
            markers, mentions = yaml_words(text)
        else:
            markers, mentions = [], []
        findings.extend({"path": path, "area": kind, "class": "marker", "line": m["line"], "detail": m["word"]} for m in markers)
        findings.extend({"path": path, "area": kind, "class": "mention", "line": m["line"], "detail": m["word"]} for m in mentions)
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
            findings.extend({"path": path, "area": kind, "class": "noop", "line": n["line"], "detail": n["run"]}
                            for n in noop_steps(text))
    tools = tool_report(files, texts, sorted(all_paths if all_paths is not None else files))
    failing = [f for f in findings if f["area"] in PRODUCTION
               and f["class"] in {"stub", "marker", "noop", "undecodable", "unparseable"}]
    failing_tools = sorted(p for p, t in tools.items() if not (t["documented"] and t["exercised"]))
    return {"findings": findings, "tools": tools, "failing": failing, "failing_tools": failing_tools,
            "status": "FAIL" if failing or failing_tools else "PASS"}


def read_tree(root, rev):
    listing = subprocess.run(["git", "-C", root, "ls-tree", "-r", "-z", "--full-tree", rev],
                             check=True, capture_output=True).stdout
    entries, skipped, tracked = [], [], []
    for item in listing.split(b"\0"):
        if not item:
            continue
        meta, path = item.split(b"\t", 1)
        mode, kind, sha = meta.split()
        path = path.decode("utf-8", "surrogateescape")
        if kind == b"blob" and mode != b"120000":
            if path.endswith(SCANNED):
                entries.append((path, sha.decode()))
            tracked.append(path)
        else:
            skipped.append({"path": path, "mode": mode.decode(), "type": kind.decode()})
    batch = subprocess.run(["git", "-C", root, "cat-file", "--batch"], check=True, capture_output=True,
                           input=b"".join(sha.encode() + b"\n" for _p, sha in entries)).stdout
    files, at = {}, 0
    for path, sha in entries:
        end = batch.index(b"\n", at)
        header = batch[at:end].split()
        if len(header) != 3 or header[0].decode() != sha:
            raise ValueError("unexpected git cat-file header for " + path)
        size = int(header[2])
        files[path] = batch[end + 1:end + 1 + size]
        at = end + 1 + size + 1
    return files, skipped, tracked


def main(argv=None):
    parser = argparse.ArgumentParser(description="CBZ-0 G11 census: stub scan and tool documentation check.")
    parser.add_argument("--root", default=".", help="git repository (default: current directory)")
    parser.add_argument("--rev", default="HEAD", help="revision to read (default: HEAD)")
    parser.add_argument("--json", help="write the full report to this file")
    args = parser.parse_args(argv)
    try:
        commit = subprocess.run(["git", "-C", args.root, "rev-parse", "--verify", args.rev + "^{commit}"],
                                check=True, capture_output=True, text=True).stdout.strip()
        files, skipped, tracked = read_tree(args.root, commit)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print("G11_CENSUS_ERROR: cannot read revision %s: %s" % (args.rev, exc), file=sys.stderr)
        return 2
    report = census(files, tracked)
    report.update(commit=commit, files=len(files), tracked=len(tracked), skipped=skipped)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
            fh.write("\n")
    for f in report["failing"]:
        print("%s:%d: %s %s (%s)" % (f["path"], f["line"], f["area"], f["class"], f["detail"]))
    for path in report["failing_tools"]:
        t = report["tools"][path]
        print("%s: tool %s" % (path, ", ".join(w for w, ok in (("undocumented", t["documented"]),
                                                              ("unexercised", t["exercised"])) if not ok)))
    counts = {}
    for f in report["findings"]:
        counts[f["class"]] = counts.get(f["class"], 0) + 1
    print("G11_CENSUS_%s: commit %s, %d files, %d tools, findings %s, failing %d, failing tools %d"
          % (report["status"], commit, len(files), len(report["tools"]), json.dumps(counts, sort_keys=True),
             len(report["failing"]), len(report["failing_tools"])))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())