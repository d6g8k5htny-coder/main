"""Export a dated graph after verifying its provenance and executable gate.

The caller can pin alternate provenance, but executable gate authority is always
the independently reviewed gate digest below. Source files are read once through
regular-file descriptors. The gate is compiled from those verified bytes, never
reopened through an import loader. This export records metadata, not proof or
scientific acceptance.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from types import ModuleType
from typing import Any, Iterator


DEFAULT_PROVENANCE_SHA256 = "c54cd92d93b2cbc650e024f30a8762fbc277d847580245053bf0c415265872c5"
TRUSTED_GATE_SHA256 = "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8"
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_PROVENANCE_BYTES = 64 * 1024
REVIEW_FIELDS = (
    "review_source", "review_issue", "review_basis", "review_provider",
    "review_providers", "review_disposition",
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute(path: str | Path) -> Path:
    # Preserve '..': lexical normalization could erase an earlier symlink
    # component before its descriptor-anchored O_NOFOLLOW check.
    supplied = Path(path)
    return supplied if supplied.is_absolute() else Path.cwd() / supplied


@contextmanager
def _directory(
    path: Path, *, create: bool = False,
    protected: tuple[int, int] | None = None,
) -> Iterator[int]:
    """Anchor every component without following directory symlinks."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        def outside_source() -> None:
            info = os.fstat(descriptor)
            _require(
                (info.st_dev, info.st_ino) != protected,
                "output directory must be outside the immutable source directory",
            )

        outside_source()
        for component in path.parts[1:]:
            outside_source()
            try:
                following = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise ValueError(f"unavailable directory component in path {path}: {component}")
                # The currently anchored parent has been checked before mkdir.
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    following = os.open(component, flags, dir_fd=descriptor)
                except OSError as exc:
                    raise ValueError(
                        f"symlink or invalid directory component in path {path}: {component}"
                    ) from exc
            except OSError as exc:
                raise ValueError(
                    f"symlink or invalid directory component in path {path}: {component}"
                ) from exc
            os.close(descriptor)
            descriptor = following
            outside_source()
        yield descriptor
    finally:
        os.close(descriptor)


def _read_regular(directory: int, name: str, limit: int) -> bytes:
    """Read an anchored regular-file leaf once, without blocking on a FIFO."""
    try:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    except OSError as exc:
        raise ValueError(f"symlink or unavailable source file: {name}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), f"source is not a regular file: {name}")
        _require(info.st_size <= limit, f"source bytes exceed size limit: {name}")
        raw = stream.read(limit + 1)
        _require(len(raw) <= limit, f"source bytes exceed size limit: {name}")
        _require(len(raw) == info.st_size, f"source byte count changed while reading: {name}")
        return raw


def _strict_json(raw: bytes, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant in {name}: {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        _require(math.isfinite(parsed), f"non-finite JSON number in {name}")
        return parsed

    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique,
        parse_constant=reject_constant, parse_float=finite_float,
    )


def _provenance(raw: bytes, expected_digest: str) -> dict[str, Any]:
    _require(
        isinstance(expected_digest, str) and HEX64.fullmatch(expected_digest) is not None,
        "expected provenance SHA256 must be 64 lowercase hexadecimal characters",
    )
    _require(
        hashlib.sha256(raw).hexdigest() == expected_digest,
        "provenance SHA256 identity mismatch",
    )
    record = _strict_json(raw, "PROVENANCE.json")
    _require(isinstance(record, dict), "malformed provenance record")
    _require(
        type(record.get("schema_version")) is int and record["schema_version"] == 1,
        "unsupported provenance schema_version",
    )
    for field in ("repository", "commit", "captured_at"):
        _require(
            isinstance(record.get(field), str) and bool(record[field].strip()),
            f"missing or malformed provenance field: {field}",
        )
    _require(HEX40.fullmatch(record["commit"]) is not None, "malformed provenance commit identity")
    _require(isinstance(record.get("files"), dict), "malformed provenance files")
    return record


def _verify_source(name: str, raw: bytes, provenance: dict[str, Any]) -> str:
    record = provenance["files"].get(name)
    _require(isinstance(record, dict), f"missing provenance file identity: {name}")
    _require(
        type(record.get("bytes")) is int and record["bytes"] >= 0,
        f"malformed provenance byte count: {name}",
    )
    _require(len(raw) == record["bytes"], f"provenance bytes identity mismatch: {name}")
    digest = hashlib.sha256(raw).hexdigest()
    _require(
        isinstance(record.get("sha256"), str) and HEX64.fullmatch(record["sha256"]) is not None,
        f"malformed provenance SHA256: {name}",
    )
    _require(digest == record["sha256"], f"provenance SHA256 identity mismatch: {name}")
    _require(
        isinstance(record.get("git_blob"), str) and HEX40.fullmatch(record["git_blob"]) is not None,
        f"malformed provenance git blob identity: {name}",
    )
    blob = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw, usedforsecurity=False
    ).hexdigest()
    _require(blob == record["git_blob"], f"provenance git blob identity mismatch: {name}")
    return digest


def _verified_gate(raw: bytes, path: Path) -> ModuleType:
    """Execute exactly the reviewed bytes using the source path for attribution."""
    _require(
        hashlib.sha256(raw).hexdigest() == TRUSTED_GATE_SHA256,
        "hard_gate.py trusted gate SHA256 pin mismatch",
    )
    module = ModuleType("_source_bound_proof_graph_gate")
    module.__file__ = str(path)
    # A file-based import loader would reopen the path after verification. Compile
    # the verified buffer instead; the attributed filename still names the source.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _dimensions(nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        node_id: {
            "source": "recorded" if node.get("source") else "not_recorded",
            "review": "recorded" if any(node.get(field) for field in REVIEW_FIELDS) else "not_recorded",
            "kernel": "not_recorded",
            "computation": "not_recorded",
            "alignment": "not_recorded",
        }
        for node_id, node in nodes.items()
    }


def _atomic_output(directory: Path, raw: bytes, protected: tuple[int, int]) -> Path:
    """Promote one completed regular output; failures retain earlier graph bytes."""
    with _directory(directory, create=True, protected=protected) as descriptor:
        try:
            existing = os.stat("graph.json", dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _require(stat.S_ISREG(existing.st_mode), "output graph.json is a symlink or nonregular file")
        temporary = ".graph-export-" + secrets.token_hex(16) + ".tmp"
        created = False
        try:
            output = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644, dir_fd=descriptor,
            )
            created = True
            with os.fdopen(output, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary, "graph.json", src_dir_fd=descriptor, dst_dir_fd=descriptor
            )
            created = False
        finally:
            if created:
                os.unlink(temporary, dir_fd=descriptor)
    return directory / "graph.json"


def export_graph(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_provenance_sha256: str = DEFAULT_PROVENANCE_SHA256,
) -> Path:
    """Validate source custody and gate semantics, then atomically export metadata."""
    source = _absolute(source_dir)
    output = _absolute(output_dir)
    _require(
        output != source and source not in output.parents,
        "output directory must be outside the immutable source directory",
    )
    with _directory(source) as descriptor:
        source_info = os.fstat(descriptor)
        protected = (source_info.st_dev, source_info.st_ino)
        provenance_raw = _read_regular(descriptor, "PROVENANCE.json", MAX_PROVENANCE_BYTES)
        graph_raw = _read_regular(descriptor, "GRAPH.json", MAX_SOURCE_BYTES)
        gate_raw = _read_regular(descriptor, "hard_gate.py", MAX_SOURCE_BYTES)
        # Keep the source descriptor alive through promotion, preventing inode
        # reuse and protecting its actual identity despite alternate spellings.
        provenance = _provenance(provenance_raw, expected_provenance_sha256)
        graph_digest = _verify_source("GRAPH.json", graph_raw, provenance)
        gate_digest = _verify_source("hard_gate.py", gate_raw, provenance)
        _require(gate_digest == TRUSTED_GATE_SHA256, "hard_gate.py trusted gate SHA256 pin mismatch")
        graph = _strict_json(graph_raw, "GRAPH.json")
        gate = _verified_gate(gate_raw, source / "hard_gate.py")
        gate.validate_graph_fail_closed(graph)
        payload = {
            "schema_version": 1,
            "source": {
                "repository": provenance["repository"],
                "commit": provenance["commit"],
                "captured_at": provenance["captured_at"],
                "graph_sha256": graph_digest,
                "gate_sha256": gate_digest,
            },
            "nodes": graph["nodes"],
            "edges": graph["edges"],
            "dimensions": _dimensions(graph["nodes"]),
            "scientific_effect": "NONE",
            "scientific_status_authority": False,
        }
        raw = (json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n").encode("utf-8")
        return _atomic_output(output, raw, protected)
