"""Data models for language parser results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from decoder.core.models import EdgeType, SymbolType


@dataclass
class ParsedSymbol:
    """A symbol extracted from source code (before storage)."""

    name: str
    qualified_name: str
    file: Path
    line: int
    end_line: int | None
    type: SymbolType
    parent_qualified_name: str | None = None


@dataclass
class CallContext:
    """Context about where a call occurs (conditional, loop, etc.)."""

    is_conditional: bool = False
    condition: str | None = None
    is_loop: bool = False
    loop_type: str | None = None
    is_try_block: bool = False
    is_except_handler: bool = False
    except_type: str | None = None


@dataclass
class ParsedEdge:
    """A relationship extracted from source code (before storage)."""

    caller_qualified_name: str
    callee_name: str
    call_line: int
    call_type: EdgeType
    is_self_call: bool = False
    is_super_call: bool = False
    is_attribute: bool = False
    import_source: str | None = None
    context: CallContext | None = None


@dataclass
class TypedVar:
    """A variable with a known type annotation."""

    name: str
    type_name: str
    scope_qualified_name: str


@dataclass
class ParseResult:
    """Result of parsing a file."""

    file: Path
    symbols: list[ParsedSymbol]
    edges: list[ParsedEdge]
    imports: dict[str, str]
    star_imports: list[str]
    typed_vars: list[TypedVar] | None = None
    class_bases: dict[str, list[str]] | None = None
