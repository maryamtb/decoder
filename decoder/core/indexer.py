"""Indexer that coordinates parsing and storage."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from pathlib import Path

from decoder.core.exceptions import ParseError, SymbolNotFoundError
from decoder.core.models import IndexStats, SymbolType
from decoder.core.storage import SymbolRepository, compute_file_hash
from decoder.languages import ParsedEdge, ParseResult, PythonParser

ProgressCallback = Callable[[Path, int, int], None]

_SELF_PREFIX = "self."

DEFAULT_EXCLUDES = [
    "__pycache__",
    "*.egg-info",
    "node_modules",
    "build",
    "dist",
    "venv",
    ".venv",
]


class Indexer:
    """Coordinates file parsing and symbol storage."""

    def __init__(self, repo: SymbolRepository) -> None:
        """Initialize with a symbol repository."""
        self._repo = repo
        self._parser = PythonParser()

        self._symbol_cache: dict[str, int] = {}

    def index_directory(
        self,
        directory: Path,
        exclude_patterns: list[str] | None = None,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IndexStats:
        """Index all Python files in a directory.

        Uses a two-pass approach:
        1. First pass: Parse all files and insert symbols
        2. Second pass: Resolve and insert all edges

        This ensures cross-file references (e.g., typed parameter calls) resolve correctly
        regardless of file processing order.

        Args:
            directory: Directory to index
            exclude_patterns: Additional glob patterns to exclude (e.g., "tests/*")
            force: If True, re-index all files regardless of hash
            on_progress: Optional callback for progress updates (file, current, total)

        Returns:
            IndexStats with counts of files/symbols/edges processed
        """
        all_excludes = DEFAULT_EXCLUDES + (exclude_patterns or [])
        stats = IndexStats()

        python_files = list(directory.rglob("*.py"))
        total_files = len(python_files)

        parse_results: list[tuple[Path, ParseResult]] = []

        for i, file in enumerate(python_files):
            relative_path = str(file.relative_to(directory))
            if self._should_exclude(relative_path, all_excludes):
                stats.skipped += 1
                if on_progress:
                    on_progress(file, i + 1, total_files)
                continue

            if not force and not self._repo.files.needs_reindex(file):
                stats.unchanged += 1
                if on_progress:
                    on_progress(file, i + 1, total_files)
                continue

            try:
                self._repo.edges.delete_for_file(file)
                self._repo.symbols.delete_in_file(file)

                result = self._parser.parse(file)
                for parsed_symbol in result.symbols:
                    symbol_id = self._repo.symbols.insert(
                        name=parsed_symbol.name,
                        qualified_name=parsed_symbol.qualified_name,
                        file=parsed_symbol.file,
                        line=parsed_symbol.line,
                        symbol_type=parsed_symbol.type,
                        end_line=parsed_symbol.end_line,
                        parent_id=self._get_parent_id(parsed_symbol.parent_qualified_name),
                    )
                    self._symbol_cache[parsed_symbol.qualified_name] = symbol_id
                    stats.symbols += 1

                file_hash = compute_file_hash(file)
                self._repo.files.upsert(file, file_hash)

                parse_results.append((file, result))
                stats.files += 1

            except ParseError as e:
                stats.errors.append(str(e))

            if on_progress:
                on_progress(file, i + 1, total_files)

        for file, result in parse_results:
            for parsed_edge in result.edges:
                callee_id = self._resolve_callee(
                    parsed_edge.callee_name,
                    parsed_edge.caller_qualified_name,
                    result,
                )
                if callee_id is not None:
                    caller_id = self._symbol_cache.get(parsed_edge.caller_qualified_name)
                    if caller_id is not None:
                        self._insert_edge(caller_id, callee_id, parsed_edge)
                        stats.edges += 1

        return stats

    def index_file(self, file: Path) -> IndexStats:
        """Index a single file.

        Returns:
            IndexStats for this file
        """
        stats = IndexStats()
        stats.files = 1

        self._repo.edges.delete_for_file(file)
        self._repo.symbols.delete_in_file(file)

        result = self._parser.parse(file)

        for parsed_symbol in result.symbols:
            symbol_id = self._repo.symbols.insert(
                name=parsed_symbol.name,
                qualified_name=parsed_symbol.qualified_name,
                file=parsed_symbol.file,
                line=parsed_symbol.line,
                symbol_type=parsed_symbol.type,
                end_line=parsed_symbol.end_line,
                parent_id=self._get_parent_id(parsed_symbol.parent_qualified_name),
            )
            self._symbol_cache[parsed_symbol.qualified_name] = symbol_id
            stats.symbols += 1

        for parsed_edge in result.edges:
            callee_id = self._resolve_callee(
                parsed_edge.callee_name,
                parsed_edge.caller_qualified_name,
                result,
            )
            if callee_id is not None:
                caller_id = self._symbol_cache.get(parsed_edge.caller_qualified_name)
                if caller_id is not None:
                    self._insert_edge(caller_id, callee_id, parsed_edge)
                    stats.edges += 1

        file_hash = compute_file_hash(file)
        self._repo.files.upsert(file, file_hash)

        return stats

    def _should_exclude(self, path: str, patterns: list[str]) -> bool:
        """Check if a path matches any exclusion pattern.

        Excludes:
        - Any path component starting with '.' (hidden files/directories)
        - Any path component matching the exclusion patterns
        - Full path matching the exclusion patterns
        """
        parts = Path(path).parts
        for part in parts:
            # Skip hidden files and directories
            if part.startswith("."):
                return True
            # Check against exclusion patterns
            for pattern in patterns:
                if fnmatch.fnmatch(part, pattern):
                    return True
                
        # Full-path matching required for glob patterns with path separators.
        # Component-level matching handles "build", but not "build/*" or "**/*.pyc" 
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            
        return False

    def _get_parent_id(self, parent_qualified_name: str | None) -> int | None:
        """Get the ID of a parent symbol."""
        if parent_qualified_name is None:
            return None
        return self._symbol_cache.get(parent_qualified_name)

    def _insert_edge(self, caller_id: int, callee_id: int, parsed_edge: ParsedEdge) -> None:
        """Insert an edge with context information."""
        ctx = parsed_edge.context
        self._repo.edges.insert(
            caller_id=caller_id,
            callee_id=callee_id,
            call_line=parsed_edge.call_line,
            call_type=parsed_edge.call_type,
            is_conditional=ctx.is_conditional if ctx else False,
            condition=ctx.condition if ctx else None,
            is_loop=ctx.is_loop if ctx else False,
            is_try_block=ctx.is_try_block if ctx else False,
            is_except_handler=ctx.is_except_handler if ctx else False,
        )

    def _resolve_callee(
        self,
        callee_name: str,
        caller_qualified_name: str,
        parse_result: ParseResult,
    ) -> int | None:
        """Resolve a callee name to a symbol ID.

        This handles:
        - Direct references (already in cache)
        - self.method calls (resolve within current class)
        - self.attr.method calls (resolve via instance variable type)
        - Typed parameter calls (service.method where service: TodoService)
        - Imported names (look up via imports dict)
        - Qualified names (module.func)
        """
        if callee_name in self._symbol_cache:
            return self._symbol_cache[callee_name]

        if callee_name.startswith(_SELF_PREFIX):
            method_name = callee_name[len(_SELF_PREFIX) :]

            if "." in method_name:
                resolved_id = self._resolve_instance_var_call(
                    callee_name, caller_qualified_name, parse_result
                )
                if resolved_id is not None:
                    return resolved_id

            class_name = self._get_enclosing_class(caller_qualified_name)
            if class_name:
                method_qualified = f"{class_name}.{method_name}"
                if method_qualified in self._symbol_cache:
                    return self._symbol_cache[method_qualified]

        if "." in callee_name:
            resolved_id = self._resolve_typed_call(callee_name, caller_qualified_name, parse_result)
            if resolved_id is not None:
                return resolved_id

        first_part = callee_name.split(".")[0]
        if first_part in parse_result.imports:
            imported_module = parse_result.imports[first_part]

            rest = callee_name[len(first_part) :]
            resolved_name = imported_module + rest
            if resolved_name in self._symbol_cache:
                return self._symbol_cache[resolved_name]

        try:
            symbol = self._repo.symbols.get_by_qualified_name(callee_name)
            return symbol.id
        except SymbolNotFoundError:
            pass

        return None

    def _resolve_instance_var_call(
        self,
        callee_name: str,
        caller_qualified_name: str,
        parse_result: ParseResult,
    ) -> int | None:
        """Resolve a method call on an instance variable (e.g., self._repo.create).

        Args:
            callee_name: The call like "self._repo.create"
            caller_qualified_name: The method containing the call
            parse_result: Parse result with typed_vars (including instance vars)

        Returns:
            Symbol ID if resolved, None otherwise
        """
        if not parse_result.typed_vars:
            return None

        parts = callee_name.split(".")
        if len(parts) < 3 or parts[0] != "self":
            return None

        attr_name = f"self.{parts[1]}"
        method_name = ".".join(parts[2:])

        class_name = self._get_enclosing_class(caller_qualified_name)
        if not class_name:
            return None

        var_type = None
        for tv in parse_result.typed_vars:
            if tv.name == attr_name and tv.scope_qualified_name == class_name:
                var_type = tv.type_name
                break

        if not var_type:
            return None

        resolved_type = var_type
        if var_type in parse_result.imports:
            resolved_type = parse_result.imports[var_type]

        method_qualified = f"{resolved_type}.{method_name}"
        if method_qualified in self._symbol_cache:
            return self._symbol_cache[method_qualified]

        try:
            symbol = self._repo.symbols.get_by_qualified_name(method_qualified)
            return symbol.id
        except SymbolNotFoundError:
            pass

        for s in self._repo.symbols.find(method_name):
            if s.qualified_name.endswith(f".{var_type}.{method_name}"):
                return s.id

        return None

    def _resolve_typed_call(
        self,
        callee_name: str,
        caller_qualified_name: str,
        parse_result: ParseResult,
    ) -> int | None:
        """Resolve a method call on a typed variable (e.g., service.create_todo).

        Args:
            callee_name: The call like "service.create_todo"
            caller_qualified_name: The function containing the call
            parse_result: Parse result with typed_vars

        Returns:
            Symbol ID if resolved, None otherwise
        """
        if not parse_result.typed_vars:
            return None

        parts = callee_name.split(".", 1)
        if len(parts) != 2:
            return None

        var_name, method_name = parts

        var_type = None
        for tv in parse_result.typed_vars:
            if tv.name == var_name and tv.scope_qualified_name == caller_qualified_name:
                var_type = tv.type_name
                break

        if not var_type:
            return None

        resolved_type = var_type
        if var_type in parse_result.imports:
            resolved_type = parse_result.imports[var_type]

        method_qualified = f"{resolved_type}.{method_name}"
        if method_qualified in self._symbol_cache:
            return self._symbol_cache[method_qualified]

        try:
            symbol = self._repo.symbols.get_by_qualified_name(method_qualified)
            return symbol.id
        except SymbolNotFoundError:
            pass

        for s in self._repo.symbols.find(method_name):
            if s.qualified_name.endswith(f".{var_type}.{method_name}"):
                return s.id

        return None

    def _get_enclosing_class(self, qualified_name: str) -> str | None:
        """Get the enclosing class name from a qualified name."""
        parts = qualified_name.rsplit(".", 1)
        if len(parts) < 2:
            return None

        parent = parts[0]
        if parent in self._symbol_cache:
            try:
                symbol = self._repo.symbols.get_by_qualified_name(parent)
                if symbol.type == SymbolType.CLASS:
                    return parent
            except SymbolNotFoundError:
                pass

        return self._get_enclosing_class(parent)
