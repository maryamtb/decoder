"""Python AST parser for extracting symbols and relationships."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from decoder.core.exceptions import ParseError
from decoder.core.models import EdgeType, SymbolType
from decoder.languages.models import (
    CallContext,
    ParsedEdge,
    ParsedSymbol,
    ParseResult,
    TypedVar,
)


class PythonParser:
    """Parser for Python source files using the ast module."""

    def supports(self, file: Path) -> bool:
        """Check if this parser supports the given file."""
        return file.suffix == ".py"

    def parse(self, file: Path) -> ParseResult:
        """Parse a Python file and extract symbols and edges."""
        try:
            source = file.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ParseError(f"Cannot read {file}: {e}") from e

        try:
            tree = ast.parse(source, filename=str(file))
        except SyntaxError as e:
            raise ParseError(f"Syntax error in {file}: {e}") from e

        visitor = _PythonVisitor(file)
        visitor.visit(tree)

        return ParseResult(
            file=file,
            symbols=visitor.symbols,
            edges=visitor.edges,
            imports=visitor.imports,
            star_imports=visitor.star_imports,
            typed_vars=visitor.typed_vars,
            class_bases=visitor.class_bases,
        )


@dataclass
class _Scope:
    """Tracks the current scope during AST traversal."""

    name: str
    qualified_name: str
    type: SymbolType
    line: int


class _PythonVisitor(ast.NodeVisitor):
    """AST visitor that extracts symbols and edges from Python code."""

    def __init__(self, file: Path) -> None:
        self.file = file
        self.symbols: list[ParsedSymbol] = []
        self.edges: list[ParsedEdge] = []
        self.imports: dict[str, str] = {}
        self.star_imports: list[str] = []
        self.typed_vars: list[TypedVar] = []

        self._module_name = self._path_to_module(file)

        self._scope_stack: list[_Scope] = []

        self._current_class: str | None = None

        self.class_bases: dict[str, list[str]] = {}

        self._property_methods: set[str] = set()

        self._current_func_params: dict[str, str] = {}

        self._context_stack: list[CallContext] = []

    def _path_to_module(self, path: Path) -> str:
        """Convert a file path to a module name."""
        parts = list(path.with_suffix("").parts)

        if parts and parts[0] == "src":
            parts = parts[1:]
        return ".".join(parts)

    def _current_scope(self) -> _Scope | None:
        """Get the current scope, or None if at module level."""
        return self._scope_stack[-1] if self._scope_stack else None

    def _current_qualified_name(self) -> str:
        """Get the current qualified name prefix."""
        scope = self._current_scope()
        return scope.qualified_name if scope else self._module_name

    def _make_qualified_name(self, name: str) -> str:
        """Create a fully qualified name for a symbol."""
        return f"{self._current_qualified_name()}.{name}"

    def _get_current_context(self) -> CallContext | None:
        """Get the merged context from the context stack."""
        if not self._context_stack:
            return None

        merged = CallContext()
        for ctx in self._context_stack:
            if ctx.is_conditional:
                merged.is_conditional = True
                if ctx.condition and not merged.condition:
                    merged.condition = ctx.condition
            if ctx.is_loop:
                merged.is_loop = True
                merged.loop_type = ctx.loop_type
            if ctx.is_try_block:
                merged.is_try_block = True
            if ctx.is_except_handler:
                merged.is_except_handler = True
                merged.except_type = ctx.except_type

        if (
            merged.is_conditional
            or merged.is_loop
            or merged.is_try_block
            or merged.is_except_handler
        ):
            return merged
        return None

    def _add_symbol(
        self,
        name: str,
        node: ast.stmt | ast.expr,
        symbol_type: SymbolType,
        end_line: int | None = None,
    ) -> str:
        """Add a symbol and return its qualified name."""
        qualified_name = self._make_qualified_name(name)
        parent_scope = self._current_scope()

        self.symbols.append(
            ParsedSymbol(
                name=name,
                qualified_name=qualified_name,
                file=self.file,
                line=node.lineno,
                end_line=end_line or getattr(node, "end_lineno", None),
                type=symbol_type,
                parent_qualified_name=parent_scope.qualified_name if parent_scope else None,
            )
        )
        return qualified_name

    def _add_edge(
        self,
        callee_name: str,
        line: int,
        edge_type: EdgeType,
        is_self_call: bool = False,
        is_super_call: bool = False,
        is_attribute: bool = False,
        import_source: str | None = None,
    ) -> None:
        """Add an edge from the current scope to a callee."""
        scope = self._current_scope()
        if scope is None:
            caller = self._module_name
        else:
            caller = scope.qualified_name

        self.edges.append(
            ParsedEdge(
                caller_qualified_name=caller,
                callee_name=callee_name,
                call_line=line,
                call_type=edge_type,
                is_self_call=is_self_call,
                is_super_call=is_super_call,
                is_attribute=is_attribute,
                import_source=import_source,
                context=self._get_current_context(),
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        """Handle: import foo, import foo.bar, import foo as f"""
        for alias in node.names:
            module_name = alias.name
            local_name = alias.asname or alias.name.split(".")[0]
            self.imports[local_name] = module_name

            self._add_edge(
                callee_name=module_name,
                line=node.lineno,
                edge_type=EdgeType.IMPORT,
                import_source=module_name,
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle: from foo import bar, from foo import bar as b, from foo import *"""
        module = node.module or ""

        if node.level > 0:
            module = self._resolve_relative_import(node.level, module)

        for alias in node.names:
            if alias.name == "*":
                self.star_imports.append(module)
                self._add_edge(
                    callee_name=f"{module}.*",
                    line=node.lineno,
                    edge_type=EdgeType.IMPORT,
                    import_source=module,
                )
            else:
                local_name = alias.asname or alias.name
                qualified_import = f"{module}.{alias.name}" if module else alias.name
                self.imports[local_name] = qualified_import

                self._add_edge(
                    callee_name=qualified_import,
                    line=node.lineno,
                    edge_type=EdgeType.IMPORT,
                    import_source=module,
                )
        self.generic_visit(node)

    def _resolve_relative_import(self, level: int, module: str) -> str:
        """Resolve a relative import to an absolute module path."""
        parts = self._module_name.split(".")
        if len(parts) < level:
            return module

        base_parts = parts[:-level] if level <= len(parts) else []
        if module:
            return ".".join(base_parts + [module])
        return ".".join(base_parts)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions and inheritance."""
        qualified_name = self._add_symbol(
            name=node.name,
            node=node,
            symbol_type=SymbolType.CLASS,
            end_line=node.end_lineno,
        )

        base_names: list[str] = []
        for base in node.bases:
            base_name = self._get_name_from_node(base)
            if base_name:
                base_names.append(base_name)
                self._add_edge(
                    callee_name=base_name,
                    line=node.lineno,
                    edge_type=EdgeType.INHERIT,
                )
        if base_names:
            self.class_bases[qualified_name] = base_names

        for decorator in node.decorator_list:
            dec_name = self._get_name_from_node(decorator)
            if dec_name:
                self._add_edge(
                    callee_name=dec_name,
                    line=decorator.lineno,
                    edge_type=EdgeType.CALL,
                )

        old_class = self._current_class
        self._current_class = qualified_name
        self._scope_stack.append(
            _Scope(
                name=node.name,
                qualified_name=qualified_name,
                type=SymbolType.CLASS,
                line=node.lineno,
            )
        )

        self.generic_visit(node)

        self._scope_stack.pop()
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function and method definitions."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Common handler for sync and async functions."""
        parent = self._current_scope()
        is_method = parent is not None and parent.type == SymbolType.CLASS

        symbol_type = SymbolType.METHOD if is_method else SymbolType.FUNCTION

        is_property = any(self._get_name_from_node(d) == "property" for d in node.decorator_list)
        if is_property:
            self._property_methods.add(self._make_qualified_name(node.name))

        qualified_name = self._add_symbol(
            name=node.name,
            node=node,
            symbol_type=symbol_type,
            end_line=node.end_lineno,
        )

        self._extract_parameter_types(node, qualified_name)

        for decorator in node.decorator_list:
            dec_name = self._get_name_from_node(decorator)
            if dec_name and dec_name != "property":  # Skip @property itself
                self._add_edge(
                    callee_name=dec_name,
                    line=decorator.lineno,
                    edge_type=EdgeType.CALL,
                )

        old_params = self._current_func_params
        self._current_func_params = {}
        if is_method and node.name == "__init__":
            for arg in node.args.args:
                if arg.annotation and arg.arg not in ("self", "cls"):
                    type_name = self._extract_type_from_annotation(arg.annotation)
                    if type_name:
                        self._current_func_params[arg.arg] = type_name

        self._scope_stack.append(
            _Scope(
                name=node.name,
                qualified_name=qualified_name,
                type=symbol_type,
                line=node.lineno,
            )
        )

        self.generic_visit(node)

        self._scope_stack.pop()
        self._current_func_params = old_params

    def _extract_parameter_types(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, func_qualified_name: str
    ) -> None:
        """Extract type annotations from function parameters."""
        for arg in node.args.args:
            if arg.annotation:
                type_name = self._extract_type_from_annotation(arg.annotation)
                if type_name and arg.arg not in ("self", "cls"):
                    self.typed_vars.append(
                        TypedVar(
                            name=arg.arg,
                            type_name=type_name,
                            scope_qualified_name=func_qualified_name,
                        )
                    )

    def _extract_type_from_annotation(self, node: ast.expr) -> str | None:
        """Extract the actual type from an annotation, handling Annotated types."""
        if isinstance(node, ast.Subscript):
            base_name = self._get_name_from_node(node.value)
            if base_name == "Annotated":
                if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                    return self._get_name_from_node(node.slice.elts[0])
                return self._get_name_from_node(node.slice)
        return self._get_name_from_node(node)

    def _track_self_assignment(self, target: ast.expr, value: ast.expr) -> None:
        """Track self.x = param assignments for type inference."""
        if not self._current_class or not self._current_func_params:
            return
        if not isinstance(target, ast.Attribute):
            return
        if not isinstance(target.value, ast.Name) or target.value.id != "self":
            return
        if not isinstance(value, ast.Name):
            return
        param_name = value.id
        if param_name not in self._current_func_params:
            return

        self.typed_vars.append(
            TypedVar(
                name=f"self.{target.attr}",
                type_name=self._current_func_params[param_name],
                scope_qualified_name=self._current_class,
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle variable assignments."""
        scope = self._current_scope()

        for target in node.targets:
            self._track_self_assignment(target, node.value)

        if scope is None or scope.type == SymbolType.CLASS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._add_symbol(
                        name=target.id,
                        node=node,
                        symbol_type=SymbolType.VARIABLE,
                    )
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self._add_symbol(
                                name=elt.id,
                                node=node,
                                symbol_type=SymbolType.VARIABLE,
                            )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments (e.g., x: int = 5)."""
        scope = self._current_scope()
        if scope is None or scope.type == SymbolType.CLASS:
            if isinstance(node.target, ast.Name):
                self._add_symbol(
                    name=node.target.id,
                    node=node,
                    symbol_type=SymbolType.VARIABLE,
                )
        self.generic_visit(node)

    def _is_super_call(self, node: ast.Call) -> tuple[bool, str | None]:
        """Check if a call is super().method() or super(Class, self).method().

        Returns (is_super, method_name) tuple.
        """
        if not isinstance(node.func, ast.Attribute):
            return False, None

        value = node.func.value
        if not isinstance(value, ast.Call):
            return False, None

        func = value.func
        if isinstance(func, ast.Name) and func.id == "super":
            return True, node.func.attr

        return False, None

    def visit_Call(self, node: ast.Call) -> None:
        """Handle function and method calls."""
        is_super, method_name = self._is_super_call(node)
        if is_super and method_name:
            self._add_edge(
                callee_name=f"super.{method_name}",
                line=node.lineno,
                edge_type=EdgeType.CALL,
                is_super_call=True,
            )
            self.generic_visit(node)
            return

        callee_name = self._get_name_from_node(node.func)
        if callee_name:
            is_self_call = callee_name.startswith("self.")
            self._add_edge(
                callee_name=callee_name,
                line=node.lineno,
                edge_type=EdgeType.CALL,
                is_self_call=is_self_call,
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Handle attribute access (but not calls - those are handled by visit_Call)."""
        if not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)
            return

        # Skip - visit_Call handles function/method calls
        # We only want pure attribute access like obj.attr (not obj.method())
        # This is tracked via _pending_calls set
        self.generic_visit(node)

    def _unparse_condition(self, node: ast.expr) -> str:
        """Convert an AST condition back to a readable string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "<condition>"

    def visit_If(self, node: ast.If) -> None:
        """Track calls inside if/else blocks as conditional."""
        condition_str = self._unparse_condition(node.test)

        self.visit(node.test)

        self._context_stack.append(
            CallContext(
                is_conditional=True,
                condition=condition_str,
            )
        )
        for stmt in node.body:
            self.visit(stmt)
        self._context_stack.pop()

        if node.orelse:
            self._context_stack.append(
                CallContext(
                    is_conditional=True,
                    condition=f"not ({condition_str})" if len(condition_str) < 30 else "else",
                )
            )
            for stmt in node.orelse:
                self.visit(stmt)
            self._context_stack.pop()

    def visit_For(self, node: ast.For) -> None:
        """Track calls inside for loops."""
        self._context_stack.append(
            CallContext(
                is_loop=True,
                loop_type="for",
            )
        )
        self.generic_visit(node)
        self._context_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        """Track calls inside while loops."""
        self.visit(node.test)

        self._context_stack.append(
            CallContext(
                is_loop=True,
                loop_type="while",
            )
        )
        for stmt in node.body:
            self.visit(stmt)
        self._context_stack.pop()

        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_Try(self, node: ast.Try) -> None:
        """Track calls inside try/except blocks."""
        self._context_stack.append(CallContext(is_try_block=True))
        for stmt in node.body:
            self.visit(stmt)
        self._context_stack.pop()

        for handler in node.handlers:
            except_type = None
            if handler.type:
                except_type = self._get_name_from_node(handler.type)
            self._context_stack.append(
                CallContext(
                    is_except_handler=True,
                    except_type=except_type,
                )
            )
            for stmt in handler.body:
                self.visit(stmt)
            self._context_stack.pop()

        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

        if node.finalbody:
            for stmt in node.finalbody:
                self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        """Track calls inside with blocks (context managers)."""
        self._context_stack.append(CallContext(is_try_block=True))
        self.generic_visit(node)
        self._context_stack.pop()

    def _get_name_from_node(self, node: ast.AST | None) -> str | None:
        """Extract a name string from various AST node types."""
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_name_from_node(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_name_from_node(node.func)
        elif isinstance(node, ast.Subscript):
            return self._get_name_from_node(node.value)
        return None
