"""Unit tests for the Python AST parser."""

import pytest
from pathlib import Path
from decoder.languages.python import PythonParser
from decoder.core.models import EdgeType, SymbolType


def parse(code: str):
    """Helper to parse a code string directly."""
    tmp = Path("/tmp/test_parser_sample.py")
    tmp.write_text(code)
    return PythonParser().parse(tmp)


class TestSymbolExtraction:
    def test_extract_function(self):
        result = parse("def hello():\n    pass\n")
        names = [s.name for s in result.symbols]
        assert "hello" in names

    def test_extract_class(self):
        result = parse("class MyClass:\n    pass\n")
        names = [s.name for s in result.symbols]
        assert "MyClass" in names

    def test_extract_method(self):
        result = parse("class MyClass:\n    def my_method(self):\n        pass\n")
        symbols = {s.name: s for s in result.symbols}
        assert "my_method" in symbols
        assert symbols["my_method"].type == SymbolType.METHOD

    def test_extract_module_variable(self):
        result = parse("MY_VAR = 42\n")
        names = [s.name for s in result.symbols]
        assert "MY_VAR" in names

    def test_extract_async_function(self):
        result = parse("async def fetch():\n    pass\n")
        names = [s.name for s in result.symbols]
        assert "fetch" in names

    def test_function_symbol_type(self):
        result = parse("def hello():\n    pass\n")
        symbols = {s.name: s for s in result.symbols}
        assert symbols["hello"].type == SymbolType.FUNCTION

    def test_class_symbol_type(self):
        result = parse("class MyClass:\n    pass\n")
        symbols = {s.name: s for s in result.symbols}
        assert symbols["MyClass"].type == SymbolType.CLASS


class TestEdgeExtraction:
    def test_direct_call(self):
        result = parse("def foo():\n    pass\n\ndef bar():\n    foo()\n")
        callees = [e.callee_name for e in result.edges]
        assert "foo" in callees

    def test_method_call(self):
        result = parse("class A:\n    def go(self):\n        self.helper()\n    def helper(self):\n        pass\n")
        self_calls = [e for e in result.edges if e.is_self_call]
        assert any("helper" in e.callee_name for e in self_calls)

    def test_chained_call(self):
        result = parse("def foo():\n    obj.method.sub()\n")
        callees = [e.callee_name for e in result.edges]
        assert any("method" in c for c in callees)

    def test_inheritance_edge(self):
        result = parse("class Child(Parent):\n    pass\n")
        inherit_edges = [e for e in result.edges if e.call_type == EdgeType.INHERIT]
        assert any("Parent" in e.callee_name for e in inherit_edges)


class TestImportResolution:
    def test_regular_import(self):
        result = parse("import os\n")
        assert "os" in result.imports

    def test_from_import(self):
        result = parse("from pathlib import Path\n")
        assert "Path" in result.imports
        assert result.imports["Path"] == "pathlib.Path"

    def test_aliased_import(self):
        result = parse("import numpy as np\n")
        assert "np" in result.imports
        assert result.imports["np"] == "numpy"

    def test_star_import(self):
        result = parse("from os.path import *\n")
        assert "os.path" in result.star_imports

    def test_import_edge(self):
        result = parse("import os\n")
        import_edges = [e for e in result.edges if e.call_type == EdgeType.IMPORT]
        assert any("os" in e.callee_name for e in import_edges)


class TestContextTracking:
    def test_call_inside_if(self):
        code = "def foo():\n    if True:\n        bar()\n"
        result = parse(code)
        conditional_edges = [e for e in result.edges if e.context and e.context.is_conditional]
        assert any("bar" in e.callee_name for e in conditional_edges)

    def test_call_inside_for_loop(self):
        code = "def foo():\n    for i in range(10):\n        bar()\n"
        result = parse(code)
        loop_edges = [e for e in result.edges if e.context and e.context.is_loop]
        assert any("bar" in e.callee_name for e in loop_edges)

    def test_call_inside_while_loop(self):
        code = "def foo():\n    while True:\n        bar()\n"
        result = parse(code)
        loop_edges = [e for e in result.edges if e.context and e.context.is_loop]
        assert any("bar" in e.callee_name for e in loop_edges)

    def test_call_inside_try(self):
        code = "def foo():\n    try:\n        bar()\n    except:\n        pass\n"
        result = parse(code)
        try_edges = [e for e in result.edges if e.context and e.context.is_try_block]
        assert any("bar" in e.callee_name for e in try_edges)


class TestDecoratorHandling:
    def test_single_decorator(self):
        code = "@my_decorator\ndef foo():\n    pass\n"
        result = parse(code)
        callees = [e.callee_name for e in result.edges]
        assert "my_decorator" in callees

    def test_decorator_with_args(self):
        code = "@decorator(arg)\ndef foo():\n    pass\n"
        result = parse(code)
        callees = [e.callee_name for e in result.edges]
        assert "decorator" in callees

    def test_stacked_decorators(self):
        code = "@decorator_one\n@decorator_two\ndef foo():\n    pass\n"
        result = parse(code)
        callees = [e.callee_name for e in result.edges]
        assert "decorator_one" in callees
        assert "decorator_two" in callees


class TestTypeAnnotations:
    def test_parameter_type(self):
        code = "def foo(x: int):\n    pass\n"
        result = parse(code)
        names = [t.name for t in result.typed_vars]
        assert "x" in names

    def test_return_type_ignored(self):
        code = "def foo() -> str:\n    pass\n"
        result = parse(code)
        assert isinstance(result.typed_vars, list)

    def test_annotated_variable(self):
        code = "x: int = 5\n"
        result = parse(code)
        names = [s.name for s in result.symbols]
        assert "x" in names
