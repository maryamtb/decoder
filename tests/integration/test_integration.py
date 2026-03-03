"""Integration tests for parser and storage."""

import tempfile
from pathlib import Path

import pytest

from decoder.core.indexer import Indexer
from decoder.core.models import EdgeType, SymbolType
from decoder.core.storage import SymbolRepository, get_default_db_path
from decoder.languages.python import PythonParser


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    code = """
class OrderStatus:
    PENDING = "pending"
    SHIPPED = "shipped"


class Order:
    def __init__(self, status: OrderStatus):
        self.status = status

    def can_ship(self) -> bool:
        return self.status == OrderStatus.PENDING


def create_order() -> Order:
    return Order(OrderStatus.PENDING)


def process_order(order: Order) -> None:
    if order.can_ship():
        print("Shipping order")
"""
    file_path = temp_dir / "orders.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def repository(temp_dir: Path) -> SymbolRepository:
    """Create a repository for testing."""
    db_path = get_default_db_path(temp_dir)
    return SymbolRepository(db_path)


class TestPythonParser:
    """Tests for the Python parser."""

    def test_parse_classes(self, sample_python_file: Path) -> None:
        """Test that classes are extracted correctly."""
        parser = PythonParser()
        result = parser.parse(sample_python_file)

        class_names = [s.name for s in result.symbols if s.type == SymbolType.CLASS]
        assert "OrderStatus" in class_names
        assert "Order" in class_names

    def test_parse_methods(self, sample_python_file: Path) -> None:
        """Test that methods are extracted correctly."""
        parser = PythonParser()
        result = parser.parse(sample_python_file)

        method_names = [s.name for s in result.symbols if s.type == SymbolType.METHOD]
        assert "__init__" in method_names
        assert "can_ship" in method_names

    def test_parse_functions(self, sample_python_file: Path) -> None:
        """Test that functions are extracted correctly."""
        parser = PythonParser()
        result = parser.parse(sample_python_file)

        func_names = [s.name for s in result.symbols if s.type == SymbolType.FUNCTION]
        assert "create_order" in func_names
        assert "process_order" in func_names

    def test_parse_edges(self, sample_python_file: Path) -> None:
        """Test that call edges are extracted correctly."""
        parser = PythonParser()
        result = parser.parse(sample_python_file)

        call_edges = [e for e in result.edges if e.call_type == EdgeType.CALL]
        callee_names = [e.callee_name for e in call_edges]

        # create_order calls Order() and OrderStatus.PENDING
        assert "Order" in callee_names
        # process_order calls order.can_ship()
        assert "order.can_ship" in callee_names


class TestRepository:
    """Tests for the symbol repository."""

    def test_insert_and_retrieve_symbol(self, repository: SymbolRepository) -> None:
        """Test inserting and retrieving a symbol."""
        symbol_id = repository.symbols.insert(
            name="my_func",
            qualified_name="test.my_func",
            file=Path("test.py"),
            line=10,
            symbol_type=SymbolType.FUNCTION,
            end_line=20,
        )

        symbol = repository.symbols.get_by_id(symbol_id)
        assert symbol.name == "my_func"
        assert symbol.qualified_name == "test.my_func"
        assert symbol.line == 10
        assert symbol.type == SymbolType.FUNCTION

    def test_find_symbols(self, repository: SymbolRepository) -> None:
        """Test fuzzy symbol search."""
        repository.symbols.insert(
            name="OrderService",
            qualified_name="services.OrderService",
            file=Path("services.py"),
            line=1,
            symbol_type=SymbolType.CLASS,
        )
        repository.symbols.insert(
            name="UserService",
            qualified_name="services.UserService",
            file=Path("services.py"),
            line=50,
            symbol_type=SymbolType.CLASS,
        )

        results = repository.symbols.find("Service")
        assert len(results) == 2

        results = repository.symbols.find("Order")
        assert len(results) == 1
        assert results[0].name == "OrderService"

    def test_get_callers_and_callees(self, repository: SymbolRepository) -> None:
        """Test retrieving callers and callees."""
        func_a = repository.symbols.insert(
            name="func_a",
            qualified_name="test.func_a",
            file=Path("test.py"),
            line=1,
            symbol_type=SymbolType.FUNCTION,
        )
        func_b = repository.symbols.insert(
            name="func_b",
            qualified_name="test.func_b",
            file=Path("test.py"),
            line=10,
            symbol_type=SymbolType.FUNCTION,
        )

        # func_a calls func_b
        repository.edges.insert(
            caller_id=func_a,
            callee_id=func_b,
            call_line=5,
            call_type=EdgeType.CALL,
        )

        # Check callees of func_a
        callees = repository.edges.get_callees(func_a)
        assert len(callees) == 1
        assert callees[0][0].name == "func_b"

        # Check callers of func_b
        callers = repository.edges.get_callers(func_b)
        assert len(callers) == 1
        assert callers[0][0].name == "func_a"


class TestIndexer:
    """Tests for the indexer."""

    def test_index_file(self, repository: SymbolRepository, sample_python_file: Path) -> None:
        """Test indexing a single file."""
        indexer = Indexer(repository)
        stats = indexer.index_file(sample_python_file)

        assert stats.files == 1
        assert stats.symbols > 0

        # Verify symbols are in the database
        symbols = repository.symbols.find("Order")
        assert len(symbols) > 0

    def test_index_directory(self, repository: SymbolRepository, temp_dir: Path) -> None:
        """Test indexing a directory."""
        # Create multiple Python files
        (temp_dir / "module_a.py").write_text("def func_a(): pass")
        (temp_dir / "module_b.py").write_text("def func_b(): pass")
        (temp_dir / "tests").mkdir()
        (temp_dir / "tests" / "test_a.py").write_text("def test_func(): pass")

        indexer = Indexer(repository)
        stats = indexer.index_directory(temp_dir)

        assert stats.files == 3  # All Python files indexed

    def test_exclude_patterns(self, repository: SymbolRepository, temp_dir: Path) -> None:
        """Test excluding files by pattern."""
        (temp_dir / "module.py").write_text("def func(): pass")
        (temp_dir / "tests").mkdir()
        (temp_dir / "tests" / "test_module.py").write_text("def test_func(): pass")

        indexer = Indexer(repository)
        stats = indexer.index_directory(temp_dir, exclude_patterns=["tests/*"])

        assert stats.files == 1  # Only module.py indexed
        assert stats.skipped == 1  # test_module.py skipped

    def test_default_excludes(self, repository: SymbolRepository, temp_dir: Path) -> None:
        """Test that all DEFAULT_EXCLUDES patterns correctly exclude directory contents."""
        from decoder.core.indexer import DEFAULT_EXCLUDES

        (temp_dir / "keep.py").write_text("def keep(): pass")

        # Create a directory and file for each non-wildcard pattern
        for pattern in DEFAULT_EXCLUDES:
            if "*" in pattern:
                continue  # Skip glob patterns for this test
            
            excluded_dir = temp_dir / pattern
            excluded_dir.mkdir(exist_ok=True)
            (excluded_dir / "file.py").write_text("def file(): pass")

        indexer = Indexer(repository)
        stats = indexer.index_directory(temp_dir)

        assert stats.files == 1  # Only keep.py indexed
        assert stats.skipped == len([p for p in DEFAULT_EXCLUDES if "*" not in p])


class TestTypedParameterResolution:
    """Tests for resolving method calls on typed parameters."""

    def test_typed_vars_extracted(self, temp_dir: Path) -> None:
        """Test that typed parameters are extracted from function signatures."""
        code = """
class UserService:
    def get_user(self, user_id: int) -> dict:
        return {"id": user_id}

def get_user_info(service: UserService, user_id: int) -> dict:
    return service.get_user(user_id)
"""
        file_path = temp_dir / "users.py"
        file_path.write_text(code)

        parser = PythonParser()
        result = parser.parse(file_path)

        # Check typed_vars were extracted
        assert result.typed_vars is not None
        assert len(result.typed_vars) >= 2

        # Find the service parameter
        service_var = next((tv for tv in result.typed_vars if tv.name == "service"), None)
        assert service_var is not None
        assert service_var.type_name == "UserService"

    def test_cross_file_typed_call_resolution(
        self, repository: SymbolRepository, temp_dir: Path
    ) -> None:
        """Test that method calls on typed parameters are resolved across files."""
        # Create a service file
        service_code = """
class OrderService:
    def create_order(self, item: str) -> dict:
        return {"item": item, "status": "created"}

    def cancel_order(self, order_id: int) -> bool:
        return True
"""
        (temp_dir / "services.py").write_text(service_code)

        # Create a routes file that uses the service
        routes_code = """
from services import OrderService

def create_order_route(service: OrderService, item: str) -> dict:
    return service.create_order(item)

def cancel_order_route(service: OrderService, order_id: int) -> dict:
    result = service.cancel_order(order_id)
    return {"cancelled": result}
"""
        (temp_dir / "routes.py").write_text(routes_code)

        # Index both files
        indexer = Indexer(repository)
        stats = indexer.index_directory(temp_dir)

        assert stats.files == 2
        assert stats.edges > 0

        # Verify the cross-file edges were created
        # Find OrderService.create_order
        create_order_symbols = repository.symbols.find("create_order")
        service_method = next(
            (s for s in create_order_symbols if "OrderService" in s.qualified_name), None
        )
        assert service_method is not None

        # Check that the route function calls the service method
        callers = repository.edges.get_callers(service_method.id)
        caller_names = [c[0].name for c in callers]
        assert "create_order_route" in caller_names

    def test_chained_typed_calls(self, repository: SymbolRepository, temp_dir: Path) -> None:
        """Test resolution of typed calls in a multi-layer architecture."""
        # Repository layer
        repo_code = """
class TodoRepository:
    def save(self, title: str) -> int:
        return 1

    def find_by_id(self, todo_id: int) -> dict:
        return {"id": todo_id}
"""
        (temp_dir / "repository.py").write_text(repo_code)

        # Service layer
        service_code = """
from repository import TodoRepository

class TodoService:
    def __init__(self, repo: TodoRepository):
        self._repo = repo

    def create_todo(self, title: str) -> int:
        return self._repo.save(title)
"""
        (temp_dir / "service.py").write_text(service_code)

        # Controller layer
        controller_code = """
from service import TodoService

def create_todo_handler(service: TodoService, title: str) -> dict:
    todo_id = service.create_todo(title)
    return {"id": todo_id}
"""
        (temp_dir / "controller.py").write_text(controller_code)

        # Index all files
        indexer = Indexer(repository)
        stats = indexer.index_directory(temp_dir)

        assert stats.files == 3

        # Verify controller -> service call
        service_methods = repository.symbols.find("create_todo")
        service_create = next(
            (s for s in service_methods if "TodoService" in s.qualified_name), None
        )
        assert service_create is not None

        callers = repository.edges.get_callers(service_create.id)
        caller_names = [c[0].name for c in callers]
        assert "create_todo_handler" in caller_names

        # Verify service -> repository call (via instance variable)
        repo_methods = repository.symbols.find("save")
        repo_save = next((s for s in repo_methods if "TodoRepository" in s.qualified_name), None)
        assert repo_save is not None

        repo_callers = repository.edges.get_callers(repo_save.id)
        repo_caller_names = [c[0].name for c in repo_callers]
        assert "create_todo" in repo_caller_names


class TestInstanceVariableTracking:
    """Tests for tracking types assigned to self.x in __init__."""

    def test_instance_var_types_extracted(self, temp_dir: Path) -> None:
        """Test that instance variable types are extracted from __init__ assignments."""
        code = """
class UserRepository:
    def find(self, user_id: int) -> dict:
        return {"id": user_id}

class UserService:
    def __init__(self, repo: UserRepository):
        self._repo = repo

    def get_user(self, user_id: int) -> dict:
        return self._repo.find(user_id)
"""
        file_path = temp_dir / "users.py"
        file_path.write_text(code)

        parser = PythonParser()
        result = parser.parse(file_path)

        # Check typed_vars includes instance variable
        assert result.typed_vars is not None

        # Find the self._repo instance variable
        instance_var = next((tv for tv in result.typed_vars if tv.name == "self._repo"), None)
        assert instance_var is not None
        assert instance_var.type_name == "UserRepository"
        # Should be scoped to the class, not __init__
        assert "UserService" in instance_var.scope_qualified_name
        assert "__init__" not in instance_var.scope_qualified_name

    def test_instance_var_method_resolution(
        self, repository: SymbolRepository, temp_dir: Path
    ) -> None:
        """Test that self._repo.method() resolves to the correct type's method."""
        code = """
class DataStore:
    def save(self, data: str) -> int:
        return 1

    def load(self, data_id: int) -> str:
        return "data"

class DataService:
    def __init__(self, store: DataStore):
        self._store = store

    def save_data(self, data: str) -> int:
        return self._store.save(data)

    def load_data(self, data_id: int) -> str:
        return self._store.load(data_id)
"""
        file_path = temp_dir / "data.py"
        file_path.write_text(code)

        indexer = Indexer(repository)
        indexer.index_file(file_path)

        # Find DataStore.save
        save_methods = repository.symbols.find("save")
        store_save = next((s for s in save_methods if "DataStore" in s.qualified_name), None)
        assert store_save is not None

        # Verify DataService.save_data calls DataStore.save
        callers = repository.edges.get_callers(store_save.id)
        caller_names = [c[0].name for c in callers]
        assert "save_data" in caller_names

        # Find DataStore.load
        load_methods = repository.symbols.find("load")
        store_load = next((s for s in load_methods if "DataStore" in s.qualified_name), None)
        assert store_load is not None

        # Verify DataService.load_data calls DataStore.load
        load_callers = repository.edges.get_callers(store_load.id)
        load_caller_names = [c[0].name for c in load_callers]
        assert "load_data" in load_caller_names
